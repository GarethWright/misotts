use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};
use rand::Rng;

use crate::config::ModelConfig;
use crate::llama::Llama;

// ─── Sampling ─────────────────────────────────────────────────────────────────

/// Top-k + temperature sampling via the Gumbel-max trick.
/// Returns shape `(batch, 1)` of dtype I64.
pub fn sample_topk(
    logits: &Tensor,
    topk: usize,
    temperature: f32,
    rng: &mut impl Rng,
) -> Result<Tensor> {
    let logits = (logits / temperature as f64)?;

    // Get the k-th largest value as threshold.
    let (sorted_vals, _) = logits.sort_last_dim(false)?; // descending (asc=false)
    let kth = sorted_vals.narrow(1, topk - 1, 1)?; // (batch, 1)
    // Mask tokens below threshold to -inf.
    let mask = logits.ge(&kth.broadcast_as(logits.shape())?)?; // u8
    let neg_inf_t = (Tensor::ones_like(&logits)? * f64::NEG_INFINITY)?;
    let masked = mask.where_cond(&logits, &neg_inf_t)?;

    let probs = candle_nn::ops::softmax_last_dim(&masked)?;

    // Gumbel-max: argmax(probs / exponential(1))
    let (batch, vocab) = probs.dims2()?;
    let exp_data: Vec<f32> = (0..batch * vocab)
        .map(|_| {
            let u: f32 = rng.gen_range(1e-10f32..1.0f32);
            -u.ln() // exponential(1) sample
        })
        .collect();
    let exp_t = Tensor::from_vec(exp_data, (batch, vocab), probs.device())?;
    let ratio = probs.broadcast_div(&exp_t)?;
    Ok(ratio.argmax_keepdim(1)?.to_dtype(DType::I64)?)
}

// ─── Causal mask helpers ──────────────────────────────────────────────────────

/// Lower-triangular boolean (u8) mask of shape `(n, n)`.
pub fn causal_mask(n: usize, device: &Device) -> Result<Tensor> {
    let data: Vec<u8> = (0..n)
        .flat_map(|r| (0..n).map(move |c| (c <= r) as u8))
        .collect();
    Ok(Tensor::from_vec(data, (n, n), device)?)
}

/// Index `mask` `(max_n, max_n)` at `positions` `(batch, seq)` → `(batch, seq, max_n)`.
pub fn index_causal_mask(mask: &Tensor, positions: &Tensor) -> Result<Tensor> {
    let (batch, seq) = positions.dims2()?;
    let max_n = mask.dim(1)?;
    let flat = positions.flatten_all()?;
    let rows = mask.index_select(&flat, 0)?; // (batch*seq, max_n)
    Ok(rows.reshape((batch, seq, max_n))?)
}

// ─── Model ────────────────────────────────────────────────────────────────────

pub struct Model {
    pub backbone: Llama,
    pub decoder: Llama,
    text_emb: Embedding,
    audio_emb: Embedding,
    projection: Linear,
    c0_head: Linear,
    /// `(audio_num_codebooks-1, decoder_dim, audio_vocab_size)`
    audio_head: Tensor,
    pub backbone_causal_mask: Tensor,
    pub decoder_causal_mask: Tensor,
    pub config: ModelConfig,
}

impl Model {
    pub fn load(cfg: ModelConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let backbone = Llama::load(&cfg.backbone, vb.pp("backbone"), device)?;
        let decoder = Llama::load(&cfg.decoder, vb.pp("decoder"), device)?;

        let text_emb =
            embedding(cfg.text_vocab_size, cfg.backbone.embed_dim, vb.pp("text_embeddings"))?;
        let audio_emb = embedding(
            cfg.audio_vocab_size * cfg.audio_num_codebooks,
            cfg.backbone.embed_dim,
            vb.pp("audio_embeddings"),
        )?;
        let projection =
            linear_no_bias(cfg.backbone.embed_dim, cfg.decoder.embed_dim, vb.pp("projection"))?;
        let c0_head =
            linear_no_bias(cfg.backbone.embed_dim, cfg.audio_vocab_size, vb.pp("codebook0_head"))?;
        let audio_head = vb.get(
            (
                cfg.audio_num_codebooks - 1,
                cfg.decoder.embed_dim,
                cfg.audio_vocab_size,
            ),
            "audio_head",
        )?;

        let backbone_causal_mask = causal_mask(cfg.backbone.max_seq_len, device)?;
        let decoder_causal_mask = causal_mask(cfg.audio_num_codebooks, device)?;

        Ok(Self {
            backbone,
            decoder,
            text_emb,
            audio_emb,
            projection,
            c0_head,
            audio_head,
            backbone_causal_mask,
            decoder_causal_mask,
            config: cfg,
        })
    }

    fn embed_audio(&self, codebook: usize, tokens: &Tensor) -> Result<Tensor> {
        // Offset each codebook's tokens into the shared embedding table.
        let offset = (codebook * self.config.audio_vocab_size) as f64;
        let shifted = (tokens.to_dtype(DType::F32)? + offset)?.to_dtype(DType::I64)?;
        Ok(self.audio_emb.forward(&shifted)?)
    }

    /// `tokens`: `(batch, seq, K+1)` — K audio codebooks + 1 text column.
    /// Returns `(batch, seq, K+1, embed_dim)`.
    fn embed_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        let (b, s, _) = tokens.dims3()?;
        let k = self.config.audio_num_codebooks;

        let text_ids = tokens.narrow(2, k, 1)?.squeeze(2)?; // (b, s)
        let text_emb = self.text_emb.forward(&text_ids)?.unsqueeze(2)?; // (b, s, 1, d)

        // Build per-codebook offsets: shape (1, 1, k)
        let offsets_data: Vec<f64> = (0..k)
            .map(|i| (i * self.config.audio_vocab_size) as f64)
            .collect();
        let offsets = Tensor::from_vec(
            offsets_data.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
            k,
            tokens.device(),
        )?
        .reshape((1, 1, k))?;

        let audio_f = tokens
            .narrow(2, 0, k)?
            .to_dtype(DType::F32)?;
        let shifted = (audio_f + offsets)?.to_dtype(DType::I64)?; // (b, s, k)
        let flat = shifted.flatten(0, 2)?; // (b*s*k,)
        let emb_flat = self.audio_emb.forward(&flat)?; // (b*s*k, d)
        let d = emb_flat.dim(1)?;
        let audio_emb = emb_flat.reshape((b, s, k, d))?;

        Ok(Tensor::cat(&[&audio_emb, &text_emb], 2)?)
    }

    /// Generate one frame of `audio_num_codebooks` token IDs.
    /// Returns `(batch, audio_num_codebooks)`.
    pub fn generate_frame(
        &mut self,
        tokens: &Tensor,
        tokens_mask: &Tensor,
        positions: &[u32],
        temperature: f32,
        topk: usize,
        rng: &mut impl Rng,
    ) -> Result<Tensor> {
        let (b, _s, _) = tokens.dims3()?;
        let n = positions.len();

        // Backbone causal mask for the current positions.
        let pos_t = Tensor::from_slice(positions, n, tokens.device())?
            .unsqueeze(0)?
            .expand((b, n))?;
        let backbone_mask = index_causal_mask(&self.backbone_causal_mask, &pos_t)?;

        // Embed and sum across the codebook+text dimension.
        let embeds = self.embed_tokens(tokens)?.to_dtype(DType::F32)?;
        let mask_f = tokens_mask
            .to_dtype(DType::F32)?
            .unsqueeze(3)?
            .broadcast_as(embeds.shape())?;
        let h = (embeds * mask_f)?.sum(2)?; // (b, seq, embed_dim)

        let h = self
            .backbone
            .forward(&h, positions, Some(&backbone_mask))?
            .to_dtype(DType::F32)?;

        let seq_len = h.dim(1)?;
        let last_h = h.narrow(1, seq_len - 1, 1)?; // (b, 1, d_bb)

        // Sample codebook-0.
        let c0_logits = self.c0_head.forward(&last_h.squeeze(1)?)?; // (b, vocab)
        let c0_sample = sample_topk(&c0_logits, topk, temperature, rng)?; // (b, 1)
        let c0_emb = self.embed_audio(0, &c0_sample)?; // (b, 1, d_bb)

        // Decoder input: concatenate backbone last hidden state + c0 embedding.
        let mut curr_h = Tensor::cat(&[&last_h, &c0_emb], 1)?; // (b, 2, d_bb)
        let mut curr_sample = c0_sample.clone();
        let mut dec_positions: Vec<u32> = (0..curr_h.dim(1)? as u32).collect();

        self.decoder.reset_caches();

        for i in 1..self.config.audio_num_codebooks {
            let dec_len = curr_h.dim(1)?;
            let dec_pos_t = Tensor::from_slice(
                &dec_positions[..dec_len],
                dec_len,
                curr_h.device(),
            )?
            .unsqueeze(0)?
            .expand((b, dec_len))?;
            let dec_mask = index_causal_mask(&self.decoder_causal_mask, &dec_pos_t)?;

            let proj = self.projection.forward(&curr_h)?; // (b, ?, d_dec)
            let dec_h = self
                .decoder
                .forward(&proj, &dec_positions[..dec_len], Some(&dec_mask))?
                .to_dtype(DType::F32)?;

            // Logits via audio_head[i-1]: shape (d_dec, vocab_size)
            let head_i = self.audio_head.narrow(0, i - 1, 1)?.squeeze(0)?;
            let last_dec = dec_h.narrow(1, dec_len - 1, 1)?.squeeze(1)?; // (b, d_dec)
            let ci_logits = last_dec.matmul(&head_i)?; // (b, vocab)
            let ci_sample = sample_topk(&ci_logits, topk, temperature, rng)?; // (b, 1)
            let ci_emb = self.embed_audio(i, &ci_sample)?;

            curr_h = ci_emb;
            curr_sample = Tensor::cat(&[&curr_sample, &ci_sample], 1)?;
            let next_pos = *dec_positions.last().unwrap() + 1;
            dec_positions = vec![next_pos];
        }

        Ok(curr_sample) // (batch, K)
    }

    pub fn reset_caches(&mut self) {
        self.backbone.reset_caches();
        self.decoder.reset_caches();
    }
}
