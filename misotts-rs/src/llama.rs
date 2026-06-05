use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::LlamaConfig;

// ─── RMS Norm ────────────────────────────────────────────────────────────────

pub struct RmsNorm {
    scale: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get(size, "scale")?;
        Ok(Self { scale, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let rms = (x_f32
            .sqr()?
            .mean_keepdim(D::Minus1)?
            + self.eps)?
            .sqrt()?;
        let normed = x_f32.broadcast_div(&rms)?;
        Ok(normed
            .broadcast_mul(&self.scale.to_dtype(DType::F32)?)?
            .to_dtype(x.dtype())?)
    }
}

// ─── Rotary Position Embeddings (scaled, Llama 3.2) ──────────────────────────

pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

fn scaled_rope_freqs(head_dim: usize, rope_base: f32, scale_factor: f32) -> Vec<f32> {
    let low_freq_factor: f32 = 1.0;
    let high_freq_factor: f32 = 4.0;
    let orig_ctx: f32 = 8192.0;
    let low_wavelen = orig_ctx / low_freq_factor;
    let high_wavelen = orig_ctx / high_freq_factor;
    let half = head_dim / 2;
    (0..half)
        .map(|i| {
            let theta = rope_base.powf(-(2.0 * i as f32) / head_dim as f32);
            let wavelen = 2.0 * std::f32::consts::PI / theta;
            if wavelen > low_wavelen {
                theta / scale_factor
            } else if wavelen <= high_wavelen {
                theta
            } else {
                let smooth = (orig_ctx / wavelen - low_freq_factor)
                    / (high_freq_factor - low_freq_factor);
                (1.0 - smooth) * theta / scale_factor + smooth * theta
            }
        })
        .collect()
}

impl RotaryEmbedding {
    pub fn new(cfg: &LlamaConfig, device: &Device) -> Result<Self> {
        let freqs = scaled_rope_freqs(cfg.head_dim(), cfg.rope_base, cfg.rope_scale_factor);
        let half = freqs.len();
        let mut cos_data = Vec::with_capacity(cfg.max_seq_len * half);
        let mut sin_data = Vec::with_capacity(cfg.max_seq_len * half);
        for pos in 0..cfg.max_seq_len {
            for &f in &freqs {
                let angle = pos as f32 * f;
                cos_data.push(angle.cos());
                sin_data.push(angle.sin());
            }
        }
        let cos = Tensor::from_vec(cos_data, (cfg.max_seq_len, half), device)?;
        let sin = Tensor::from_vec(sin_data, (cfg.max_seq_len, half), device)?;
        Ok(Self { cos, sin })
    }

    /// Apply RoPE to `x: (batch, heads, seq, head_dim)`.
    /// `positions`: flat slice of `seq` position indices.
    pub fn apply(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let idx = Tensor::from_slice(positions, positions.len(), x.device())?;
        let cos_rows = self.cos.index_select(&idx, 0)?; // (seq, half)
        let sin_rows = self.sin.index_select(&idx, 0)?;
        // Duplicate halves so each covers full head_dim after broadcast
        let cos = Tensor::cat(&[&cos_rows, &cos_rows], 1)?
            .unsqueeze(0)?
            .unsqueeze(0)?; // (1, 1, seq, head_dim)
        let sin = Tensor::cat(&[&sin_rows, &sin_rows], 1)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        let head_dim = x.dim(D::Minus1)?;
        let half = head_dim / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let x_rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        Ok((x.broadcast_mul(&cos)? + x_rot.broadcast_mul(&sin)?)?)
    }
}

// ─── KV Cache ────────────────────────────────────────────────────────────────

pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn update(&mut self, new_k: Tensor, new_v: Tensor) -> Result<(Tensor, Tensor)> {
        let k = match &self.k {
            None => new_k,
            Some(prev) => Tensor::cat(&[prev, &new_k], 2)?,
        };
        let v = match &self.v {
            None => new_v,
            Some(prev) => Tensor::cat(&[prev, &new_v], 2)?,
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

// ─── Attention ────────────────────────────────────────────────────────────────

pub struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_groups: usize,
    pub kv_cache: KvCache,
    rope: std::sync::Arc<RotaryEmbedding>,
}

fn repeat_kv(x: &Tensor, n_groups: usize) -> Result<Tensor> {
    if n_groups == 1 {
        return Ok(x.clone());
    }
    let (b, kv_h, seq, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .expand((b, kv_h, n_groups, seq, d))?
        .reshape((b, kv_h * n_groups, seq, d))?)
}

impl Attention {
    pub fn load(
        cfg: &LlamaConfig,
        rope: std::sync::Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hd = cfg.head_dim();
        let q = linear_no_bias(cfg.embed_dim, cfg.num_heads * hd, vb.pp("q_proj"))?;
        let k = linear_no_bias(cfg.embed_dim, cfg.num_kv_heads * hd, vb.pp("k_proj"))?;
        let v = linear_no_bias(cfg.embed_dim, cfg.num_kv_heads * hd, vb.pp("v_proj"))?;
        let o = linear_no_bias(cfg.num_heads * hd, cfg.embed_dim, vb.pp("output_proj"))?;
        Ok(Self {
            q,
            k,
            v,
            o,
            n_heads: cfg.num_heads,
            n_kv_heads: cfg.num_kv_heads,
            head_dim: hd,
            n_groups: cfg.n_groups(),
            kv_cache: KvCache::new(),
            rope,
        })
    }

    /// `mask`: `(batch, q_len, kv_len_total)` u8 tensor — 1 = can attend.
    pub fn forward(
        &mut self,
        x: &Tensor,
        positions: &[u32],
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, q_len, _) = x.dims3()?;

        let q = self
            .q
            .forward(x)?
            .reshape((b, q_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k
            .forward(x)?
            .reshape((b, q_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v
            .forward(x)?
            .reshape((b, q_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.rope.apply(&q, positions)?;
        let k = self.rope.apply(&k, positions)?;

        let (k, v) = self.kv_cache.update(k, v)?;
        let kv_len = k.dim(2)?;

        let k = repeat_kv(&k, self.n_groups)?;
        let v = repeat_kv(&v, self.n_groups)?;

        let scale = (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? / scale)?; // (b, heads, q, kv)

        let scores = if let Some(m) = mask {
            let m = m.narrow(2, 0, kv_len)?; // slice to actual kv length
            // m is u8: 1=attend→additive 0.0, 0=ignore→additive -1e9
            let m_f = m.to_dtype(scores.dtype())?;
            let additive = ((Tensor::ones_like(&m_f)? - &m_f)? * -1e9)?;
            (scores + additive.unsqueeze(1)?)?
        } else {
            scores
        };

        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, q_len, self.n_heads * self.head_dim))?;
        Ok(self.o.forward(&out)?)
    }
}

// ─── SwiGLU MLP ──────────────────────────────────────────────────────────────

pub struct Mlp {
    gate: Linear,
    down: Linear,
    up: Linear,
}

impl Mlp {
    pub fn load(embed_dim: usize, intermediate_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: linear_no_bias(embed_dim, intermediate_dim, vb.pp("w1"))?,
            down: linear_no_bias(intermediate_dim, embed_dim, vb.pp("w2"))?,
            up: linear_no_bias(embed_dim, intermediate_dim, vb.pp("w3"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        let activated = (gate * up)?;
        Ok(self.down.forward(&activated)?)
    }
}

// ─── Transformer Block ───────────────────────────────────────────────────────

pub struct TransformerBlock {
    pub attn: Attention,
    pub mlp: Mlp,
    sa_norm: RmsNorm,
    mlp_norm: RmsNorm,
}

impl TransformerBlock {
    pub fn load(
        cfg: &LlamaConfig,
        rope: std::sync::Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(cfg, rope, vb.pp("attn"))?,
            mlp: Mlp::load(cfg.embed_dim, cfg.intermediate_dim, vb.pp("mlp"))?,
            sa_norm: RmsNorm::load(cfg.embed_dim, cfg.norm_eps, vb.pp("sa_norm"))?,
            mlp_norm: RmsNorm::load(cfg.embed_dim, cfg.norm_eps, vb.pp("mlp_norm"))?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        positions: &[u32],
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let x = (self.attn.forward(&self.sa_norm.forward(x)?, positions, mask)? + x)?;
        let x = (self.mlp.forward(&self.mlp_norm.forward(&x)?)? + &x)?;
        Ok(x)
    }
}

// ─── Full Llama Transformer ───────────────────────────────────────────────────

pub struct Llama {
    pub layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    pub rope: std::sync::Arc<RotaryEmbedding>,
}

impl Llama {
    pub fn load(cfg: &LlamaConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let rope = std::sync::Arc::new(RotaryEmbedding::new(cfg, device)?);
        let mut layers = Vec::with_capacity(cfg.num_layers);
        let layers_vb = vb.pp("layers");
        for i in 0..cfg.num_layers {
            layers.push(TransformerBlock::load(
                cfg,
                std::sync::Arc::clone(&rope),
                layers_vb.pp(i),
            )?);
        }
        Ok(Self {
            layers,
            norm: RmsNorm::load(cfg.embed_dim, cfg.norm_eps, vb.pp("norm"))?,
            rope,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        positions: &[u32],
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x, positions, mask)?;
        }
        self.norm.forward(&x)
    }

    pub fn reset_caches(&mut self) {
        for layer in &mut self.layers {
            layer.attn.kv_cache.reset();
        }
    }
}
