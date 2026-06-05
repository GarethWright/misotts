use anyhow::{bail, Result};
use candle_core::{DType, Tensor};
use rand::{rngs::StdRng, SeedableRng};

use crate::mimi::{MimiCodec, NUM_CODEBOOKS, SAMPLE_RATE};
use crate::model::Model;
use crate::tokenizer::TextTokenizer;

/// One segment of prior conversational audio used as acoustic context.
pub struct Segment {
    pub speaker: u32,
    pub text: String,
    /// Mono waveform at `SAMPLE_RATE` Hz.
    pub audio: Vec<f32>,
}

pub struct Generator {
    model: Model,
    tokenizer: TextTokenizer,
    mimi: MimiCodec,
    rng: StdRng,
    device: candle_core::Device,
    frame_size: usize, // audio_num_codebooks + 1
}

impl Generator {
    pub fn new(
        model: Model,
        tokenizer: TextTokenizer,
        mimi: MimiCodec,
        device: candle_core::Device,
    ) -> Self {
        let frame_size = model.config.audio_num_codebooks + 1;
        Self {
            model,
            tokenizer,
            mimi,
            rng: StdRng::from_entropy(),
            device,
            frame_size,
        }
    }

    // ── Tokenisation helpers ──────────────────────────────────────────────────

    fn tokenise_text(&self, text: &str, speaker: u32) -> Result<(Tensor, Tensor)> {
        let ids = self.tokenizer.encode(text, speaker)?;
        let n = ids.len();
        let fs = self.frame_size;
        let mut tok_data = vec![0i64; n * fs];
        let mut msk_data = vec![0u8; n * fs];
        for (row, &id) in ids.iter().enumerate() {
            tok_data[row * fs + (fs - 1)] = id as i64;
            msk_data[row * fs + (fs - 1)] = 1;
        }
        let tokens = Tensor::from_vec(tok_data, (n, fs), &self.device)?;
        let mask = Tensor::from_vec(msk_data, (n, fs), &self.device)?;
        Ok((tokens, mask))
    }

    fn tokenise_audio(&mut self, audio: &[f32]) -> Result<(Tensor, Tensor)> {
        let audio_t = Tensor::from_slice(audio, audio.len(), &self.device)?
            .reshape((1usize, 1usize, audio.len()))?;
        let codes = self.mimi.encode(&audio_t)?; // (1, K, T)
        let k = NUM_CODEBOOKS;
        let t = codes.dim(2)?;

        // Append an all-zero EOS frame.
        let eos = Tensor::zeros((1usize, k, 1usize), DType::I64, &self.device)?;
        let codes = Tensor::cat(&[&codes.to_dtype(DType::I64)?, &eos], 2)?; // (1, K, T+1)
        let t1 = t + 1;
        let fs = self.frame_size;

        let codes_2d = codes.squeeze(0)?.transpose(0, 1)?; // (T+1, K)
        let codes_vec: Vec<i64> = codes_2d.flatten_all()?.to_vec1()?;
        let mut tok_data = vec![0i64; t1 * fs];
        let mut msk_data = vec![0u8; t1 * fs];
        for row in 0..t1 {
            for col in 0..k {
                tok_data[row * fs + col] = codes_vec[row * k + col];
                msk_data[row * fs + col] = 1;
            }
        }
        let tokens = Tensor::from_vec(tok_data, (t1, fs), &self.device)?;
        let mask = Tensor::from_vec(msk_data, (t1, fs), &self.device)?;
        Ok((tokens, mask))
    }

    fn tokenise_segment(&mut self, seg: &Segment) -> Result<(Tensor, Tensor)> {
        let (tt, tm) = self.tokenise_text(&seg.text, seg.speaker)?;
        let (at, am) = self.tokenise_audio(&seg.audio)?;
        Ok((Tensor::cat(&[&tt, &at], 0)?, Tensor::cat(&[&tm, &am], 0)?))
    }

    // ── Generation ───────────────────────────────────────────────────────────

    /// Generate speech for `text` / `speaker`.
    ///
    /// Returns mono f32 waveform at `SAMPLE_RATE` Hz.
    pub fn generate(
        &mut self,
        text: &str,
        speaker: u32,
        context: &[Segment],
        max_audio_ms: f32,
        temperature: f32,
        topk: usize,
    ) -> Result<Vec<f32>> {
        self.model.reset_caches();

        let max_frames = (max_audio_ms / 80.0) as usize; // 80 ms per frame
        let max_seq = 2048usize;
        let max_context = max_seq - max_frames;

        let mut all_tokens: Vec<Tensor> = Vec::new();
        let mut all_masks: Vec<Tensor> = Vec::new();

        for seg in context {
            // We need to tokenise segments which requires &mut self for mimi.
            // Clone the audio/text to avoid borrow conflicts.
            let (t, m) = {
                let audio = seg.audio.clone();
                let text_s = seg.text.clone();
                let spk = seg.speaker;
                let (tt, tm) = self.tokenise_text(&text_s, spk)?;
                let (at, am) = self.tokenise_audio(&audio)?;
                (Tensor::cat(&[&tt, &at], 0)?, Tensor::cat(&[&tm, &am], 0)?)
            };
            all_tokens.push(t);
            all_masks.push(m);
        }
        let (gen_t, gen_m) = self.tokenise_text(text, speaker)?;
        all_tokens.push(gen_t);
        all_masks.push(gen_m);

        let prompt_tokens = Tensor::cat(&all_tokens, 0)?;
        let prompt_mask = Tensor::cat(&all_masks, 0)?;
        let prompt_len = prompt_tokens.dim(0)?;

        if prompt_len >= max_context {
            bail!(
                "prompt ({prompt_len} tokens) exceeds max context ({max_context})"
            );
        }

        let mut curr_tokens = prompt_tokens.unsqueeze(0)?; // (1, L, fs)
        let mut curr_mask = prompt_mask.unsqueeze(0)?;
        let mut curr_pos: Vec<u32> = (0..prompt_len as u32).collect();
        let mut samples: Vec<Tensor> = Vec::new();

        let k = NUM_CODEBOOKS;

        for _ in 0..max_frames {
            // Reborrow rng separately to satisfy borrow checker.
            let rng = &mut self.rng;
            let frame = self.model.generate_frame(
                &curr_tokens,
                &curr_mask,
                &curr_pos,
                temperature,
                topk,
                rng,
            )?; // (1, K)

            let frame_sum: i64 = frame.sum_all()?.to_scalar()?;
            if frame_sum == 0 {
                break;
            }
            samples.push(frame.clone());

            // Build next step: audio frame (K columns) + zero text column.
            let zeros_text = Tensor::zeros((1usize, 1usize), DType::I64, &self.device)?;
            curr_tokens =
                Tensor::cat(&[&frame, &zeros_text], 1)?.unsqueeze(1)?; // (1,1,K+1)
            let ones_audio =
                Tensor::ones((1usize, k), DType::U8, &self.device)?;
            let zeros_text_m =
                Tensor::zeros((1usize, 1usize), DType::U8, &self.device)?;
            curr_mask =
                Tensor::cat(&[&ones_audio, &zeros_text_m], 1)?.unsqueeze(1)?;
            let next_pos = *curr_pos.last().unwrap() + 1;
            curr_pos = vec![next_pos];
        }

        if samples.is_empty() {
            return Ok(vec![]);
        }

        // Stack samples: (num_frames, 1, K) → permute → (1, K, num_frames).
        let stacked = Tensor::cat(&samples, 0)?
            .unsqueeze(1)? // (F, 1, K)
            .permute((1, 2, 0))?  // (1, K, F)
            .contiguous()?;

        let waveform = self.mimi.decode(&stacked)?; // (1, 1, num_samples)
        let flat: Vec<f32> = waveform.squeeze(0)?.squeeze(0)?.to_vec1()?;
        Ok(flat)
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }
}
