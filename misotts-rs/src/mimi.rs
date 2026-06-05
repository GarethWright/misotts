/// Wrapper around kyutai-labs/moshi's Mimi neural audio codec.
///
/// Mimi encodes 24 kHz mono audio into 32 RVQ codebook streams and decodes
/// them back.  Weight file:
///   repo  = "kyutai/moshiko-pytorch-bf16"
///   file  = "tokenizer-e351c8d8-checkpoint125.safetensors"
use anyhow::{Context, Result};
// The moshi crate re-exports candle as `candle`; its Device/Tensor types are
// identical to ours because Cargo unifies the semver-compatible crate.
use moshi::candle::{DType, Device, Tensor};
use moshi::candle_nn::VarBuilder;

pub use moshi::candle as candle_core;

pub const SAMPLE_RATE: u32 = 24_000;
pub const NUM_CODEBOOKS: usize = 32;

pub struct MimiCodec {
    inner: moshi::mimi::Mimi,
}

impl MimiCodec {
    /// Load Mimi weights from a safetensors file.
    pub fn load(weights_path: &std::path::Path, device: &Device) -> Result<Self> {
        let inner = moshi::mimi::load(
            weights_path
                .to_str()
                .context("non-UTF-8 path for Mimi weights")?,
            Some(NUM_CODEBOOKS),
            device,
        )
        .context("loading Mimi codec")?;
        Ok(Self { inner })
    }

    /// Encode a mono waveform → codebook tokens.
    ///
    /// Input:  `(1, 1, num_samples)` f32 at 24 kHz.
    /// Output: `(1, NUM_CODEBOOKS, num_frames)` i64.
    pub fn encode(&mut self, audio: &Tensor) -> Result<Tensor> {
        self.inner
            .encode(audio)
            .context("Mimi encode")
    }

    /// Decode codebook tokens → mono waveform.
    ///
    /// Input:  `(1, NUM_CODEBOOKS, num_frames)` i64.
    /// Output: `(1, 1, num_samples)` f32.
    pub fn decode(&mut self, codes: &Tensor) -> Result<Tensor> {
        self.inner
            .decode(codes)
            .context("Mimi decode")
    }
}
