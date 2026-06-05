/// Hyper-parameters for one Llama transformer (backbone or decoder).
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub embed_dim: usize,
    pub intermediate_dim: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
    /// Long-context RoPE scale factor (Llama 3.2).
    pub rope_scale_factor: f32,
    pub norm_eps: f64,
}

impl LlamaConfig {
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    pub fn n_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

/// Full model configuration (mirrors Python `ModelArgs`).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub backbone: LlamaConfig,
    pub decoder: LlamaConfig,
    pub text_vocab_size: usize,
    pub audio_vocab_size: usize,
    pub audio_num_codebooks: usize,
}

/// 8 B backbone + 300 M decoder, matching `MISO_TTS_8B_CONFIG`.
pub fn miso_tts_8b_config() -> ModelConfig {
    ModelConfig {
        backbone: LlamaConfig {
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            embed_dim: 4096,
            intermediate_dim: 14_336,
            max_seq_len: 2048,
            rope_base: 500_000.0,
            rope_scale_factor: 32.0,
            norm_eps: 1e-5,
        },
        decoder: LlamaConfig {
            num_layers: 8,
            num_heads: 24,
            num_kv_heads: 6,
            embed_dim: 1536,
            intermediate_dim: 6912,
            max_seq_len: 2048,
            rope_base: 500_000.0,
            rope_scale_factor: 32.0,
            norm_eps: 1e-5,
        },
        text_vocab_size: 128_256,
        audio_vocab_size: 2051,
        audio_num_codebooks: 32,
    }
}

pub const DEFAULT_REPO_ID: &str = "MisoLabs/MisoTTS";
pub const MIMI_REPO_ID: &str = "kyutai/moshiko-pytorch-bf16";
pub const MIMI_FILENAME: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";
pub const LLAMA_TOKENIZER_REPO_ID: &str = "meta-llama/Llama-3.2-1B";
