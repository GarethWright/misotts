use anyhow::{Context, Result};
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::api::tokio::ApiBuilder;
use std::path::PathBuf;
use tracing::info;

// Use moshi's re-exported candle so Device/Tensor types unify with Mimi.
use misotts::mimi::candle_core::{DType, Device};

use misotts::audio::{load_wav, resample, save_wav};
use misotts::config::{
    miso_tts_8b_config, DEFAULT_REPO_ID, LLAMA_TOKENIZER_REPO_ID, MIMI_FILENAME, MIMI_REPO_ID,
};
use misotts::generator::{Generator, Segment};
use misotts::mimi::{MimiCodec, SAMPLE_RATE};
use misotts::model::Model;
use misotts::tokenizer::TextTokenizer;

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "misotts", about = "Miso TTS — Rust native inference")]
struct Args {
    /// Text to synthesise.
    text: String,

    /// Speaker ID (0 or 1 for a two-speaker conversation).
    #[arg(long, default_value_t = 0)]
    speaker: u32,

    /// Output WAV path.
    #[arg(long, default_value = "output.wav")]
    output: PathBuf,

    /// Audio file to use as acoustic context (24 kHz mono WAV recommended).
    #[arg(long)]
    context_audio: Option<PathBuf>,

    /// Transcript for the context audio.
    #[arg(long, default_value = "")]
    context_text: String,

    /// Speaker ID for the context audio.
    #[arg(long, default_value_t = 0)]
    context_speaker: u32,

    /// Maximum output duration in milliseconds.
    #[arg(long, default_value_t = 90_000.0)]
    max_audio_ms: f32,

    /// Sampling temperature (higher = more varied).
    #[arg(long, default_value_t = 0.9)]
    temperature: f32,

    /// Top-k vocabulary size for sampling.
    #[arg(long, default_value_t = 50)]
    topk: usize,

    /// HuggingFace repo ID for the TTS model weights.
    #[arg(long)]
    model_repo: Option<String>,

    /// Local path to the TTS model .safetensors file (skips HF download).
    #[arg(long)]
    model_path: Option<PathBuf>,

    /// Use CUDA device 0.
    #[arg(long)]
    cuda: bool,

    /// Weight / activation precision: bf16 (default on GPU), f16, or f32.
    /// bf16 halves VRAM and fits a 3090/4090; f32 needs 40 GB+.
    #[arg(long, value_name = "DTYPE")]
    dtype: Option<String>,

    /// HuggingFace access token (for gated models such as the Llama tokenizer).
    /// Falls back to the HF_TOKEN / HUGGING_FACE_HUB_TOKEN env vars, then
    /// ~/.cache/huggingface/token written by `huggingface-cli login`.
    #[arg(long, value_name = "TOKEN")]
    hf_token: Option<String>,
}

// ─── Main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("misotts=info".parse()?),
        )
        .init();

    let args = Args::parse();

    let device = if args.cuda {
        Device::new_cuda(0).context("failed to initialise CUDA device")?
    } else {
        Device::Cpu
    };

    // Default: BF16 on GPU (fits 24 GB cards), F32 on CPU.
    let dtype = match args.dtype.as_deref() {
        Some("bf16") => DType::BF16,
        Some("f16")  => DType::F16,
        Some("f32")  => DType::F32,
        None => match &device {
            Device::Cpu => DType::F32,
            _ => DType::BF16,
        },
        Some(other) => anyhow::bail!("unknown dtype {other:?} — use bf16, f16, or f32"),
    };
    info!("Using dtype: {dtype:?}");

    // Token priority: --hf-token flag > HF_TOKEN env > HUGGING_FACE_HUB_TOKEN env > ~/.cache/huggingface/token
    let hf_token = args
        .hf_token
        .clone()
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());

    let api = if hf_token.is_some() {
        ApiBuilder::new().with_token(hf_token).build()?
    } else {
        ApiBuilder::new().build()?
    };

    // ── Locate / download weights ─────────────────────────────────────────────
    let model_path = if let Some(p) = &args.model_path {
        p.clone()
    } else {
        let repo_id = args.model_repo.as_deref().unwrap_or(DEFAULT_REPO_ID);
        info!("Downloading TTS model from {repo_id}");
        PathBuf::from(api.model(repo_id.to_string()).get("model.safetensors").await?)
    };

    info!("Downloading Mimi codec weights");
    let mimi_path =
        PathBuf::from(api.model(MIMI_REPO_ID.to_string()).get(MIMI_FILENAME).await?);

    info!("Downloading tokenizer");
    let tokenizer_path = PathBuf::from(
        api.model(LLAMA_TOKENIZER_REPO_ID.to_string())
            .get("tokenizer.json")
            .await?,
    );

    // ── Load components ───────────────────────────────────────────────────────
    info!("Loading tokenizer");
    let tokenizer = TextTokenizer::from_file(&tokenizer_path)?;

    info!("Loading Mimi codec");
    let mimi = MimiCodec::load(&mimi_path, &device)?;

    info!("Loading TTS model weights (may take a while for 8 B parameters)");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&model_path], dtype, &device)?
    };
    let model = Model::load(miso_tts_8b_config(), vb, &device)?;

    let mut generator = Generator::new(model, tokenizer, mimi, device);

    // ── Prepare context ───────────────────────────────────────────────────────
    let context: Vec<Segment> = if let Some(ctx_path) = &args.context_audio {
        let (raw, sr) = load_wav(ctx_path)?;
        let audio = if sr != SAMPLE_RATE {
            info!("Resampling context from {sr} Hz → {SAMPLE_RATE} Hz");
            resample(&raw, sr, SAMPLE_RATE)?
        } else {
            raw
        };
        vec![Segment {
            speaker: args.context_speaker,
            text: args.context_text.clone(),
            audio,
        }]
    } else {
        vec![]
    };

    // ── Generate ──────────────────────────────────────────────────────────────
    info!("Generating: {:?}", args.text);
    let audio = generator.generate(
        &args.text,
        args.speaker,
        &context,
        args.max_audio_ms,
        args.temperature,
        args.topk,
    )?;

    info!(
        "Generated {} samples ({:.2} s) — writing {}",
        audio.len(),
        audio.len() as f32 / SAMPLE_RATE as f32,
        args.output.display()
    );
    save_wav(&args.output, &audio, SAMPLE_RATE)?;

    Ok(())
}
