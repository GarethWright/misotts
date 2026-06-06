<div align="center">

<img src="images/repo_banner.png" alt="Miso TTS 8B" width="100%">

# Miso TTS 8B

### State-of-the-Art Text-to-Speech Model

<p>
  <a href="https://misolabs.ai"><img alt="Website" src="https://img.shields.io/badge/Website-misolabs.ai-black?style=for-the-badge"></a>
  <a href="https://huggingface.co/MisoLabs/MisoTTS"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-MisoTTS-yellow?style=for-the-badge"></a>
  <a href="https://github.com/MisoLabsAI"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-MisoLabsAI-181717?style=for-the-badge&logo=github&labelColor=555555"></a>
  <a href="https://x.com/MisoLabsAI"><img alt="X" src="https://img.shields.io/badge/-MisoLabsAI-181717?style=for-the-badge&logo=x&labelColor=555555"></a>
</p>

<p>
  <a href="#quickstart">Quickstart</a> |
  <a href="#model-introduction">Model Introduction</a> |
  <a href="#model-summary">Model Summary</a> |
  <a href="#usage">Usage</a> |
  <a href="#rust">Rust</a> |
  <a href="#safety">Safety</a>
</p>

</div>

---

## Quickstart

To quickly try the model, you can use the demo hosted on our [landing page](https://misolabs.ai)
at misolabs.ai. To try it locally, follow the instructions below.

If you do not have `uv` installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone the repository and create the environment:

```bash
git clone https://github.com/MisoLabsAI/MisoTTS.git
cd MisoTTS
uv sync --python 3.10
source .venv/bin/activate
```

Then run the example conversation. By default, `run_misotts.py` loads the public
model from [MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS) and
downloads it into the Hugging Face cache if it is not already present on your
machine:

```bash
uv run python run_misotts.py
```

The script writes `full_conversation.wav` in the repository root.

With `pip` instead of `uv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
python run_misotts.py
```

### Rust quickstart

A native Rust inference binary lives in `misotts-rs/`. It downloads the same
weights automatically and produces identical output. No Python, PyTorch, or
virtual environment required.

Prerequisites: a working [Rust toolchain](https://rustup.rs) (stable, 1.75+).

```bash
cd misotts-rs
cargo build --release          # CPU
# or, for CUDA:
cargo build --release --features cuda
```

Run the demo:

```bash
./target/release/misotts "Hello from Miso." --output hello.wav
```

The first run fetches the model checkpoint, the Mimi codec, and the Llama
tokenizer from Hugging Face Hub into `~/.cache/huggingface/hub/` (the same
cache the Python version uses).

---

## Model Introduction

Miso TTS 8B is a text-to-dialogue RVQ Transformer inspired by the Sesame CSM architecture. It
generates Mimi audio codes from text and optional audio context, using a large
Llama 3.2-style backbone and a smaller autoregressive audio decoder. To find out more
about the architecture, read [our blog post](https://misolabs.ai/blog/miso-tts-8b).

The model is designed for high-quality conversational speech generation.
This repository contains the inference
code, model definition, and setup instructions for running Miso TTS locally.

> **Language support:** Miso TTS 8B currently supports **English only**.

---

## Model Summary

| Item                | Value           |
| ------------------- | --------------- |
| Model               | Miso TTS 8B     |
| Organization        | Miso Labs       |
| Task                | Text-to-speech  |
| Architecture        | RVQ Transformer |
| Backbone            | `llama-8B`      |
| Audio decoder       | `llama-300M`    |
| Text vocabulary     | `128,256`       |
| Audio vocabulary    | `2,051`         |
| Audio codebooks     | `32`            |
| Audio tokenizer     | Mimi            |
| Max sequence length | `2,048`         |
| Languages           | English only    |

### Architecture

Miso TTS 8B uses two transformer components:

- A large backbone transformer that consumes text/audio-frame embeddings.
- A smaller decoder transformer that autoregressively predicts higher-order
  audio codebooks within each frame.

The backbone accepts interleaved text and audio tokens, allowing it to condition its generations on
the conversation history.

---

## Usage

### Python

```python
import torch
import torchaudio

from generator import load_miso_8b

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = load_miso_8b(
    device=device,
    model_path_or_repo_id="MisoLabs/MisoTTS",
)

audio = generator.generate(
    text="Hello from Miso.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("miso.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

### Prompted generation

Miso TTS can condition on prior audio for voice cloning.
This is optional; the quickstart example above runs without
prompt audio.

```python
import torchaudio

from generator import Segment, load_miso_8b

generator = load_miso_8b(device="cuda")

prompt_audio, sample_rate = torchaudio.load("prompt.wav")
prompt_audio = torchaudio.functional.resample(
    prompt_audio.squeeze(0),
    orig_freq=sample_rate,
    new_freq=generator.sample_rate,
)

context = [
    Segment(
        speaker=0,
        text="This is the transcript for the prompt audio.",
        audio=prompt_audio,
    )
]

audio = generator.generate(
    text="This is the next sentence to synthesize.",
    speaker=0,
    context=context,
    max_audio_length_ms=10_000,
)
```

---

## Weights

The model weights are hosted publicly on Hugging Face:

```bash
uv run python run_misotts.py
```

The default model repository is
[MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS). The first run
downloads the model automatically through Hugging Face Hub; later runs reuse the
cached copy.

The first run also downloads the SilentCipher watermarking model from
`sony/silentcipher`. If that separate download times out, rerun the command; the
Hugging Face cache resumes from files that already completed.

---

## Rust

The `misotts-rs/` directory contains a self-contained Rust crate that
reimplements the full inference pipeline natively — no Python interpreter,
no PyTorch, no virtual environment.

### Build

```bash
cd misotts-rs

# CPU (works on any machine)
cargo build --release

# NVIDIA GPU (requires CUDA toolkit)
cargo build --release --features cuda

# Apple GPU (requires macOS 13+)
cargo build --release --features metal
```

### CLI

```
USAGE:
    misotts [OPTIONS] <TEXT>

ARGS:
    <TEXT>    Text to synthesise

OPTIONS:
    --speaker <N>             Speaker ID [default: 0]
    --output <FILE>           Output WAV path [default: output.wav]
    --context-audio <FILE>    Prior audio for voice conditioning
    --context-text <TEXT>     Transcript of the context audio
    --context-speaker <N>     Speaker ID for the context audio [default: 0]
    --max-audio-ms <MS>       Maximum output length in ms [default: 90000]
    --temperature <F>         Sampling temperature [default: 0.9]
    --topk <N>                Top-k vocabulary size [default: 50]
    --model-repo <REPO>       HuggingFace repo ID for model weights
    --model-path <FILE>       Local path to model .safetensors file
    --cuda                    Use CUDA device 0
    --dtype <DTYPE>           bf16 (default on GPU), f16, or f32
```

**Basic synthesis:**

```bash
./target/release/misotts "Hello from Miso." --output hello.wav
```

**Multi-speaker conversation** (mirrors the Python demo):

```bash
./target/release/misotts "I'm just honestly not that into him, you know?" \
    --speaker 0 --output turn1.wav

./target/release/misotts "Yeah, I get it." \
    --speaker 1 \
    --context-audio turn1.wav \
    --context-text "I'm just honestly not that into him, you know?" \
    --context-speaker 0 \
    --output turn2.wav
```

**Voice conditioning** (provide a reference clip to match a speaker's voice):

```bash
./target/release/misotts "This is the generated line." \
    --context-audio reference.wav \
    --context-text "This is the reference transcript." \
    --output conditioned.wav
```

**Use a local model checkpoint** (skips the HuggingFace download):

```bash
./target/release/misotts "Hello." \
    --model-path /path/to/model.safetensors \
    --output hello.wav
```

### Library API

Add `misotts-rs` as a Cargo dependency:

```toml
[dependencies]
misotts = { path = "../misotts-rs" }
```

```rust
use misotts::{
    config::miso_tts_8b_config,
    generator::{Generator, Segment},
    mimi::MimiCodec,
    model::Model,
    tokenizer::TextTokenizer,
};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

let device = Device::Cpu;

let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&["model.safetensors"], DType::F32, &device)?
};
let model     = Model::load(miso_tts_8b_config(), vb, &device)?;
let tokenizer = TextTokenizer::from_file("tokenizer.json".as_ref())?;
let mimi      = MimiCodec::load("mimi.safetensors".as_ref(), &device)?;

let mut generator = Generator::new(model, tokenizer, mimi, device);

let audio: Vec<f32> = generator.generate(
    "Hello from Miso.",
    0,       // speaker
    &[],     // no context
    10_000.0,  // max_audio_ms
    0.9,     // temperature
    50,      // topk
)?;
// audio is a mono f32 waveform at 24 000 Hz
```

### Notes

- **Watermarking** is not applied by the Rust binary. The Python version
  embeds a SilentCipher watermark by default; no equivalent Rust library
  exists for SilentCipher. If watermarking is required, post-process the
  output WAV with the Python `watermarking.py` script.
- **Precision:** the Rust binary defaults to `bfloat16` on CUDA/Metal and
  `float32` on CPU — the same policy as the Python version. Override with
  `--dtype f32 | bf16 | f16`. `bf16` fits comfortably on a 24 GB card
  (RTX 3090/4090); `f32` requires ~40 GB. Outputs in `bf16` and `f32` are
  numerically close but not bit-for-bit identical.
- **Weight cache:** both the Rust and Python runtimes use the same
  `~/.cache/huggingface/hub/` directory, so weights only need to be
  downloaded once regardless of which runtime you use first.

---

## System Requirements

Miso TTS 8B is a **large** model (~8.2B parameters across the backbone, audio
decoder, embeddings, and heads). It is **not** a lightweight CPU model — plan for
a high-VRAM GPU for interactive use.

The numbers below are approximate and cover the model weights plus headroom for
the Mimi codec, the SilentCipher watermarker, the KV cache, and activations.

| Precision        | Weights (approx.) | Recommended VRAM | Example GPUs                       |
| ---------------- | ----------------- | ---------------- | ---------------------------------- |
| `bfloat16`/`fp16`| ~16 GB            | **24 GB**        | RTX 3090 / 4090, A5000, L4 (24 GB) |
| `float32`        | ~33 GB            | **40 GB+**       | A100 40 GB, A6000 48 GB, H100      |

**CPU:** inference runs but is slow. Budget at least ~20 GB RAM for `bfloat16`
and ~40 GB for `float32`.

**Disk:** the first run downloads ~30–40 GB total — the model checkpoint plus the
Mimi codec, the SilentCipher watermarker (Python only), and the Llama 3.2
tokenizer — into the Hugging Face cache. The Rust and Python runtimes share
the same cache directory. Make sure you have the free space before starting.

Both the Python and Rust runtimes default to `bfloat16` on GPU. A 24 GB
card (RTX 3090 / 4090) comfortably fits the bf16 weights. `float32` requires
~40 GB+. Smaller consumer GPUs (4–16 GB) are not sufficient for the full model.

---

## Safety

Miso TTS is a speech generation model. Do not use it to impersonate people,
create deceptive audio, commit fraud, or generate harmful content.

Generated audio is watermarked by default. If you deploy this model in another
application, use your own private watermark key and keep it secret.

---

## Links

- Website: [misolabs.ai](https://misolabs.ai)
- Hugging Face: [MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS)
- GitHub: [MisoLabsAI](https://github.com/MisoLabsAI)
- X: [@MisoLabsAI](https://x.com/MisoLabsAI)
