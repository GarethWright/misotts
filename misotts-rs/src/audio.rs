use anyhow::{Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use rubato::{FftFixedInOut, Resampler};
use std::path::Path;

/// Write a mono f32 waveform to a WAV file.
pub fn save_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("creating WAV file: {}", path.display()))?;
    for &s in samples {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Load a mono (or down-mixed) f32 waveform from a WAV / any format hound supports.
pub fn load_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<hound::Result<Vec<_>>>()?,
        SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .collect::<hound::Result<Vec<_>>>()?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Down-mix to mono if multi-channel.
    let mono: Vec<f32> = if spec.channels == 1 {
        samples
    } else {
        let ch = spec.channels as usize;
        samples
            .chunks_exact(ch)
            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
            .collect()
    };

    Ok((mono, sample_rate))
}

/// Resample `samples` from `orig_sr` Hz to `target_sr` Hz using a high-quality
/// FFT-based sinc resampler.
pub fn resample(samples: &[f32], orig_sr: u32, target_sr: u32) -> Result<Vec<f32>> {
    if orig_sr == target_sr {
        return Ok(samples.to_vec());
    }

    // rubato's FftFixedInOut wants a chunk size; pick something reasonable.
    let chunk = 1024usize;
    let mut resampler = FftFixedInOut::<f32>::new(
        orig_sr as usize,
        target_sr as usize,
        chunk,
        1, // mono
    )?;

    let mut output: Vec<f32> = Vec::with_capacity(
        (samples.len() as f64 * target_sr as f64 / orig_sr as f64) as usize + chunk,
    );

    let mut pos = 0usize;
    loop {
        let end = (pos + chunk).min(samples.len());
        let mut frame = samples[pos..end].to_vec();
        if frame.len() < chunk {
            frame.resize(chunk, 0.0);
        }
        let out = resampler.process(&[&frame], None)?;
        output.extend_from_slice(&out[0]);
        pos += chunk;
        if pos >= samples.len() {
            break;
        }
    }

    Ok(output)
}
