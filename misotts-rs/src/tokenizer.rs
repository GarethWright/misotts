use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// Thin wrapper around the HuggingFace `tokenizers` crate, configured to
/// match the Python code's `load_llama3_tokenizer()` behaviour:
/// each encoded sequence is wrapped with BOS…EOS tokens and a speaker tag.
pub struct TextTokenizer {
    inner: Tokenizer,
    bos_id: u32,
    eos_id: u32,
}

impl TextTokenizer {
    /// Load from a `tokenizer.json` file on disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

        let bos_id = inner
            .token_to_id("<|begin_of_text|>")
            .context("BOS token not found in vocabulary")?;
        let eos_id = inner
            .token_to_id("<|end_of_text|>")
            .context("EOS token not found in vocabulary")?;

        Ok(Self { inner, bos_id, eos_id })
    }

    /// Encode `"[{speaker}] {text}"` with BOS prepended and EOS appended,
    /// matching the Python `TemplateProcessing` post-processor.
    pub fn encode(&self, text: &str, speaker: u32) -> Result<Vec<u32>> {
        let formatted = format!("[{speaker}] {}", text.trim_start());
        let encoding = self
            .inner
            .encode(formatted.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
        let mut ids = vec![self.bos_id];
        ids.extend_from_slice(encoding.get_ids());
        ids.push(self.eos_id);
        Ok(ids)
    }
}
