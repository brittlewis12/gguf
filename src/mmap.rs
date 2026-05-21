//! `MmapGGUF`: in-memory snapshot reader.
//!
//! `open()` reads the whole file into an anonymous mapped buffer
//! (`MmapMut::map_anon` + `read_exact`) and parses from that frozen
//! copy. This is not a lazy file mapping; size is capped by
//! [`DEFAULT_MAX_HELPER_INPUT_BYTES`](crate::DEFAULT_MAX_HELPER_INPUT_BYTES)
//! (256 MiB default; configurable via `open_with_limits`).
//!
//! The snapshot is not atomic with respect to in-place mutation during
//! the read itself — a concurrent writer can produce a torn image and
//! the reader does not detect it. For stronger guarantees, hash-verify
//! the file before parsing.
//!
//! For header-only parsing prefer the streaming helpers
//! ([`get_gguf_container`](crate::get_gguf_container),
//! [`get_gguf_container_array_size`](crate::get_gguf_container_array_size)).
//!
//! # Example
//!
//! ```rust,no_run
//! use gguf_rs::mmap::MmapGGUF;
//!
//! let mmap = MmapGGUF::open("model.gguf")?;
//! let model = mmap.model();
//!
//! println!("Architecture: {}", model.model_family());
//! println!("Tensors: {}", model.num_tensor());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use anyhow::{anyhow, Result};
use memmap2::{Mmap, MmapMut, MmapOptions};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use crate::{GGUFModel, DEFAULT_MAX_HELPER_INPUT_BYTES, FILE_MAGIC_GGUF_BE, FILE_MAGIC_GGUF_LE};

/// In-memory snapshot GGUF reader.
pub struct MmapGGUF {
    #[allow(dead_code)]
    mmap: Arc<Mmap>,
    model: GGUFModel,
}

struct MmapReader {
    mmap: Arc<Mmap>,
    pos: usize,
}

impl Read for MmapReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.mmap.len() {
            return Ok(0);
        }
        let n = buf.len().min(self.mmap.len() - self.pos);
        buf[..n].copy_from_slice(&self.mmap[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

impl MmapGGUF {
    /// Open a GGUF file by snapshotting it into an anonymous in-memory
    /// buffer and decoding from that frozen copy.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file does not exist
    /// - The snapshot allocation fails
    /// - The file has an invalid magic number
    /// - The file exceeds [`DEFAULT_MAX_HELPER_INPUT_BYTES`]
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use gguf_rs::mmap::MmapGGUF;
    ///
    /// let mmap = MmapGGUF::open("model.gguf")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_limits(path, 3, DEFAULT_MAX_HELPER_INPUT_BYTES)
    }

    pub fn open_with_array_size<P: AsRef<Path>>(path: P, max_array_size: u64) -> Result<Self> {
        Self::open_with_limits(path, max_array_size, DEFAULT_MAX_HELPER_INPUT_BYTES)
    }

    pub fn open_with_limits<P: AsRef<Path>>(
        path: P,
        max_array_size: u64,
        max_snapshot_bytes: u64,
    ) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(anyhow!("file not found: {}", path.display()));
        }

        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len > max_snapshot_bytes {
            return Err(anyhow!(
                "file size {} exceeds snapshot cap {} bytes",
                file_len,
                max_snapshot_bytes
            ));
        }

        if file_len < 4 {
            return Err(anyhow!("file too small to be a valid GGUF file"));
        }

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        match i32::from_le_bytes(magic) {
            FILE_MAGIC_GGUF_LE | FILE_MAGIC_GGUF_BE => {}
            _ => return Err(anyhow!("invalid file magic: not a GGUF file")),
        }

        let mmap = if file_len == 0 {
            Arc::new(MmapMut::map_anon(0)?.make_read_only()?)
        } else {
            let len = usize::try_from(file_len)
                .map_err(|_| anyhow!("file too large to snapshot into memory on this platform"))?;
            file.seek(SeekFrom::Start(0))?;
            let mut snapshot = MmapOptions::new().len(len).map_anon()?;
            file.read_exact(&mut snapshot)?;
            Arc::new(snapshot.make_read_only()?)
        };

        let reader = MmapReader {
            mmap: Arc::clone(&mmap),
            pos: 0,
        };
        let mut container =
            crate::GGUFContainer::new(Box::new(reader), max_array_size)?.with_input_len(file_len);
        let model = container.decode()?;

        Ok(Self { mmap, model })
    }

    /// Get the decoded GGUF model.
    pub fn model(&self) -> &GGUFModel {
        &self.model
    }

    /// Borrow the in-memory snapshot of the file's bytes.
    pub fn as_slice(&self) -> &[u8] {
        self.mmap.as_ref()
    }

    /// Length in bytes of the in-memory snapshot.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Whether the snapshot is empty (file of length 0).
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

// Implement Deref to allow direct access to GGUFModel methods
impl std::ops::Deref for MmapGGUF {
    type Target = GGUFModel;

    fn deref(&self) -> &Self::Target {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_open() {
        let mmap = MmapGGUF::open("tests/test-le-v3.gguf").unwrap();
        assert!(!mmap.is_empty());
    }

    #[test]
    fn test_mmap_decode() {
        let mmap = MmapGGUF::open("tests/test-le-v3.gguf").unwrap();
        assert_eq!(mmap.model().get_version(), "v3");
        assert_eq!(mmap.model().model_family(), "llama");
    }

    #[test]
    fn test_mmap_deref() {
        let mmap = MmapGGUF::open("tests/test-le-v3.gguf").unwrap();
        // Test Deref allows direct access to model methods
        assert_eq!(mmap.get_version(), "v3");
        assert_eq!(mmap.model_family(), "llama");
    }

    #[test]
    fn test_mmap_file_not_found() {
        let result = MmapGGUF::open("nonexistent.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn test_mmap_snapshot_limit_is_enforced() {
        let result = MmapGGUF::open_with_limits("tests/test-le-v3.gguf", 3, 1);
        assert!(result.is_err());
    }
}
