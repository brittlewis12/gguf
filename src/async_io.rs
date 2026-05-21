//! Async GGUF parsing on top of tokio.
//!
//! `AsyncGGUF::open` and `read_gguf*` stream from the file. The
//! `*_with_limits` entry points materialize the entire file in memory
//! and apply a byte cap; they are deprecated in favor of either the
//! streaming defaults or `MmapGGUF::open_with_limits`.
//!
//! # Example
//!
//! ```rust,no_run
//! use gguf_rs::async_io::AsyncGGUF;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut container = AsyncGGUF::open("model.gguf").await?;
//!     let model = container.decode().await?;
//!
//!     println!("Architecture: {}", model.model_family());
//!     println!("Tensors: {}", model.num_tensor());
//!
//!     Ok(())
//! }
//! ```

use std::io::{Read as _, Seek, SeekFrom};

use anyhow::{anyhow, Result};
use tokio::fs::File;
use tokio::io::AsyncReadExt;

use crate::{ByteOrder, GGUFModel, FILE_MAGIC_GGUF_BE, FILE_MAGIC_GGUF_LE};

/// Async GGUF file container.
///
/// `open()` retains the `std::fs::File` it validated; `decode()` clones
/// that descriptor and parses on `tokio::task::spawn_blocking`. The
/// parser sees bytes from the descriptor open() opened, not from
/// re-resolving the path.
pub struct AsyncGGUF {
    file: std::fs::File,
    byte_order: ByteOrder,
    max_array_size: u64,
    /// `Some(cap)` routes `decode()` through the deprecated snapshot
    /// path; `None` streams.
    snapshot_cap: Option<u64>,
}

impl AsyncGGUF {
    /// Open a GGUF file and validate its magic. Streaming default;
    /// `decode()` parses only the header bytes.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use gguf_rs::async_io::AsyncGGUF;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let container = AsyncGGUF::open("model.gguf").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        Self::open_inner(path.as_ref(), None).await
    }

    /// Open a GGUF file with an explicit snapshot byte cap. Snapshots
    /// the file into memory at decode time; rejects files larger than
    /// `max_input_bytes` with
    /// `"file size {N} exceeds async input cap {M} bytes"`.
    ///
    /// Prefer [`AsyncGGUF::open`] for streaming header parsing, or
    /// `MmapGGUF::open_with_limits` for frozen snapshot semantics.
    #[deprecated(
        since = "0.2.0",
        note = "snapshots the entire file into memory; use AsyncGGUF::open \
                for streaming header parsing, or MmapGGUF::open_with_limits \
                for frozen snapshot semantics"
    )]
    pub async fn open_with_limits<P: AsRef<std::path::Path>>(
        path: P,
        max_array_size: u64,
        max_input_bytes: u64,
    ) -> Result<Self> {
        let mut this = Self::open_inner(path.as_ref(), Some(max_input_bytes)).await?;
        this.max_array_size = max_array_size;
        Ok(this)
    }

    async fn open_inner(path: &std::path::Path, snapshot_cap: Option<u64>) -> Result<Self> {
        let mut async_file = match File::open(path).await {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(anyhow!("file not found: {}", path.display()));
            }
            Err(e) => return Err(e.into()),
        };
        let input_len = async_file.metadata().await?.len();

        // Cap check before magic so an oversized non-GGUF reports the
        // cap, not "invalid magic".
        if let Some(cap) = snapshot_cap {
            if input_len > cap {
                return Err(anyhow!(
                    "file size {} exceeds async input cap {} bytes",
                    input_len,
                    cap
                ));
            }
        }

        if input_len < 4 {
            return Err(anyhow!("file too small to be a valid GGUF file"));
        }

        let mut magic_bytes = [0u8; 4];
        async_file.read_exact(&mut magic_bytes).await?;
        let byte_order = match i32::from_le_bytes(magic_bytes) {
            FILE_MAGIC_GGUF_LE => ByteOrder::LE,
            FILE_MAGIC_GGUF_BE => ByteOrder::BE,
            _ => return Err(anyhow!("invalid file magic: not a GGUF file")),
        };

        let std_file = async_file.into_std().await;

        Ok(Self {
            file: std_file,
            byte_order,
            max_array_size: 3,
            snapshot_cap,
        })
    }

    /// Override the max metadata array size.
    pub fn with_max_array_size(mut self, max_array_size: u64) -> Self {
        self.max_array_size = max_array_size;
        self
    }

    /// Decode the GGUF file. Runs the sync parser on
    /// `tokio::task::spawn_blocking` against a clone of the descriptor
    /// retained by `open()`.
    pub async fn decode(&mut self) -> Result<GGUFModel> {
        let file = self.file.try_clone()?;
        let byte_order = self.byte_order.clone();
        let max_array_size = self.max_array_size;
        let snapshot_cap = self.snapshot_cap;

        tokio::task::spawn_blocking(move || -> Result<GGUFModel> {
            match snapshot_cap {
                None => decode_streaming(file, &byte_order, max_array_size),
                Some(cap) => decode_snapshot(file, &byte_order, max_array_size, cap),
            }
        })
        .await
        .map_err(|e| anyhow!("async GGUF decode task failed: {e}"))?
    }
}

fn decode_streaming(
    mut file: std::fs::File,
    byte_order: &ByteOrder,
    max_array_size: u64,
) -> Result<GGUFModel> {
    // Re-stat at decode time so tensor EOF/range validation sees the
    // file's current length, not the value sampled at open().
    let input_len = file.metadata()?.len();
    if input_len < 4 {
        return Err(anyhow!("file too small to be a valid GGUF file"));
    }

    // Re-validate magic and byte order against the live bytes; the
    // file may have been truncated and rewritten between open() and
    // here. Callers needing stronger guarantees should use
    // `MmapGGUF::open_with_limits`.
    file.seek(SeekFrom::Start(0))?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    let actual = match i32::from_le_bytes(magic) {
        FILE_MAGIC_GGUF_LE => ByteOrder::LE,
        FILE_MAGIC_GGUF_BE => ByteOrder::BE,
        _ => return Err(anyhow!("invalid file magic: not a GGUF file")),
    };
    if std::mem::discriminant(byte_order) != std::mem::discriminant(&actual) {
        return Err(anyhow!("GGUF byte order changed between open() and decode()"));
    }

    let reader = std::io::BufReader::new(file);
    let mut container =
        crate::GGUFContainer::new_after_magic(actual, Box::new(reader), max_array_size)
            .with_input_len(input_len);
    container.decode()
}

fn decode_snapshot(
    mut file: std::fs::File,
    byte_order: &ByteOrder,
    max_array_size: u64,
    max_input_bytes: u64,
) -> Result<GGUFModel> {
    // Re-stat so a file that grew past the cap after open() is still
    // rejected before we allocate.
    let input_len = file.metadata()?.len();
    if input_len > max_input_bytes {
        return Err(anyhow!(
            "file size {} exceeds async input cap {} bytes",
            input_len,
            max_input_bytes
        ));
    }

    // Fail before the snapshot allocation if the live bytes no longer
    // describe a GGUF file, or if the byte order flipped under us.
    file.seek(SeekFrom::Start(0))?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    let actual = match i32::from_le_bytes(magic) {
        FILE_MAGIC_GGUF_LE => ByteOrder::LE,
        FILE_MAGIC_GGUF_BE => ByteOrder::BE,
        _ => return Err(anyhow!("invalid file magic: not a GGUF file")),
    };
    if std::mem::discriminant(byte_order) != std::mem::discriminant(&actual) {
        return Err(anyhow!("GGUF byte order changed between open() and decode()"));
    }

    file.seek(SeekFrom::Start(0))?;
    let len = usize::try_from(input_len)
        .map_err(|_| anyhow!("file too large to snapshot into memory on this platform"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|e| anyhow!("failed to reserve async snapshot ({len} bytes): {e}"))?;
    bytes.resize(len, 0);
    file.read_exact(&mut bytes)?;

    // Re-check magic/byte order on the snapshot. The bytes may have
    // changed between the pre-allocation check above and `read_exact`.
    if bytes.len() < 4 {
        return Err(anyhow!("file too small to be a valid GGUF file"));
    }
    let snapshot_magic = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let snapshot_order = match snapshot_magic {
        FILE_MAGIC_GGUF_LE => ByteOrder::LE,
        FILE_MAGIC_GGUF_BE => ByteOrder::BE,
        _ => return Err(anyhow!("invalid file magic: not a GGUF file")),
    };
    if std::mem::discriminant(byte_order) != std::mem::discriminant(&snapshot_order) {
        return Err(anyhow!("GGUF byte order changed between open() and decode()"));
    }

    let cursor = std::io::Cursor::new(bytes);
    let mut container =
        crate::GGUFContainer::new(Box::new(cursor), max_array_size)?.with_input_len(input_len);
    container.decode()
}

/// Open and decode a GGUF file in one async operation using the streaming
/// default.
///
/// # Example
///
/// ```rust,no_run
/// use gguf_rs::async_io::read_gguf;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = read_gguf("model.gguf").await?;
/// println!("Architecture: {}", model.model_family());
/// # Ok(())
/// # }
/// ```
pub async fn read_gguf<P: AsRef<std::path::Path>>(path: P) -> Result<GGUFModel> {
    let mut container = AsyncGGUF::open(path).await?;
    container.decode().await
}

/// Open and decode a GGUF file with a custom max array size, using the
/// streaming default.
pub async fn read_gguf_with_array_size<P: AsRef<std::path::Path>>(
    path: P,
    max_array_size: u64,
) -> Result<GGUFModel> {
    let mut container = AsyncGGUF::open(path)
        .await?
        .with_max_array_size(max_array_size);
    container.decode().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_open() {
        let container = AsyncGGUF::open("tests/test-le-v3.gguf").await;
        assert!(container.is_ok());
    }

    #[tokio::test]
    async fn test_async_decode() {
        let mut container = AsyncGGUF::open("tests/test-le-v3.gguf").await.unwrap();
        let model = container.decode().await.unwrap();
        assert_eq!(model.get_version(), "v3");
        assert_eq!(model.model_family(), "llama");
    }

    #[tokio::test]
    async fn test_async_read_gguf() {
        let model = read_gguf("tests/test-le-v3.gguf").await.unwrap();
        assert_eq!(model.model_family(), "llama");
    }

    #[tokio::test]
    async fn test_async_file_not_found() {
        let result = AsyncGGUF::open("nonexistent.gguf").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn test_async_input_limit_is_enforced() {
        let result = AsyncGGUF::open_with_limits("tests/test-le-v3.gguf", 3, 1).await;
        assert!(result.is_err());
    }

    fn write_large_empty_gguf(path: &std::path::Path, total_size: u64) {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&crate::FILE_MAGIC_GGUF_LE.to_le_bytes())
            .unwrap();
        f.write_all(&crate::GGUF_VERSION_V3.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap(); // num_tensors
        f.write_all(&0u64.to_le_bytes()).unwrap(); // num_kv
        f.set_len(total_size).unwrap();
    }

    #[tokio::test]
    async fn async_streaming_default_parses_large_file_with_small_header() {
        let path = std::env::temp_dir()
            .join(format!("gguf-async-streaming-large-{}.gguf", std::process::id()));
        let size = crate::DEFAULT_MAX_HELPER_INPUT_BYTES + 4096;
        write_large_empty_gguf(&path, size);

        let result = read_gguf(&path).await;
        let _ = std::fs::remove_file(&path);
        let model = result.expect("streaming async helper should parse large file");
        assert_eq!(model.num_tensor(), 0);
        assert_eq!(model.num_kv(), 0);
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn async_snapshot_helper_rejects_large_file() {
        let path = std::env::temp_dir()
            .join(format!("gguf-async-snapshot-large-{}.gguf", std::process::id()));
        let size = crate::DEFAULT_MAX_HELPER_INPUT_BYTES + 4096;
        write_large_empty_gguf(&path, size);

        let result =
            AsyncGGUF::open_with_limits(&path, 3, crate::DEFAULT_MAX_HELPER_INPUT_BYTES).await;
        let _ = std::fs::remove_file(&path);
        let err = match result {
            Ok(_) => panic!("snapshot async helper should reject oversized file"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("exceeds async input cap"),
            "expected legacy async cap error, got: {msg}"
        );
    }
}
