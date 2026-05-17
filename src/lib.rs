//! # GGUF File Parser
//!
//! A Rust library for parsing and reading GGUF (GGML Universal Format) files.
//!
//! GGUF files are binary files that contain key-value metadata and tensors,
//! commonly used for storing quantized machine learning models like LLaMA, Phi, etc.
//!
//! ## Features
//!
//! - Decode GGUF files (v1, v2, v3)
//! - Access key-value metadata
//! - Access tensor information
//! - Support for little-endian and big-endian files
//! - CLI tool for quick inspection
//! - Optional memory-mapped file support (enable `mmap` feature)
//!
//! ## Example
//!
//! ```rust,no_run
//! use gguf_rs::get_gguf_container;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Open a GGUF file
//!     let mut container = get_gguf_container("model.gguf")?;
//!     let model = container.decode()?;
//!
//!     // Print model info
//!     println!("Version: {}", model.get_version());
//!     println!("Architecture: {}", model.model_family());
//!     println!("Parameters: {}", model.model_parameters());
//!     println!("File type: {}", model.file_type());
//!     println!("Tensors: {}", model.num_tensor());
//!
//!     // List tensors
//!     for tensor in model.tensors() {
//!         println!("  {}: {:?} {:?}", tensor.name, tensor.kind, tensor.shape);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## CLI Usage
//!
//! Install the CLI tool:
//! ```bash
//! cargo install gguf-rs
//! ```
//!
//! Show model info:
//! ```bash
//! gguf model.gguf
//! ```
//!
//! Show tensors:
//! ```bash
//! gguf model.gguf --tensors
//! ```
//!
//! ## Memory-Mapped Files
//!
//! For large files, enable the `mmap` feature for more efficient access:
//!
//! ```toml
//! [dependencies]
//! gguf-rs = { version = "0.1", features = ["mmap"] }
//! ```
//!
//! ```rust,ignore
//! use gguf_rs::mmap::MmapGGUF;
//!
//! let mmap = MmapGGUF::open("large_model.gguf")?;
//! let model = mmap.model();
//! println!("{}", model.model_family());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Async I/O
//!
//! For async applications, enable the `async` feature:
//!
//! ```toml
//! [dependencies]
//! gguf-rs = { version = "0.1", features = ["async"] }
//! ```
//!
//! ```rust,ignore
//! use gguf_rs::async_io::AsyncGGUF;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut container = AsyncGGUF::open("model.gguf").await?;
//!     let model = container.decode().await?;
//!
//!     println!("Architecture: {}", model.model_family());
//!     Ok(())
//! }
//! ```

use anyhow::{anyhow, Result};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
#[cfg(feature = "debug")]
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{borrow::Borrow, collections::BTreeMap, fmt::Display, io::Read};

/// Magic constant for `ggml` files (unversioned).
pub const FILE_MAGIC_GGML: i32 = 0x67676d6c;
/// Magic constant for `ggml` files (versioned, ggmf).
pub const FILE_MAGIC_GGMF: i32 = 0x67676d66;
/// Magic constant for `ggml` files (versioned, ggjt).
pub const FILE_MAGIC_GGJT: i32 = 0x67676a74;
/// Magic constant for `ggla` files (LoRA adapter).
pub const FILE_MAGIC_GGLA: i32 = 0x67676C61;
/// Magic constant for `gguf` files (versioned, gguf)
pub const FILE_MAGIC_GGUF_LE: i32 = 0x46554747;
pub const FILE_MAGIC_GGUF_BE: i32 = 0x47475546;

pub const GGUF_VERSION_V1: i32 = 0x00000001;
pub const GGUF_VERSION_V2: i32 = 0x00000002;
pub const GGUF_VERSION_V3: i32 = 0x00000003;

const THOUSAND: u64 = 1000;
const MILLION: u64 = 1_000_000;
const BILLION: u64 = 1_000_000_000;

const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
const MAX_ALIGNMENT: u64 = 65536;
const MAX_TENSORS: u64 = 100_000;
const MAX_KV: u64 = 100_000;
const MAX_METADATA_KEY_LEN: u64 = 65535;
const MAX_TENSOR_NAME_LEN: u64 = 64;
const MAX_STRING_VALUE_LEN: u64 = 16 * 1024 * 1024;
const MAX_ARRAY_LEN: u64 = 1_000_000;
const MAX_STORED_ARRAY_ITEMS: u64 = 300_000;
const MAX_DIMENSION: u64 = 1u64 << 30;
const MAX_ELEMENTS: u64 = 1u64 << 40;
const MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 128 * 1024 * 1024;
const DEFAULT_MAX_HELPER_INPUT_BYTES: u64 = 256 * 1024 * 1024;

/// Convert a number to a human-readable string.
fn human_number(value: u64) -> String {
    match value {
        _ if value > BILLION => format!("{:.0}B", value as f64 / BILLION as f64),
        _ if value > MILLION => format!("{:.0}M", value as f64 / MILLION as f64),
        _ if value > THOUSAND => format!("{:.0}K", value as f64 / THOUSAND as f64),
        _ => format!("{}", value),
    }
}

/// Convert a file type to a human-readable string.
/// GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
fn file_type(ft: u64) -> String {
    match ft {
        0 => "All F32",
        1 => "Mostly F16",
        2 => "Mostly Q4_0",
        3 => "Mostly Q4_1",
        4 => "Mostly Q4_1 Some F16",
        5 => "Mostly Q4_2 (UNSUPPORTED)",
        6 => "Mostly Q4_3 (UNSUPPORTED)",
        7 => "Mostly Q8_0",
        8 => "Mostly Q5_0",
        9 => "Mostly Q5_1",
        10 => "Mostly Q2_K",
        11 => "Mostly Q3_K",
        12 => "Mostly Q4_K",
        13 => "Mostly Q5_K",
        14 => "Mostly Q6_K",
        15 => "Mostly IQ2_XXS",
        16 => "Mostly IQ2_XS",
        17 => "Mostly IQ3_XXS",
        18 => "Mostly IQ1_S",
        19 => "Mostly IQ4_NL",
        20 => "Mostly IQ3_S",
        21 => "Mostly IQ2_S",
        22 => "Mostly IQ4_XS",
        23 => "Mostly IQ1_M",
        24 => "Mostly BF16",
        _ => "unknown",
    }
    .to_string()
}

/// Byte order of the GGUF file.
#[derive(Default, Debug, Clone)]
pub enum ByteOrder {
    #[default]
    LE,
    BE,
}

/// Version of the GGUF file.
#[derive(Debug, Clone)]
pub enum Version {
    V1(V1),
    V2(V2),
    V3(V3),
}

/// Version 1 of the GGUF file.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct V1 {
    num_tensor: u32,
    num_kv: u32,
}

/// Version 2 of the GGUF file.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct V2 {
    num_tensor: u64,
    num_kv: u64,
}

/// Version 3 of the GGUF file.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct V3 {
    num_tensor: u64,
    num_kv: u64,
}

/// GGUF file container for reading GGUF binary files.
///
/// The container wraps a reader and provides methods to decode the GGUF file
/// into a [`GGUFModel`].
///
/// Use [`get_gguf_container`] for a convenient way to open a file.
pub struct GGUFContainer {
    bo: ByteOrder,
    version: Version,
    reader: Box<dyn std::io::Read + 'static>,
    max_array_size: u64,
    input_bounds: InputBounds,
}

#[derive(Debug, Clone, Copy)]
enum InputBounds {
    Unknown,
    Known(u64),
    TrustedUnbounded,
}

impl GGUFContainer {
    /// Create a new `GGUFContainer` from a byte order and a reader.
    ///
    /// This is a low-level constructor. For checked decoding of untrusted
    /// inputs, callers should also provide the total input length via
    /// [`GGUFContainer::with_input_len`]. Callers that intentionally want to
    /// parse an unbounded/trusted stream must opt in explicitly with
    /// [`GGUFContainer::allow_unbounded_input`]. File-backed helpers in this
    /// crate set the input length automatically.
    ///
    /// # Arguments
    ///
    /// * `reader` - A reader implementing `std::io::Read`
    /// * `max_array_size` - Maximum size for array metadata values
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use gguf_rs::GGUFContainer;
    /// use std::fs::File;
    ///
    /// let file = File::open("model.gguf")?;
    /// let container = GGUFContainer::new(Box::new(file), 1024)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(mut reader: Box<dyn std::io::Read>, max_array_size: u64) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        let bo = detect_magic(magic)?;
        Ok(Self::new_after_magic(bo, reader, max_array_size))
    }

    fn new_after_magic(bo: ByteOrder, reader: Box<dyn std::io::Read>, max_array_size: u64) -> Self {
        Self {
            bo,
            version: Version::V1(V1::default()),
            reader,
            max_array_size,
            input_bounds: InputBounds::Unknown,
        }
    }

    pub fn with_input_len(mut self, input_len: u64) -> Self {
        self.input_bounds = InputBounds::Known(input_len);
        self
    }

    /// Explicitly opt into decoding a trusted unbounded reader.
    ///
    /// This disables EOF-based tensor range validation. Prefer
    /// [`GGUFContainer::with_input_len`] for untrusted inputs.
    pub fn allow_unbounded_input(mut self) -> Self {
        self.input_bounds = InputBounds::TrustedUnbounded;
        self
    }

    /// Get the version of the GGUF file container.
    ///
    /// Returns the default version ("v1") before decoding.
    /// After successful decode, returns the actual file version ("v1", "v2", or "v3").
    pub fn get_version(&self) -> String {
        match &self.version {
            Version::V1(_) => String::from("v1"),
            Version::V2(_) => String::from("v2"),
            Version::V3(_) => String::from("v3"),
        }
    }

    /// Decode the GGUF file and return a `GGUFModel`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file has an invalid or unsupported GGUF version
    /// - The file contains malformed data
    /// - An I/O error occurs while reading
    /// - The caller used [`GGUFContainer::new`] without either
    ///   [`GGUFContainer::with_input_len`] or
    ///   [`GGUFContainer::allow_unbounded_input`]
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use gguf_rs::get_gguf_container;
    ///
    /// let mut container = get_gguf_container("model.gguf")?;
    /// let model = container.decode()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decode(&mut self) -> Result<GGUFModel> {
        if matches!(self.input_bounds, InputBounds::Unknown) {
            return Err(anyhow!(
                "input length is required for checked decoding; use with_input_len(...) or explicitly opt into allow_unbounded_input()"
            ));
        }

        let input_bounds = self.input_bounds;
        let mut reader = std::mem::replace(&mut self.reader, Box::new(std::io::empty()));
        let remaining_after_magic = match input_bounds {
            InputBounds::Known(total_len) => total_len
                .checked_sub(4)
                .ok_or_else(|| anyhow!("input length {total_len} is smaller than GGUF magic"))?,
            InputBounds::TrustedUnbounded => 0,
            InputBounds::Unknown => unreachable!(),
        };

        let version = match self.bo {
            ByteOrder::LE => {
                if matches!(input_bounds, InputBounds::Known(_)) {
                    BoundedReader::new(&mut reader, remaining_after_magic)
                        .read_i32::<LittleEndian>()?
                } else {
                    reader.read_i32::<LittleEndian>()?
                }
            }
            ByteOrder::BE => {
                if matches!(input_bounds, InputBounds::Known(_)) {
                    BoundedReader::new(&mut reader, remaining_after_magic)
                        .read_i32::<BigEndian>()?
                } else {
                    reader.read_i32::<BigEndian>()?
                }
            }
        };
        let remaining_after_version = remaining_after_magic.saturating_sub(4);

        #[cfg(feature = "debug")]
        {
            debug!("version {}", version);
        }

        match version {
            GGUF_VERSION_V1 => {
                let mut buffer: [u32; 2] = [0; 2];
                match self.bo {
                    ByteOrder::LE => {
                        if matches!(input_bounds, InputBounds::Known(_)) {
                            BoundedReader::new(&mut reader, remaining_after_version)
                                .read_u32_into::<LittleEndian>(&mut buffer)?
                        } else {
                            reader.read_u32_into::<LittleEndian>(&mut buffer)?
                        }
                    }
                    ByteOrder::BE => {
                        if matches!(input_bounds, InputBounds::Known(_)) {
                            BoundedReader::new(&mut reader, remaining_after_version)
                                .read_u32_into::<BigEndian>(&mut buffer)?
                        } else {
                            reader.read_u32_into::<BigEndian>(&mut buffer)?
                        }
                    }
                };

                self.version = Version::V1(V1 {
                    num_tensor: buffer[0],
                    num_kv: buffer[1],
                });
            }
            GGUF_VERSION_V2 | GGUF_VERSION_V3 => {
                let mut buffer: [u64; 2] = [0; 2];
                match self.bo {
                    ByteOrder::LE => {
                        if matches!(input_bounds, InputBounds::Known(_)) {
                            BoundedReader::new(&mut reader, remaining_after_version)
                                .read_u64_into::<LittleEndian>(&mut buffer)?
                        } else {
                            reader.read_u64_into::<LittleEndian>(&mut buffer)?
                        }
                    }
                    ByteOrder::BE => {
                        if matches!(input_bounds, InputBounds::Known(_)) {
                            BoundedReader::new(&mut reader, remaining_after_version)
                                .read_u64_into::<BigEndian>(&mut buffer)?
                        } else {
                            reader.read_u64_into::<BigEndian>(&mut buffer)?
                        }
                    }
                };

                if version == GGUF_VERSION_V2 {
                    self.version = Version::V2(V2 {
                        num_tensor: buffer[0],
                        num_kv: buffer[1],
                    });
                } else {
                    self.version = Version::V3(V3 {
                        num_tensor: buffer[0],
                        num_kv: buffer[1],
                    });
                }
            }
            invalid_version => {
                return Err(anyhow!(
                    "invalid version {}, only support version: 1 | 2 | 3",
                    invalid_version
                ));
            }
        };

        validate_counts(self.num_tensor(), self.num_kv())?;

        let mut model = GGUFModel {
            kv: BTreeMap::new(),
            kv_types: BTreeMap::new(),
            tensors: Vec::new(),
            parameters: 0,
            max_array_size: self.max_array_size,
            bo: self.bo.clone(),
            version: self.version.clone(),
        };

        let header_preamble_len = match self.version {
            Version::V1(_) => 4 + 4 + 4 + 4,
            Version::V2(_) | Version::V3(_) => 4 + 4 + 8 + 8,
        } as u64;

        let remaining_after_container_header = match input_bounds {
            InputBounds::Known(total_len) => total_len
                .checked_sub(header_preamble_len)
                .ok_or_else(|| anyhow!("input length {total_len} is smaller than GGUF header"))?,
            InputBounds::TrustedUnbounded => 0,
            InputBounds::Unknown => unreachable!(),
        };

        if let InputBounds::Known(_) = input_bounds {
            let bounded = BoundedReader::new(reader, remaining_after_container_header);
            model.decode(bounded, input_bounds, header_preamble_len)?;
        } else {
            model.decode(reader, input_bounds, header_preamble_len)?;
        }
        Ok(model)
    }

    fn num_kv(&self) -> u64 {
        match &self.version {
            Version::V1(v1) => v1.num_kv as u64,
            Version::V2(v2) => v2.num_kv,
            Version::V3(v3) => v3.num_kv,
        }
    }

    fn num_tensor(&self) -> u64 {
        match &self.version {
            Version::V1(v1) => v1.num_tensor as u64,
            Version::V2(v2) => v2.num_tensor,
            Version::V3(v3) => v3.num_tensor,
        }
    }
}

/// Tensor in the GGUF file.
///
/// Represents a single tensor with its metadata including name, type, offset, size, and shape.
///
/// # Example
///
/// ```rust,no_run
/// use gguf_rs::get_gguf_container;
///
/// let mut container = get_gguf_container("model.gguf")?;
/// let model = container.decode()?;
///
/// for tensor in model.tensors() {
///     println!("Tensor: {} (shape: {:?})", tensor.name, tensor.shape);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Name of the tensor (e.g., "token_embd.weight", "blk.0.attn_q.weight")
    pub name: String,
    /// GGML type identifier (see [`GGMLType`] for interpretation)
    pub kind: u32,
    /// Byte offset relative to the GGUF tensor-data section where this tensor begins
    pub offset: u64,
    /// Size of tensor data in bytes
    pub size: u64,
    /// Shape dimensions (number of elements in each dimension)
    pub shape: Vec<u64>,
}

/// Decoded GGUF model containing metadata and tensors.
///
/// Use [`get_gguf_container`] to create a container, then call [`GGUFContainer::decode`]
/// to get a `GGUFModel`.
///
/// # Example
///
/// ```rust,no_run
/// use gguf_rs::get_gguf_container;
///
/// let mut container = get_gguf_container("model.gguf")?;
/// let model = container.decode()?;
///
/// println!("Model: {}", model.model_family());
/// println!("Parameters: {}", model.model_parameters());
/// println!("Tensors: {}", model.num_tensor());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct GGUFModel {
    kv: BTreeMap<String, Value>,
    kv_types: BTreeMap<String, MetadataValueType>,
    tensors: Vec<Tensor>,
    parameters: u64,
    max_array_size: u64,
    bo: ByteOrder,
    version: Version,
}

/// Metadata value type in GGUF files.
///
/// Represents the type of a metadata value in the key-value store.
/// Used when decoding metadata to determine how to interpret bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for MetadataValueType {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => MetadataValueType::Uint8,
            1 => MetadataValueType::Int8,
            2 => MetadataValueType::Uint16,
            3 => MetadataValueType::Int16,
            4 => MetadataValueType::Uint32,
            5 => MetadataValueType::Int32,
            6 => MetadataValueType::Float32,
            7 => MetadataValueType::Bool,
            8 => MetadataValueType::String,
            9 => MetadataValueType::Array,
            10 => MetadataValueType::Uint64,
            11 => MetadataValueType::Int64,
            12 => MetadataValueType::Float64,
            _ => return Err(anyhow!("unsupport metadata value type")),
        })
    }
}

/// GGML type of a tensor in the GGUF file.
///
/// Represents the quantization format used for tensor data.
/// Most types are quantized formats that compress float values
/// to reduce memory footprint while maintaining accuracy.
#[derive(Debug, Serialize)]
#[allow(non_camel_case_types)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_2 = 4, // Unsupported
    Q4_3 = 5, // Unsupported
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    Q4_0_4_4 = 31, // Unsupported
    Q4_0_4_8 = 32, // Unsupported
    Q4_0_8_8 = 33, // Unsupported
    TQ1_0 = 34,
    TQ2_0 = 35,
    IQ4_NL_4_4 = 36, // Unsupported
    IQ4_NL_4_8 = 37, // Unsupported
    IQ4_NL_8_8 = 38, // Unsupported
    MXFP4 = 39,
    Count = 40,
}

impl Display for GGMLType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GGMLType::F32 => write!(f, "F32"),
            GGMLType::F16 => write!(f, "F16"),
            GGMLType::Q4_0 => write!(f, "Q4_0"),
            GGMLType::Q4_1 => write!(f, "Q4_1"),
            GGMLType::Q4_2 => write!(f, "Q4_2 (UNSUPPORTED)"),
            GGMLType::Q4_3 => write!(f, "Q4_3 (UNSUPPORTED)"),
            GGMLType::Q5_0 => write!(f, "Q5_0"),
            GGMLType::Q5_1 => write!(f, "Q5_1"),
            GGMLType::Q8_0 => write!(f, "Q8_0"),
            GGMLType::Q8_1 => write!(f, "Q8_1"),
            GGMLType::Q2_K => write!(f, "Q2_K"),
            GGMLType::Q3_K => write!(f, "Q3_K"),
            GGMLType::Q4_K => write!(f, "Q4_K"),
            GGMLType::Q5_K => write!(f, "Q5_K"),
            GGMLType::Q6_K => write!(f, "Q6_K"),
            GGMLType::Q8_K => write!(f, "Q8_K"),
            GGMLType::IQ2_XXS => write!(f, "IQ2_XXS"),
            GGMLType::IQ2_XS => write!(f, "IQ2_XS"),
            GGMLType::IQ3_XXS => write!(f, "IQ3_XXS"),
            GGMLType::IQ1_S => write!(f, "IQ1_S"),
            GGMLType::IQ4_NL => write!(f, "IQ4_NL"),
            GGMLType::IQ3_S => write!(f, "IQ3_S"),
            GGMLType::IQ2_S => write!(f, "IQ2_S"),
            GGMLType::IQ4_XS => write!(f, "IQ4_XS"),
            GGMLType::I8 => write!(f, "I8"),
            GGMLType::I16 => write!(f, "I16"),
            GGMLType::I32 => write!(f, "I32"),
            GGMLType::I64 => write!(f, "I64"),
            GGMLType::F64 => write!(f, "F64"),
            GGMLType::IQ1_M => write!(f, "IQ1_M"),
            GGMLType::BF16 => write!(f, "BF16"),
            GGMLType::Q4_0_4_4 => write!(f, "Q4_0_4_4 (UNSUPPORTED)"),
            GGMLType::Q4_0_4_8 => write!(f, "Q4_0_4_8 (UNSUPPORTED)"),
            GGMLType::Q4_0_8_8 => write!(f, "Q4_0_8_8 (UNSUPPORTED)"),
            GGMLType::TQ1_0 => write!(f, "TQ1_0"),
            GGMLType::TQ2_0 => write!(f, "TQ2_0"),
            GGMLType::IQ4_NL_4_4 => write!(f, "IQ4_NL_4_4 (UNSUPPORTED)"),
            GGMLType::IQ4_NL_4_8 => write!(f, "IQ4_NL_4_8 (UNSUPPORTED)"),
            GGMLType::IQ4_NL_8_8 => write!(f, "IQ4_NL_8_8 (UNSUPPORTED)"),
            GGMLType::MXFP4 => write!(f, "MXFP4"),
            GGMLType::Count => write!(f, "Count"),
        }
    }
}

impl TryFrom<u32> for GGMLType {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> std::prelude::v1::Result<Self, Self::Error> {
        Ok(match value {
            0 => GGMLType::F32,
            1 => GGMLType::F16,
            2 => GGMLType::Q4_0,
            3 => GGMLType::Q4_1,
            6 => GGMLType::Q5_0,
            7 => GGMLType::Q5_1,
            8 => GGMLType::Q8_0,
            9 => GGMLType::Q8_1,
            10 => GGMLType::Q2_K,
            11 => GGMLType::Q3_K,
            12 => GGMLType::Q4_K,
            13 => GGMLType::Q5_K,
            14 => GGMLType::Q6_K,
            15 => GGMLType::Q8_K,
            16 => GGMLType::IQ2_XXS,
            17 => GGMLType::IQ2_XS,
            18 => GGMLType::IQ3_XXS,
            19 => GGMLType::IQ1_S,
            20 => GGMLType::IQ4_NL,
            21 => GGMLType::IQ3_S,
            22 => GGMLType::IQ2_S,
            23 => GGMLType::IQ4_XS,
            24 => GGMLType::I8,
            25 => GGMLType::I16,
            26 => GGMLType::I32,
            27 => GGMLType::I64,
            28 => GGMLType::F64,
            29 => GGMLType::IQ1_M,
            30 => GGMLType::BF16,
            31 => GGMLType::Q4_0_4_4,
            32 => GGMLType::Q4_0_4_8,
            33 => GGMLType::Q4_0_8_8,
            34 => GGMLType::TQ1_0,
            35 => GGMLType::TQ2_0,
            36 => GGMLType::IQ4_NL_4_4,
            37 => GGMLType::IQ4_NL_4_8,
            38 => GGMLType::IQ4_NL_8_8,
            39 => GGMLType::MXFP4,
            _ => return Err(anyhow!("invalid GGML type")),
        })
    }
}

fn validate_counts(num_tensors: u64, num_kv: u64) -> Result<()> {
    if num_tensors > MAX_TENSORS {
        return Err(anyhow!("tensor count {num_tensors} exceeds cap {MAX_TENSORS}"));
    }
    if num_kv > MAX_KV {
        return Err(anyhow!("kv count {num_kv} exceeds cap {MAX_KV}"));
    }
    Ok(())
}

fn detect_magic(magic: [u8; 4]) -> Result<ByteOrder> {
    match i32::from_le_bytes(magic) {
        FILE_MAGIC_GGUF_LE => Ok(ByteOrder::LE),
        FILE_MAGIC_GGUF_BE => Ok(ByteOrder::BE),
        _ => Err(anyhow!("invalid file magic: not a GGUF file")),
    }
}

fn value_as_u64(v: &Value) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_i64().and_then(|i| u64::try_from(i).ok()))
}

fn ggml_type_layout(kind: GGMLType) -> Result<(u64, u64)> {
    const K: u64 = 256;
    let (block_size, type_size) = match kind {
        GGMLType::F32 => (1, 4),
        GGMLType::F16 => (1, 2),
        GGMLType::Q4_0 => (32, 2 + 32 / 2),
        GGMLType::Q4_1 => (32, 2 + 2 + 32 / 2),
        GGMLType::Q4_2 | GGMLType::Q4_3 => (0, 0),
        GGMLType::Q5_0 => (32, 2 + 4 + 32 / 2),
        GGMLType::Q5_1 => (32, 2 + 2 + 4 + 32 / 2),
        GGMLType::Q8_0 => (32, 2 + 32),
        GGMLType::Q8_1 => (32, 4 + 4 + 32),
        GGMLType::Q2_K => (K, K / 16 + K / 4 + 2 + 2),
        GGMLType::Q3_K => (K, K / 8 + K / 4 + 12 + 2),
        GGMLType::Q4_K => (K, 2 + 2 + 12 + K / 2),
        GGMLType::Q5_K => (K, 2 + 2 + 12 + K / 8 + K / 2),
        GGMLType::Q6_K => (K, K / 2 + K / 4 + K / 16 + 2),
        GGMLType::Q8_K => (K, 4 + K + K / 16 * 2),
        GGMLType::IQ2_XXS => (K, 2 + K / 8 * 2),
        GGMLType::IQ2_XS => (K, 2 + K / 8 * 2 + K / 32),
        GGMLType::IQ3_XXS => (K, 2 + 3 * (K / 8)),
        GGMLType::IQ1_S => (K, 2 + K / 8 + K / 16),
        GGMLType::IQ4_NL => (32, 2 + 16),
        GGMLType::IQ3_S => (K, 2 + 13 * (K / 32) + K / 64),
        GGMLType::IQ2_S => (K, 2 + K / 4 + K / 16),
        GGMLType::IQ4_XS => (K, 2 + 2 + K / 64 + K / 2),
        GGMLType::I8 => (1, 1),
        GGMLType::I16 => (1, 2),
        GGMLType::I32 => (1, 4),
        GGMLType::I64 => (1, 8),
        GGMLType::F64 => (1, 8),
        GGMLType::IQ1_M => (K, K / 8 + K / 16 + K / 32),
        GGMLType::BF16 => (1, 2),
        GGMLType::Q4_0_4_4
        | GGMLType::Q4_0_4_8
        | GGMLType::Q4_0_8_8
        | GGMLType::IQ4_NL_4_4
        | GGMLType::IQ4_NL_4_8
        | GGMLType::IQ4_NL_8_8 => (0, 0),
        GGMLType::TQ1_0 => (K, 2 + K / 64 + (K - 4 * (K / 64)) / 5),
        GGMLType::TQ2_0 => (K, 2 + K / 4),
        GGMLType::MXFP4 => (32, 17),
        GGMLType::Count => (0, 0),
    };
    if block_size == 0 || type_size == 0 {
        return Err(anyhow!("unsupported or removed GGML tensor type"));
    }
    Ok((block_size, type_size))
}

fn skip_exact(mut reader: impl std::io::Read, mut len: u64) -> Result<()> {
    let mut buf = [0u8; 8192];
    while len > 0 {
        let n = usize::try_from(len.min(buf.len() as u64)).unwrap();
        reader.read_exact(&mut buf[..n])?;
        len -= n as u64;
    }
    Ok(())
}

struct CountingReader<R> {
    inner: R,
    bytes_read: u64,
}

impl<R> CountingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            bytes_read: 0,
        }
    }

    fn bytes_read(&self) -> u64 {
        self.bytes_read
    }
}

impl<R: std::io::Read> std::io::Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.bytes_read = self.bytes_read.saturating_add(n as u64);
        Ok(n)
    }
}

struct BoundedReader<R> {
    inner: R,
    remaining: u64,
}

impl<R> BoundedReader<R> {
    fn new(inner: R, remaining: u64) -> Self {
        Self { inner, remaining }
    }
}

impl<R: std::io::Read> std::io::Read for BoundedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let n = usize::try_from(self.remaining.min(buf.len() as u64)).unwrap();
        let read = self.inner.read(&mut buf[..n])?;
        self.remaining -= read as u64;
        Ok(read)
    }
}

impl GGUFModel {
    /// Decode the GGUF file.
    pub(crate) fn decode(
        &mut self,
        reader: impl std::io::Read,
        input_bounds: InputBounds,
        header_preamble_len: u64,
    ) -> Result<()> {
        let mut reader = CountingReader::new(reader);
        let mut seen_tensor_names =
            std::collections::HashSet::with_capacity(self.num_tensor() as usize);
        // decode kv
        for _i in 0..self.num_kv() {
            let key =
                self.read_string_limited(&mut reader, MAX_METADATA_KEY_LEN, "metadata key", false)?;
            if self.kv.contains_key(&key) {
                return Err(anyhow!("duplicate metadata key {key:?}"));
            }
            let value_type: MetadataValueType = self.read_u32(&mut reader)?.try_into()?;
            let value = match value_type {
                MetadataValueType::Uint8 => Value::from(self.read_u8(&mut reader)?),
                MetadataValueType::Int8 => Value::from(self.read_i8(&mut reader)?),
                MetadataValueType::Uint16 => Value::from(self.read_u16(&mut reader)?),
                MetadataValueType::Int16 => Value::from(self.read_i16(&mut reader)?),
                MetadataValueType::Uint32 => Value::from(self.read_u32(&mut reader)?),
                MetadataValueType::Int32 => Value::from(self.read_i32(&mut reader)?),
                MetadataValueType::Float32 => Value::from(self.read_f32(&mut reader)?),
                MetadataValueType::Bool => Value::from(self.read_bool(&mut reader)?),
                MetadataValueType::String => Value::from(self.read_string_limited(
                    &mut reader,
                    MAX_STRING_VALUE_LEN,
                    "metadata string value",
                    true,
                )?),
                MetadataValueType::Array => Value::from(self.read_array(&mut reader)?),
                MetadataValueType::Uint64 => Value::from(self.read_u64(&mut reader)?),
                MetadataValueType::Int64 => Value::from(self.read_i64(&mut reader)?),
                MetadataValueType::Float64 => Value::from(self.read_f64(&mut reader)?),
            };
            #[cfg(feature = "debug")]
            {
                debug!("kv [{}] vtype {:?} key={}, value={}", _i, value_type, key, value);
            }
            self.kv_types.insert(key.clone(), value_type);
            self.kv.insert(key, value);
            if reader.bytes_read() > MAX_METADATA_BYTES {
                return Err(anyhow!("metadata section exceeds cap {} bytes", MAX_METADATA_BYTES));
            }
        }

        let alignment = self.alignment()?;

        // decode tensors
        for _ in 0..self.num_tensor() {
            let name =
                self.read_string_limited(&mut reader, MAX_TENSOR_NAME_LEN, "tensor name", false)?;
            if !seen_tensor_names.insert(name.clone()) {
                return Err(anyhow!("duplicate tensor name {name:?}"));
            }
            let dims = self.read_u32(&mut reader)?;
            if dims == 0 || dims > 4 {
                return Err(anyhow!("tensor {name:?} declares {dims} dimensions (must be 1..=4)"));
            }
            let mut shape = [1; 4];
            let mut elements = 1u64;
            for i in 0..dims {
                let dim = self.read_u64(&mut reader)?;
                if dim == 0 {
                    return Err(anyhow!("tensor {name:?} has a zero-length dimension"));
                }
                if dim > MAX_DIMENSION {
                    return Err(anyhow!("tensor {name:?} dimension {dim} exceeds {MAX_DIMENSION}"));
                }
                elements = elements
                    .checked_mul(dim)
                    .ok_or_else(|| anyhow!("tensor {name:?} dimension product overflows u64"))?;
                shape[i as usize] = dim;
            }
            if elements > MAX_ELEMENTS {
                return Err(anyhow!(
                    "tensor {name:?} element count {elements} exceeds {MAX_ELEMENTS}"
                ));
            }

            let kind = self.read_u32(&mut reader)?;
            let offset = self.read_u64(&mut reader)?;
            let ggml_type_kind: GGMLType = kind.try_into()?;
            let (block_size, type_size) = ggml_type_layout(ggml_type_kind)?;
            let row_elems = shape[0];
            if row_elems % block_size != 0 {
                return Err(anyhow!(
                    "tensor {name:?} row width {row_elems} is not divisible by block size {block_size}"
                ));
            }
            if offset % alignment != 0 {
                return Err(anyhow!(
                    "tensor {name:?} offset {offset} is not aligned to {alignment}"
                ));
            }

            let row_count = shape[1]
                .checked_mul(shape[2])
                .and_then(|v| v.checked_mul(shape[3]))
                .ok_or_else(|| anyhow!("tensor {name:?} row count overflows u64"))?;
            let row_size = row_elems
                .checked_div(block_size)
                .and_then(|blocks| blocks.checked_mul(type_size))
                .ok_or_else(|| anyhow!("tensor {name:?} byte size overflows u64"))?;
            let size = row_size
                .checked_mul(row_count)
                .ok_or_else(|| anyhow!("tensor {name:?} byte size overflows u64"))?;

            self.tensors.push(Tensor {
                name,
                kind,
                offset,
                size,
                shape: shape.to_vec(),
            });

            self.parameters = self
                .parameters
                .checked_add(elements)
                .ok_or_else(|| anyhow!("total parameter count overflows u64"))?;

            if reader.bytes_read() > MAX_HEADER_BYTES {
                return Err(anyhow!("GGUF header exceeds cap {} bytes", MAX_HEADER_BYTES));
            }
        }

        let tensor_data_start = header_preamble_len
            .checked_add(reader.bytes_read())
            .ok_or_else(|| anyhow!("header length overflows u64"))?;
        let tensor_data_start = tensor_data_start
            .checked_add(alignment - 1)
            .ok_or_else(|| anyhow!("aligned tensor data start overflows u64"))?
            / alignment
            * alignment;

        self.validate_tensor_ranges(tensor_data_start, input_bounds)?;

        Ok(())
    }

    fn alignment(&self) -> Result<u64> {
        let raw = match self.kv.get("general.alignment") {
            Some(v) => {
                match self.kv_types.get("general.alignment") {
                    Some(
                        MetadataValueType::Uint8
                        | MetadataValueType::Uint16
                        | MetadataValueType::Uint32
                        | MetadataValueType::Uint64,
                    ) => {}
                    _ => return Err(anyhow!("general.alignment is missing or has invalid type")),
                }
                value_as_u64(v)
                    .ok_or_else(|| anyhow!("general.alignment is missing or has invalid type"))?
            }
            None => GGUF_DEFAULT_ALIGNMENT,
        };
        if raw == 0 {
            return Err(anyhow!("general.alignment is 0"));
        }
        if raw % 8 != 0 {
            return Err(anyhow!("general.alignment {raw} is not a multiple of 8"));
        }
        if raw > MAX_ALIGNMENT {
            return Err(anyhow!("general.alignment {raw} exceeds cap {MAX_ALIGNMENT}"));
        }
        Ok(raw)
    }

    fn validate_tensor_ranges(
        &self,
        tensor_data_start: u64,
        input_bounds: InputBounds,
    ) -> Result<()> {
        let mut ranges: Vec<(&str, u64, u64)> = self
            .tensors
            .iter()
            .map(|t| {
                let abs_start = tensor_data_start
                    .checked_add(t.offset)
                    .ok_or_else(|| anyhow!("tensor {:?} absolute offset overflows u64", t.name))?;
                let abs_end = abs_start
                    .checked_add(t.size)
                    .ok_or_else(|| anyhow!("tensor {:?} byte range overflows u64", t.name))?;
                if let InputBounds::Known(input_len) = input_bounds {
                    if abs_end > input_len {
                        return Err(anyhow!(
                            "tensor {:?} extends past EOF: end={} file_size={}",
                            t.name,
                            abs_end,
                            input_len
                        ));
                    }
                }
                Ok((t.name.as_str(), abs_start, abs_end))
            })
            .collect::<Result<_>>()?;
        ranges.sort_by_key(|(_, start, _)| *start);
        for w in ranges.windows(2) {
            let (a_name, _a_start, a_end) = w[0];
            let (b_name, b_start, _b_end) = w[1];
            if a_end > b_start {
                return Err(anyhow!(
                    "tensors {a_name:?} and {b_name:?} have overlapping data ranges"
                ));
            }
        }
        Ok(())
    }

    fn read_u8(&self, mut reader: impl std::io::Read) -> Result<u8> {
        Ok(reader.read_u8()?)
    }

    fn read_u32(&self, mut reader: impl std::io::Read) -> Result<u32> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_u32::<LittleEndian>()?,
            ByteOrder::BE => reader.read_u32::<BigEndian>()?,
        })
    }

    fn read_f32(&self, mut reader: impl std::io::Read) -> Result<f32> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_f32::<LittleEndian>()?,
            ByteOrder::BE => reader.read_f32::<BigEndian>()?,
        })
    }

    fn read_f64(&self, mut reader: impl std::io::Read) -> Result<f64> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_f64::<LittleEndian>()?,
            ByteOrder::BE => reader.read_f64::<BigEndian>()?,
        })
    }

    fn read_u64(&self, mut reader: impl std::io::Read) -> Result<u64> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_u64::<LittleEndian>()?,
            ByteOrder::BE => reader.read_u64::<BigEndian>()?,
        })
    }

    fn read_i8(&self, mut reader: impl std::io::Read) -> Result<i8> {
        Ok(reader.read_i8()?)
    }

    fn read_u16(&self, mut reader: impl std::io::Read) -> Result<u16> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_u16::<LittleEndian>()?,
            ByteOrder::BE => reader.read_u16::<BigEndian>()?,
        })
    }

    fn read_i16(&self, mut reader: impl std::io::Read) -> Result<i16> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_i16::<LittleEndian>()?,
            ByteOrder::BE => reader.read_i16::<BigEndian>()?,
        })
    }

    fn read_i32(&self, mut reader: impl std::io::Read) -> Result<i32> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_i32::<LittleEndian>()?,
            ByteOrder::BE => reader.read_i32::<BigEndian>()?,
        })
    }

    fn read_i64(&self, mut reader: impl std::io::Read) -> Result<i64> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_i64::<LittleEndian>()?,
            ByteOrder::BE => reader.read_i64::<BigEndian>()?,
        })
    }

    fn read_bool(&self, mut reader: impl std::io::Read) -> Result<bool> {
        match reader.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(anyhow!("invalid bool value {other}; expected 0 or 1")),
        }
    }

    fn read_string_limited(
        &self,
        mut reader: impl std::io::Read,
        max_len: u64,
        context: &str,
        allow_empty: bool,
    ) -> Result<String> {
        let name_len = self.read_version_size(&mut reader)?;
        if name_len == 0 && !allow_empty {
            return Err(anyhow!("{context} has zero length"));
        }
        if name_len > max_len {
            return Err(anyhow!("{context} length {name_len} exceeds cap {max_len}"));
        }
        let len = usize::try_from(name_len)
            .map_err(|_| anyhow!("{context} length {name_len} does not fit in usize"))?;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(len)
            .map_err(|e| anyhow!("failed to reserve {context} buffer ({len} bytes): {e}"))?;
        buffer.resize(len, 0);
        reader.read_exact(&mut buffer)?;
        String::from_utf8(buffer).map_err(|e| anyhow!("{context} is not valid UTF-8: {e}"))
    }

    fn read_array<R: std::io::Read>(&self, reader: &mut CountingReader<R>) -> Result<Vec<Value>> {
        let mut data = Vec::new();
        let item_type: MetadataValueType = self.read_u32(&mut *reader)?.try_into()?;
        let array_len = self.read_version_size(&mut *reader)?;
        if array_len > MAX_ARRAY_LEN {
            return Err(anyhow!("array length {array_len} exceeds cap {MAX_ARRAY_LEN}"));
        }
        let read_count = usize::try_from(
            array_len
                .min(self.max_array_size)
                .min(MAX_STORED_ARRAY_ITEMS),
        )
        .map_err(|_| anyhow!("array storage length does not fit in usize"))?;
        data.try_reserve_exact(read_count)
            .map_err(|e| anyhow!("failed to reserve array buffer ({read_count} items): {e}"))?;
        for i in 0..array_len {
            if data.len() < read_count {
                let value = self.read_array_value(reader, &item_type)?;
                data.push(value);
            } else {
                self.skip_array_value(reader, &item_type)
                    .map_err(|e| anyhow!("failed to skip array item {i}: {e}"))?;
            }
            if reader.bytes_read() > MAX_METADATA_BYTES {
                return Err(anyhow!("metadata section exceeds cap {} bytes", MAX_METADATA_BYTES));
            }
        }

        Ok(data)
    }

    fn read_array_value<R: std::io::Read>(
        &self,
        reader: &mut CountingReader<R>,
        item_type: &MetadataValueType,
    ) -> Result<Value> {
        Ok(match item_type {
            MetadataValueType::Uint8 => Value::from(self.read_u8(reader)?),
            MetadataValueType::Int8 => Value::from(self.read_i8(reader)?),
            MetadataValueType::Uint16 => Value::from(self.read_u16(reader)?),
            MetadataValueType::Int16 => Value::from(self.read_i16(reader)?),
            MetadataValueType::Uint32 => Value::from(self.read_u32(reader)?),
            MetadataValueType::Int32 => Value::from(self.read_i32(reader)?),
            MetadataValueType::Float32 => Value::from(self.read_f32(reader)?),
            MetadataValueType::Bool => Value::from(self.read_bool(reader)?),
            MetadataValueType::String => Value::from(self.read_string_limited(
                reader,
                MAX_STRING_VALUE_LEN,
                "array string value",
                true,
            )?),
            MetadataValueType::Uint64 => Value::from(self.read_u64(reader)?),
            MetadataValueType::Int64 => Value::from(self.read_i64(reader)?),
            MetadataValueType::Float64 => Value::from(self.read_f64(reader)?),
            MetadataValueType::Array => return Err(anyhow!("unsupported item value type: Array")),
        })
    }

    fn skip_array_value<R: std::io::Read>(
        &self,
        reader: &mut CountingReader<R>,
        item_type: &MetadataValueType,
    ) -> Result<()> {
        match item_type {
            MetadataValueType::Bool => {
                let _ = self.read_bool(reader)?;
                Ok(())
            }
            MetadataValueType::Uint8 | MetadataValueType::Int8 => skip_exact(reader, 1),
            MetadataValueType::Uint16 | MetadataValueType::Int16 => skip_exact(reader, 2),
            MetadataValueType::Uint32 | MetadataValueType::Int32 | MetadataValueType::Float32 => {
                skip_exact(reader, 4)
            }
            MetadataValueType::Uint64 | MetadataValueType::Int64 | MetadataValueType::Float64 => {
                skip_exact(reader, 8)
            }
            MetadataValueType::String => {
                let _ = self.read_string_limited(
                    reader,
                    MAX_STRING_VALUE_LEN,
                    "array string value",
                    true,
                )?;
                Ok(())
            }
            MetadataValueType::Array => Err(anyhow!("unsupported item value type: Array")),
        }
    }

    fn read_version_size(&self, mut reader: impl std::io::Read) -> Result<u64> {
        Ok(match self.version.borrow() {
            Version::V1(_) => self.read_u32(&mut reader)? as u64,
            Version::V2(_) => self.read_u64(&mut reader)?,
            Version::V3(_) => self.read_u64(&mut reader)?,
        })
    }

    /// Get the version of the decoded GGUF model.
    ///
    /// Returns one of: "v1", "v2", or "v3".
    pub fn get_version(&self) -> String {
        match &self.version {
            Version::V1(_) => String::from("v1"),
            Version::V2(_) => String::from("v2"),
            Version::V3(_) => String::from("v3"),
        }
    }

    /// Get the number of key-value pairs in the GGUF file.
    pub fn num_kv(&self) -> u64 {
        match &self.version {
            Version::V1(v1) => v1.num_kv as u64,
            Version::V2(v2) => v2.num_kv,
            Version::V3(v3) => v3.num_kv,
        }
    }

    /// Get the number of tensors in the GGUF file.
    ///
    /// Returns the total count of tensors stored in the model.
    pub fn num_tensor(&self) -> u64 {
        match &self.version {
            Version::V1(v1) => v1.num_tensor as u64,
            Version::V2(v2) => v2.num_tensor,
            Version::V3(v3) => v3.num_tensor,
        }
    }

    /// Get the model family/architecture of the GGUF file.
    ///
    /// Returns the value of `general.architecture` metadata key,
    /// or "unknown" if not present.
    ///
    /// Common values include: "llama", "phi", "mistral", "qwen", etc.
    pub fn model_family(&self) -> String {
        let arch = self
            .kv
            .get("general.architecture")
            .cloned()
            .unwrap_or(Value::from("unknown"));

        match arch {
            Value::String(arch) => arch,
            _ => String::from("unknown"),
        }
    }

    /// Get the estimated number of parameters in the model.
    ///
    /// Returns a human-readable string (e.g., "7B", "13B", "192").
    /// Returns "unknown" if parameters cannot be determined.
    pub fn model_parameters(&self) -> String {
        if self.parameters > 0 {
            human_number(self.parameters)
        } else {
            String::from("unknown")
        }
    }

    /// Get the quantization file type of the GGUF file.
    ///
    /// Returns a human-readable description of the quantization method
    /// (e.g., "All F32", "Mostly Q4_0", "Mostly BF16").
    /// Returns "unknown" if not present.
    pub fn file_type(&self) -> String {
        if let Some(ft) = self.kv.get("general.file_type") {
            ft.as_u64()
                .map(file_type)
                .unwrap_or_else(|| "unknown".into())
        } else {
            String::from("unknown")
        }
    }

    /// Get the key-value metadata of the GGUF file.
    ///
    /// Returns a reference to the metadata map containing all key-value pairs
    /// from the GGUF file. Values are JSON values for flexibility.
    ///
    /// Common keys include:
    /// - `general.architecture`: Model architecture (e.g., "llama")
    /// - `general.name`: Model name
    /// - `tokenizer.ggml.tokens`: Tokenizer vocabulary
    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.kv
    }

    /// Get the tensors of the GGUF file.
    ///
    /// Returns a reference to the vector of tensors, each containing
    /// name, type, offset, size, and shape information.
    pub fn tensors(&self) -> &Vec<Tensor> {
        &self.tensors
    }
}

/// Get a `GGUFContainer` from a file, truncating all arrays to length 3.
///
/// # Errors
///
/// Returns an error if:
/// - The file does not exist
/// - The file has an unsupported format (ggml, ggmf, ggjt, ggla)
/// - The file has an invalid magic number
/// - An I/O error occurs while reading the file
///
/// # Examples
///
/// ```rust,no_run
/// use gguf_rs::get_gguf_container;
///
/// let container = get_gguf_container("model.gguf")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn get_gguf_container(file: &str) -> Result<GGUFContainer> {
    get_gguf_container_array_size_with_limit(file, 3, DEFAULT_MAX_HELPER_INPUT_BYTES)
}

/// Get a `GGUFContainer` from a file with the provided max array size.
///
/// # Arguments
///
/// * `file` - Path to the GGUF file
/// * `max_array_size` - Maximum number of elements to read from array metadata
///
/// # Errors
///
/// Returns an error if:
/// - The file does not exist
/// - The file has an unsupported format (ggml, ggmf, ggjt, ggla)
/// - The file has an invalid magic number
/// - An I/O error occurs while reading the file
///
/// # Examples
///
/// ```rust,no_run
/// use gguf_rs::get_gguf_container_array_size;
///
/// // Read all array elements
/// let container = get_gguf_container_array_size("model.gguf", u64::MAX)?;
///
/// // Limit arrays to 100 elements for performance
/// let container = get_gguf_container_array_size("model.gguf", 100)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn get_gguf_container_array_size(file: &str, max_array_size: u64) -> Result<GGUFContainer> {
    get_gguf_container_array_size_with_limit(file, max_array_size, DEFAULT_MAX_HELPER_INPUT_BYTES)
}

pub fn get_gguf_container_array_size_with_limit(
    file: &str,
    max_array_size: u64,
    max_input_bytes: u64,
) -> Result<GGUFContainer> {
    if !std::path::Path::new(file).exists() {
        return Err(anyhow!("file not found"));
    }
    let mut reader = std::fs::File::open(file)?;
    let input_len = reader.metadata()?.len();
    if input_len > max_input_bytes {
        return Err(anyhow!(
            "file size {} exceeds helper input cap {} bytes",
            input_len,
            max_input_bytes
        ));
    }
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    detect_magic(magic)?;
    use std::io::Seek;
    reader.seek(std::io::SeekFrom::Start(0))?;
    let len = usize::try_from(input_len)
        .map_err(|_| anyhow!("file too large to snapshot into memory on this platform"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|e| anyhow!("failed to reserve helper snapshot ({len} bytes): {e}"))?;
    bytes.resize(len, 0);
    reader.read_exact(&mut bytes)?;
    let cursor = std::io::Cursor::new(bytes);
    Ok(GGUFContainer::new(Box::new(cursor), max_array_size)?.with_input_len(input_len))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    fn push_string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn decode_bytes(bytes: Vec<u8>) -> anyhow::Result<super::GGUFModel> {
        use std::io::Cursor;
        let len = bytes.len() as u64;
        let cursor = Cursor::new(bytes);
        let mut container =
            super::GGUFContainer::new_after_magic(super::ByteOrder::LE, Box::new(cursor), 3)
                .with_input_len(len + 4);
        container.decode()
    }

    fn empty_v3(num_tensors: u64, num_kv: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&super::GGUF_VERSION_V3.to_le_bytes());
        b.extend_from_slice(&num_tensors.to_le_bytes());
        b.extend_from_slice(&num_kv.to_le_bytes());
        b
    }

    fn one_tensor_v3(name: &str, dims: &[u64], kind: u32, offset: u64) -> Vec<u8> {
        let mut b = empty_v3(1, 0);
        push_string(&mut b, name);
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &dim in dims {
            b.extend_from_slice(&dim.to_le_bytes());
        }
        b.extend_from_slice(&kind.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        b
    }

    fn one_kv_v3(key: &str, value_type: u32, value_bytes: &[u8]) -> Vec<u8> {
        let mut b = empty_v3(0, 1);
        push_string(&mut b, key);
        b.extend_from_slice(&value_type.to_le_bytes());
        b.extend_from_slice(value_bytes);
        b
    }

    enum TestKv<'a> {
        U64(&'a str, u64),
        I64(&'a str, i64),
        StrLen(&'a str, u64),
        Bool(&'a str, u8),
    }

    struct TestTensor<'a> {
        name: &'a str,
        dims: &'a [u64],
        kind: u32,
        offset: u64,
    }

    fn build_v3(kvs: &[TestKv<'_>], tensors: &[TestTensor<'_>]) -> Vec<u8> {
        let mut b = empty_v3(tensors.len() as u64, kvs.len() as u64);
        for kv in kvs {
            match kv {
                TestKv::U64(key, value) => {
                    push_string(&mut b, key);
                    b.extend_from_slice(&10u32.to_le_bytes());
                    b.extend_from_slice(&value.to_le_bytes());
                }
                TestKv::I64(key, value) => {
                    push_string(&mut b, key);
                    b.extend_from_slice(&11u32.to_le_bytes());
                    b.extend_from_slice(&value.to_le_bytes());
                }
                TestKv::StrLen(key, len) => {
                    push_string(&mut b, key);
                    b.extend_from_slice(&8u32.to_le_bytes());
                    b.extend_from_slice(&len.to_le_bytes());
                }
                TestKv::Bool(key, value) => {
                    push_string(&mut b, key);
                    b.extend_from_slice(&7u32.to_le_bytes());
                    b.push(*value);
                }
            }
        }
        for tensor in tensors {
            push_string(&mut b, tensor.name);
            b.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
            for &dim in tensor.dims {
                b.extend_from_slice(&dim.to_le_bytes());
            }
            b.extend_from_slice(&tensor.kind.to_le_bytes());
            b.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        b
    }

    #[test]
    fn test_read_le_v3_gguf() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        assert_eq!(model.get_version(), "v3");
        assert_eq!(model.model_family(), "llama");
        assert_eq!(model.file_type(), "unknown");
        assert_eq!(model.model_parameters(), "192");
        assert_eq!(
            serde_json::to_value(model.kv).unwrap(),
            json!({
                "general.architecture": "llama",
                "llama.block_count": 12,
                "general.alignment": 64,
                "answer": 42,
                "answer_in_float": 42.0,
                "tokenizer.ggml.tokens": ["a", "b", "c"],
            })
        );
    }

    #[test]
    fn test_read_le_v3_gguf_with_tokens() {
        let mut container =
            super::get_gguf_container_array_size("tests/test-le-v3.gguf", u64::MAX).unwrap();
        let model = container.decode().unwrap();
        assert_eq!(model.get_version(), "v3");
        assert_eq!(model.model_family(), "llama");
        assert_eq!(model.file_type(), "unknown");
        assert_eq!(model.model_parameters(), "192");
        println!("{:?}", model.kv);
        assert_eq!(
            serde_json::to_value(model.kv).unwrap(),
            json!({
                "general.architecture": "llama", 
                "llama.block_count": 12, 
                "general.alignment": 64, 
                "answer": 42, 
                "answer_in_float": 42.0,
                "tokenizer.ggml.tokens": ["a", "b", "c", "d", "e"],})
        );
    }

    #[test]
    fn test_file_not_found() {
        let result = super::get_gguf_container("nonexistent.gguf");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_invalid_file_magic() {
        use std::io::Cursor;
        let invalid_data = vec![0x00, 0x00, 0x00, 0x00];
        let cursor = Cursor::new(invalid_data);
        let result = super::GGUFContainer::new(Box::new(cursor), u64::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata_value_type_conversion() {
        use super::MetadataValueType;
        use std::convert::TryFrom;

        assert!(matches!(MetadataValueType::try_from(0), Ok(MetadataValueType::Uint8)));
        assert!(matches!(MetadataValueType::try_from(6), Ok(MetadataValueType::Float32)));
        assert!(matches!(MetadataValueType::try_from(8), Ok(MetadataValueType::String)));
        assert!(MetadataValueType::try_from(100).is_err());
    }

    #[test]
    fn test_ggml_type_conversion() {
        use super::GGMLType;
        use std::convert::TryFrom;

        assert!(matches!(GGMLType::try_from(0), Ok(GGMLType::F32)));
        assert!(matches!(GGMLType::try_from(2), Ok(GGMLType::Q4_0)));
        assert!(GGMLType::try_from(100).is_err());
    }

    #[test]
    fn test_byte_order_default() {
        use super::ByteOrder;
        let bo = ByteOrder::default();
        assert!(matches!(bo, ByteOrder::LE));
    }

    #[test]
    fn test_tensors() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        let tensors = model.tensors();
        assert!(!tensors.is_empty());

        for tensor in tensors {
            assert!(!tensor.name.is_empty());
            assert!(!tensor.shape.is_empty());
        }
    }

    #[test]
    fn test_num_tensor() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        assert!(model.num_tensor() > 0);
    }

    #[test]
    fn test_get_version() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        assert_eq!(container.get_version(), "v1"); // Before decode, default is v1
        let _ = container.decode().unwrap();
        // After decode, version should be v3
    }

    // ========== Additional tests for improved coverage ==========

    #[test]
    fn test_human_number_small() {
        assert_eq!(super::human_number(999), "999");
        assert_eq!(super::human_number(1000), "1000");
        assert_eq!(super::human_number(1001), "1K");
        assert_eq!(super::human_number(1500), "2K");
    }

    #[test]
    fn test_human_number_medium() {
        assert_eq!(super::human_number(1_000_000), "1000K");
        assert_eq!(super::human_number(1_000_001), "1M");
        assert_eq!(super::human_number(2_000_001), "2M");
        assert_eq!(super::human_number(3_500_000), "4M");
    }

    #[test]
    fn test_human_number_large() {
        assert_eq!(super::human_number(1_000_000_000), "1000M");
        assert_eq!(super::human_number(1_000_000_001), "1B");
        assert_eq!(super::human_number(7_500_000_000), "8B");
    }

    #[test]
    fn test_file_type_all_values() {
        assert_eq!(super::file_type(0), "All F32");
        assert_eq!(super::file_type(1), "Mostly F16");
        assert_eq!(super::file_type(2), "Mostly Q4_0");
        assert_eq!(super::file_type(7), "Mostly Q8_0");
        assert_eq!(super::file_type(14), "Mostly Q6_K");
        assert_eq!(super::file_type(24), "Mostly BF16");
        assert_eq!(super::file_type(99), "unknown");
    }

    #[test]
    fn test_metadata_value_type_all_variants() {
        use super::MetadataValueType;
        use std::convert::TryFrom;

        // Test all valid type values
        assert!(matches!(MetadataValueType::try_from(0), Ok(MetadataValueType::Uint8)));
        assert!(matches!(MetadataValueType::try_from(1), Ok(MetadataValueType::Int8)));
        assert!(matches!(MetadataValueType::try_from(2), Ok(MetadataValueType::Uint16)));
        assert!(matches!(MetadataValueType::try_from(3), Ok(MetadataValueType::Int16)));
        assert!(matches!(MetadataValueType::try_from(4), Ok(MetadataValueType::Uint32)));
        assert!(matches!(MetadataValueType::try_from(5), Ok(MetadataValueType::Int32)));
        assert!(matches!(MetadataValueType::try_from(6), Ok(MetadataValueType::Float32)));
        assert!(matches!(MetadataValueType::try_from(7), Ok(MetadataValueType::Bool)));
        assert!(matches!(MetadataValueType::try_from(8), Ok(MetadataValueType::String)));
        assert!(matches!(MetadataValueType::try_from(9), Ok(MetadataValueType::Array)));
        assert!(matches!(MetadataValueType::try_from(10), Ok(MetadataValueType::Uint64)));
        assert!(matches!(MetadataValueType::try_from(11), Ok(MetadataValueType::Int64)));
        assert!(matches!(MetadataValueType::try_from(12), Ok(MetadataValueType::Float64)));
    }

    #[test]
    fn test_ggml_type_all_valid_types() {
        use super::GGMLType;
        use std::convert::TryFrom;

        // Test a representative sample of GGML types
        assert!(matches!(GGMLType::try_from(1), Ok(GGMLType::F16)));
        assert!(matches!(GGMLType::try_from(3), Ok(GGMLType::Q4_1)));
        assert!(matches!(GGMLType::try_from(6), Ok(GGMLType::Q5_0)));
        assert!(matches!(GGMLType::try_from(7), Ok(GGMLType::Q5_1)));
        assert!(matches!(GGMLType::try_from(8), Ok(GGMLType::Q8_0)));
        assert!(matches!(GGMLType::try_from(10), Ok(GGMLType::Q2_K)));
        assert!(matches!(GGMLType::try_from(30), Ok(GGMLType::BF16)));
        assert!(matches!(GGMLType::try_from(39), Ok(GGMLType::MXFP4)));
    }

    #[test]
    fn test_mxfp4_layout_matches_current_ggml() {
        let (block_size, type_size) = super::ggml_type_layout(super::GGMLType::MXFP4).unwrap();
        assert_eq!((block_size, type_size), (32, 17));
    }

    #[test]
    fn test_ggml_type_invalid() {
        use super::GGMLType;
        use std::convert::TryFrom;

        assert!(GGMLType::try_from(100).is_err());
        assert!(GGMLType::try_from(255).is_err());
    }

    #[test]
    fn test_model_family_unknown() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        // This test file has "llama" architecture
        assert_eq!(model.model_family(), "llama");
    }

    #[test]
    fn test_model_parameters_format() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        // Test file has 192 parameters
        assert_eq!(model.model_parameters(), "192");
    }

    #[test]
    fn test_metadata_accessor() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        let metadata = model.metadata();
        assert!(metadata.contains_key("general.architecture"));
        assert!(metadata.contains_key("llama.block_count"));
    }

    #[test]
    fn test_num_kv() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        assert!(model.num_kv() > 0);
    }

    #[test]
    fn test_tensor_properties() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        let tensors = model.tensors();

        for tensor in tensors {
            // Verify tensor has valid properties
            assert!(!tensor.name.is_empty());
            assert!(!tensor.shape.is_empty());
            // Offset and size should be non-negative (u64)
            let _ = tensor.offset;
            let _ = tensor.size;
            let _ = tensor.kind;
        }
    }

    #[test]
    fn test_container_new() {
        use super::GGUFContainer;
        use std::io::Cursor;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&super::FILE_MAGIC_GGUF_LE.to_le_bytes());
        let cursor = Cursor::new(bytes);
        let container = GGUFContainer::new(Box::new(cursor), 100).unwrap();
        assert_eq!(container.get_version(), "v1");
    }

    #[test]
    fn test_byte_order_variants() {
        use super::ByteOrder;

        let le = ByteOrder::LE;
        let be = ByteOrder::BE;

        // Just verify we can create both variants
        let _ = format!("{:?}", le);
        let _ = format!("{:?}", be);
    }

    #[test]
    fn test_version_variants() {
        use super::{Version, V1, V2, V3};

        let v1 = Version::V1(V1::default());
        let v2 = Version::V2(V2::default());
        let v3 = Version::V3(V3::default());

        // Verify we can create all version variants
        let _ = format!("{:?}", v1);
        let _ = format!("{:?}", v2);
        let _ = format!("{:?}", v3);
    }

    #[test]
    fn test_invalid_file_magic_detailed() {
        use std::io::Cursor;

        // Test with various invalid magic numbers
        let invalid_magics = vec![
            vec![0x00, 0x00, 0x00, 0x00],
            vec![0xFF, 0xFF, 0xFF, 0xFF],
            vec![0x12, 0x34, 0x56, 0x78],
        ];

        for magic in invalid_magics {
            let cursor = Cursor::new(magic);
            let result = super::GGUFContainer::new(Box::new(cursor), u64::MAX);
            assert!(result.is_err(), "Expected error for invalid magic");
        }
    }

    #[test]
    fn test_file_not_found_message() {
        let result = super::get_gguf_container("this_file_does_not_exist.gguf");
        assert!(result.is_err());
        // Check error message
        if let Err(err) = result {
            assert!(
                err.to_string().contains("file not found"),
                "Error message should mention 'file not found'"
            );
        }
    }

    #[test]
    fn test_get_gguf_container_array_size() {
        // Test with custom array size
        let result = super::get_gguf_container_array_size("tests/test-le-v3.gguf", 1);
        assert!(result.is_ok());

        let mut container = result.unwrap();
        let model = container.decode().unwrap();

        // With max_array_size=1, arrays should be truncated
        let tokens = model.kv.get("tokenizer.ggml.tokens");
        if let Some(tokens_arr) = tokens {
            if let serde_json::Value::Array(arr) = tokens_arr {
                assert!(arr.len() <= 1, "Array should be truncated to max size");
            }
        }
    }

    #[test]
    fn rejects_tensor_count_over_cap() {
        assert!(decode_bytes(empty_v3(super::MAX_TENSORS + 1, 0)).is_err());
    }

    #[test]
    fn rejects_kv_count_over_cap() {
        assert!(decode_bytes(empty_v3(0, super::MAX_KV + 1)).is_err());
    }

    #[test]
    fn rejects_dims_zero_or_too_large() {
        assert!(decode_bytes(one_tensor_v3("t", &[], 0, 0)).is_err());
        assert!(decode_bytes(one_tensor_v3("t", &[1, 1, 1, 1, 1], 0, 0)).is_err());
    }

    #[test]
    fn rejects_invalid_kind_count() {
        assert!(decode_bytes(one_tensor_v3("t", &[1], 40, 0)).is_err());
    }

    #[test]
    fn rejects_bad_quant_block_alignment() {
        assert!(decode_bytes(one_tensor_v3("t", &[1], 12, 0)).is_err());
    }

    #[test]
    fn rejects_bad_quant_row_width_even_when_total_elements_align() {
        assert!(decode_bytes(one_tensor_v3("t", &[1, 256], 12, 0)).is_err());
    }

    #[test]
    fn rejects_invalid_bool_value() {
        let bytes = build_v3(&[TestKv::Bool("bad.bool", 2)], &[]);
        assert!(decode_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_oversized_metadata_key() {
        let long_key = "k".repeat((super::MAX_METADATA_KEY_LEN + 1) as usize);
        let bytes = one_kv_v3(&long_key, 10, &1u64.to_le_bytes());
        assert!(decode_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_oversized_tensor_name() {
        let long_name = "t".repeat((super::MAX_TENSOR_NAME_LEN + 1) as usize);
        let bytes = one_tensor_v3(&long_name, &[1], 0, 0);
        assert!(decode_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_oversized_metadata_string_length_without_allocating() {
        let bytes = build_v3(
            &[TestKv::StrLen(
                "big.string",
                super::MAX_STRING_VALUE_LEN + 1,
            )],
            &[],
        );
        assert!(decode_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_oversized_array_length() {
        let mut bytes = empty_v3(0, 1);
        push_string(&mut bytes, "big.array");
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&10u32.to_le_bytes());
        bytes.extend_from_slice(&(super::MAX_ARRAY_LEN + 1).to_le_bytes());
        assert!(decode_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_unaligned_tensor_offset() {
        let bytes = build_v3(
            &[TestKv::U64("general.alignment", 64)],
            &[TestTensor {
                name: "t",
                dims: &[1],
                kind: 0,
                offset: 1,
            }],
        );
        assert!(decode_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_alignment_not_multiple_of_eight() {
        let bytes = build_v3(
            &[TestKv::U64("general.alignment", 10)],
            &[TestTensor {
                name: "t",
                dims: &[1],
                kind: 0,
                offset: 0,
            }],
        );
        assert!(decode_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_alignment_wrong_type() {
        let bytes = build_v3(
            &[TestKv::StrLen("general.alignment", 0)],
            &[TestTensor {
                name: "t",
                dims: &[1],
                kind: 0,
                offset: 0,
            }],
        );
        assert!(decode_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_alignment_signed_integer_type() {
        let bytes = build_v3(
            &[TestKv::I64("general.alignment", 16)],
            &[TestTensor {
                name: "t",
                dims: &[1],
                kind: 0,
                offset: 0,
            }],
        );
        assert!(decode_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_overlapping_tensor_ranges() {
        let bytes = build_v3(
            &[],
            &[
                TestTensor {
                    name: "a",
                    dims: &[1],
                    kind: 0,
                    offset: 0,
                },
                TestTensor {
                    name: "b",
                    dims: &[1],
                    kind: 0,
                    offset: 0,
                },
            ],
        );
        assert!(decode_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_tensor_range_past_eof_when_length_known() {
        let bytes = one_tensor_v3("t", &[1], 0, 0);
        assert!(decode_bytes(bytes).is_err());
    }

    #[test]
    fn validates_skipped_bool_array_entries() {
        use std::io::Cursor;
        let mut bytes = empty_v3(0, 1);
        push_string(&mut bytes, "bool.array");
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.push(1);
        bytes.push(2);

        let len = bytes.len() as u64;
        let cursor = Cursor::new(bytes);
        let mut container =
            super::GGUFContainer::new_after_magic(super::ByteOrder::LE, Box::new(cursor), 1)
                .with_input_len(len + 4);
        assert!(container.decode().is_err());
    }

    #[test]
    fn rejects_duplicate_metadata_keys() {
        let mut bytes = empty_v3(0, 2);
        push_string(&mut bytes, "dup");
        bytes.extend_from_slice(&10u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        push_string(&mut bytes, "dup");
        bytes.extend_from_slice(&10u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        assert!(decode_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_duplicate_tensor_names() {
        let bytes = build_v3(
            &[],
            &[
                TestTensor {
                    name: "dup",
                    dims: &[1],
                    kind: 0,
                    offset: 0,
                },
                TestTensor {
                    name: "dup",
                    dims: &[1],
                    kind: 0,
                    offset: 4,
                },
            ],
        );
        assert!(decode_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_invalid_utf8_in_strings() {
        use std::io::Cursor;
        let mut bytes = empty_v3(0, 1);
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.push(0xff);
        bytes.extend_from_slice(&10u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        let len = bytes.len() as u64;
        let cursor = Cursor::new(bytes);
        let mut container =
            super::GGUFContainer::new_after_magic(super::ByteOrder::LE, Box::new(cursor), 3)
                .with_input_len(len + 4);
        assert!(container.decode().is_err());
    }

    #[test]
    fn direct_new_requires_explicit_input_policy() {
        use std::io::Cursor;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&super::FILE_MAGIC_GGUF_LE.to_le_bytes());
        bytes.extend_from_slice(&super::GGUF_VERSION_V3.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let cursor = Cursor::new(bytes);
        let mut container = super::GGUFContainer::new(Box::new(cursor), 3).unwrap();
        assert!(container.decode().is_err());
    }

    #[test]
    fn allow_unbounded_input_opt_in_restores_streaming_decode() {
        use std::io::Cursor;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&super::FILE_MAGIC_GGUF_LE.to_le_bytes());
        bytes.extend_from_slice(&super::GGUF_VERSION_V3.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let cursor = Cursor::new(bytes);
        let mut container = super::GGUFContainer::new(Box::new(cursor), 3)
            .unwrap()
            .allow_unbounded_input();
        assert!(container.decode().is_ok());
    }

    #[test]
    fn malformed_file_type_does_not_panic() {
        let bytes = build_v3(&[TestKv::StrLen("general.file_type", 0)], &[]);
        let model = decode_bytes(bytes).unwrap();
        assert_eq!(model.file_type(), "unknown");
    }

    #[test]
    fn stored_array_items_are_capped_even_for_large_requests() {
        let n = super::MAX_STORED_ARRAY_ITEMS + 10;
        let mut bytes = empty_v3(0, 1);
        push_string(&mut bytes, "big.tokens");
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&10u32.to_le_bytes());
        bytes.extend_from_slice(&n.to_le_bytes());
        for i in 0..n {
            bytes.extend_from_slice(&i.to_le_bytes());
        }

        let len = bytes.len() as u64;
        let cursor = std::io::Cursor::new(bytes);
        let mut container =
            super::GGUFContainer::new_after_magic(super::ByteOrder::LE, Box::new(cursor), u64::MAX)
                .with_input_len(len + 4);
        let model = container.decode().unwrap();
        let arr = model
            .metadata()
            .get("big.tokens")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(arr.len(), super::MAX_STORED_ARRAY_ITEMS as usize);
    }
}

/// Memory-mapped file support (requires `mmap` feature)
#[cfg(feature = "mmap")]
pub mod mmap;

/// Async I/O support (requires `async` feature)
#[cfg(feature = "async")]
pub mod async_io;

/// GGUF file writing support
pub mod writer;
