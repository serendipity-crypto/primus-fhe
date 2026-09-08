//! Plaintext encoding and decoding for homomorphic encryption.
//!
//! Single-modulus codecs support native and explicit ciphertext moduli.
//! The optional `rns` feature provides coefficient scaling and decoding over an
//! RNS ciphertext basis. These codecs do not implement integer slot packing or
//! CKKS canonical embedding.
//!
//! The `simd` feature enables nightly SIMD arithmetic in dependencies.

#![deny(missing_docs)]

/// Encoding and decoding under one native or explicit ciphertext modulus.
pub mod single_modulus;
pub use single_modulus::{RoundedCodec, ScaledCodec};

/// RNS coefficient encoding and decoding.
#[cfg(feature = "rns")]
pub mod rns;
#[cfg(feature = "rns")]
pub use rns::BfvRnsCodec;

/// Plaintext embedding used when lifting residues from `Z_t` into the ciphertext modulus.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlaintextEmbedding {
    /// Lifts messages as unsigned residues in `[0, t)`.
    Unsigned,
    /// Lifts messages into the centered interval `[-floor(t/2), ceil(t/2))`.
    Centered,
}
