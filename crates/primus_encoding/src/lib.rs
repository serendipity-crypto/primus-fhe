//! Plaintext encoding and decoding for homomorphic encryption.
//!
//! [`RoundedCodec`] rounds each scaled message; [`ScaledCodec`] uses a fixed
//! rounded integer scale. Both support native and explicit ciphertext moduli.
//! The optional `rns` feature provides BFV coefficient scaling and decoding over
//! an RNS ciphertext basis. Public types are available at the crate root.
//! These codecs do not implement integer slot packing or CKKS canonical embedding.
//!
//! The `simd` feature enables nightly SIMD arithmetic in dependencies.

#![deny(missing_docs)]

mod decode;
mod helpers;
mod integer_scale;
mod rounded;
mod scaled;

pub use rounded::RoundedCodec;
pub use scaled::ScaledCodec;

#[cfg(feature = "rns")]
mod bfv_rns;
#[cfg(feature = "rns")]
pub use bfv_rns::BfvRnsCodec;

/// Plaintext embedding used when lifting residues from `Z_t` into the ciphertext modulus.
/// Both variants accept messages as canonical residues in `[0,t)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlaintextEmbedding {
    /// Lifts messages as unsigned residues in `[0, t)`.
    Unsigned,
    /// Uses `m` below `ceil(t/2)` and `m-t` otherwise, giving the centered
    /// interval `[-floor(t/2), ceil(t/2))`. For `t = 2`, residue `1` lifts to `-1`.
    Centered,
}
