//! Single-modulus LWE parameters, keys, encryption and key switching.
//!
//! [`LweSecretKey`] encodes messages using [`LweParameters`]. Its `encrypt_to`
//! methods reuse caller-owned ciphertext storage, and decryption accepts both
//! owned and borrowed [`primus_lattice::lwe::Lwe`] values.
//!
//! [`LweSecretKeyRef`] borrows encoded or signed coefficients for raw operations:
//! `encrypt_encoded` encrypts an already scaled residue (zero gives a randomized
//! encryption of zero), and `decrypt_phase` returns `b - <a,s>` without decoding.
//! These operations take the modulus and samplers directly, so ring-key callers
//! can reuse their own noise parameters without constructing LWE parameters.

#![deny(missing_docs)]

mod key_switch;
mod parameter;
mod secret_key;

/// Owned single-message LWE ciphertext.
pub type LweCiphertext<T> = primus_lattice::lwe::Lwe<Vec<T>>;

/// Owned packed multi-message LWE ciphertext.
pub type MultiMsgLweCiphertext<T> = primus_lattice::lwe::MultiMsgLwe<Vec<T>>;

pub use key_switch::LweKeySwitchingKey;
pub use parameter::LweParameters;
pub use primus_distr::SecretKeyDistr;
pub use secret_key::{LweSecretKey, LweSecretKeyRef};

use primus_encoding::{PlaintextEmbedding, RoundedCodec};
