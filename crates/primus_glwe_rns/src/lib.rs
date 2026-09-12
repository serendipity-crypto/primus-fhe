//! CRT/DCRT GLWE operations and Hybrid-RNS key switching.

#![deny(missing_docs)]

mod crt;
mod dcrt;
mod key_switch;
mod parameter;
mod public_key;
mod secret_key;

/// Coefficient-domain GLWE ciphertext over an ordered RNS basis.
pub type CrtGlweCiphertext<T> = primus_lattice::glwe::CrtGlwe<T>;

/// NTT-domain GLWE ciphertext over an ordered RNS basis.
pub type DcrtGlweCiphertext<T> = primus_lattice::glwe::DcrtGlwe<T>;

pub use crt::{
    CrtGlweAutoContext, CrtGlweAutoKey, CrtGlweExpandCoeffContext, CrtGlweExpandCoeffKey,
    CrtGlweExpandCoeffSyncPool, CrtGlweTraceContext, CrtGlweTraceKey,
};
pub use dcrt::{
    DcrtGlweAutoKey, DcrtGlweExpandCoeffContext, DcrtGlweExpandCoeffKey,
    DcrtGlweExpandCoeffSyncPool, DcrtGlweRevTraceContext, DcrtGlweRevTraceKey,
    DcrtGlweTraceContext, DcrtGlweTraceKey,
};
pub use key_switch::{
    DcrtGlweKeySwitchingContext, DcrtGlweKeySwitchingKey, HybridRnsGlweKeySwitchingContext,
    HybridRnsGlweKeySwitchingKey,
};
pub use parameter::{
    CrtGgswParameters, CrtGlevParameters, CrtGlevParametersError, CrtGlweParameters,
    DcrtGadgetDomain, GadgetDomainError, HybridRnsKeySwitchDomain,
};
pub use primus_encoding::BfvRnsCodec;
pub use primus_glwe::{GlweSecretKey, SecretKeyDistr};
pub use primus_lattice::{RnsGadgetSize, RnsGlweSize};
pub use public_key::DcrtGlwePublicKey;
pub use secret_key::{DcrtGlweDecryptContext, DcrtGlweSecretKey};
