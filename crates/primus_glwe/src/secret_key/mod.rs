//! GLWE secret key types organized by domain representation.

mod coeff;
mod fourier;
mod ntt;

pub use coeff::GlweSecretKey;
pub use fourier::{
    FourierGadgetEncryptContext, FourierGlweDecryptContext, FourierGlweEncryptContext,
    FourierGlweSecretKey,
};
pub use ntt::{NttGadgetEncryptContext, NttGlweSecretKey};
