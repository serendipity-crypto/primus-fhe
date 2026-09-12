//! Trace keys for coefficient-domain native-torus ciphertexts.
use crate::{
    FourierGadgetEncryptContext, FourierGlweAutomorphismContext, FourierGlweAutomorphismKey,
    FourierGlweSecretKey, GlevParameters, GlweSecretKey, GlweSize,
};
use primus_decompose::primitive::ApproxSignedBasis;
use primus_fft::{FftEngine, FftTable, TorusFftValue};
use primus_lattice::glwe::Glwe;
use primus_modulus::NativeModulus;

/// Reusable workspace for Fourier-key trace and coefficient projection.
pub struct FourierGlweTraceContext<T: TorusFftValue> {
    pub(super) automorphism_output: Glwe<Vec<T>>,
    pub(super) automorphism: FourierGlweAutomorphismContext<T>,
}
impl<T: TorusFftValue> FourierGlweTraceContext<T> {
    /// Allocates workspace for one immutable GLWE layout.
    #[must_use]
    pub fn new(size: GlweSize) -> Self {
        Self {
            automorphism_output: Glwe::zero(size.glwe_len()),
            automorphism: FourierGlweAutomorphismContext::new(size),
        }
    }
}

/// Owns the log2(N) automorphism keys shared by trace, projection and packing.
/// All ciphertext inputs/outputs are coefficient-domain native-torus GLWEs.
#[derive(Clone)]
pub struct FourierGlweTraceKey<T: TorusFftValue> {
    pub(super) automorphism_keys: Vec<FourierGlweAutomorphismKey<T>>,
    pub(super) glwe_size: GlweSize,
}
impl<T: TorusFftValue> FourierGlweTraceKey<T> {
    /// Generates keys for degrees N+1,N/2+1,...,3. Inherits the key consistency,
    /// layout and FFT requirements of [`FourierGlweAutomorphismKey::generate`].
    pub fn generate<Table: FftTable, R: rand::Rng + rand::CryptoRng>(
        secret: &GlweSecretKey<T>,
        fourier_secret: &FourierGlweSecretKey,
        params: &GlevParameters<T, NativeModulus<T>>,
        fft: &mut FftEngine<'_, Table>,
        rng: &mut R,
        context: &mut FourierGadgetEncryptContext<T>,
    ) -> Self {
        let glwe_size = secret.glwe_size();
        let automorphism_keys = (1..=glwe_size.poly_length().trailing_zeros())
            .rev()
            .map(|shift| {
                FourierGlweAutomorphismKey::generate(
                    (1 << shift) + 1,
                    secret,
                    fourier_secret,
                    params,
                    fft,
                    rng,
                    context,
                )
            })
            .collect();
        Self {
            automorphism_keys,
            glwe_size,
        }
    }
    /// Returns the number of available automorphism keys.
    #[must_use]
    pub fn automorphism_count(&self) -> usize {
        self.automorphism_keys.len()
    }
    /// Returns the decomposition basis shared by the keys.
    #[must_use]
    pub fn basis(&self) -> &ApproxSignedBasis<T> {
        self.automorphism_keys[0].basis()
    }
}
