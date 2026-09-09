use primus_integer::{FheUint, Size};
use primus_reduce::RingContext;
use rand::distr::Distribution;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{LweParameters, SecretKeyDistr};

use super::LweSecretKeyRef;

/// Represents a secret key for the Learning with Errors (LWE) cryptographic scheme.
///
/// Secret coefficients are erased when the key (including each clone) is dropped.
/// Message operations use [`LweParameters`] for encoding and sampling; use
/// [`Self::as_view`] to borrow the coefficients for single-ciphertext raw operations.
/// Batch operations, including raw encoded encryption and phase decryption,
/// require this owned key, ensuring that reused single-ciphertext operations
/// always receive encoded coefficients.
/// Construction guarantees that the ciphertext length `dimension() + 1` fits
/// in `usize`.
///
/// # Correctness
///
/// Operations require the key and parameters to have the same dimension and
/// ciphertext modulus. Coefficients must be canonical residues under that
/// modulus, satisfying [`primus_reduce::ReduceDotProduct`]. The modulus is not
/// stored in the key; callers must preserve it when supplying parameters.
#[derive(Clone)]
pub struct LweSecretKey<T: FheUint> {
    data: Vec<T>,
    distr: SecretKeyDistr,
}

impl<T: FheUint> Zeroize for LweSecretKey<T> {
    #[inline]
    fn zeroize(&mut self) {
        self.data.zeroize();
    }
}

impl<T: FheUint> ZeroizeOnDrop for LweSecretKey<T> {}

impl<T: FheUint> Drop for LweSecretKey<T> {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl<T: FheUint> AsRef<[T]> for LweSecretKey<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<T: FheUint> Size for LweSecretKey<T> {
    #[inline]
    fn byte_count(&self) -> usize {
        self.data.byte_count()
    }
}

impl<T: FheUint> LweSecretKey<T> {
    /// Creates a new [`LweSecretKey<T>`].
    ///
    /// # Correctness
    ///
    /// `key` must contain canonical residues for the modulus used in subsequent
    /// operations and satisfy `distr`. Neither condition is checked here.
    ///
    /// # Panics
    ///
    /// Panics if `key.len() + 1` overflows `usize`.
    #[inline]
    pub fn new(key: Vec<T>, distr: SecretKeyDistr) -> Self {
        key.len().checked_add(1).expect("LWE length overflow");
        Self { data: key, distr }
    }

    /// Borrows the canonical secret coefficients for raw LWE operations.
    #[must_use]
    #[inline]
    pub fn as_view(&self) -> LweSecretKeyRef<'_, T> {
        LweSecretKeyRef::Encoded(self.as_ref())
    }

    /// Returns the dimension of this [`LweSecretKey<T>`].
    #[inline]
    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    /// Returns the distribution of this [`LweSecretKey<T>`].
    #[inline]
    pub fn distr(&self) -> SecretKeyDistr {
        self.distr
    }

    /// Generates a new [`LweSecretKey<T>`] with random values.
    #[inline]
    pub fn generate<R, M>(params: &LweParameters<T, M>, rng: &mut R) -> Self
    where
        R: rand::Rng + rand::CryptoRng,
        M: RingContext<T>,
    {
        let distr = params.secret_key_distr();
        let key = match distr {
            SecretKeyDistr::UniformBinary => {
                primus_distr::sample_uniform_binary_values(params.dimension(), rng)
            }
            SecretKeyDistr::Binary { one_probability } => {
                primus_distr::sample_binary_values_with_probability(
                    params.dimension(),
                    one_probability,
                    rng,
                )
            }
            SecretKeyDistr::SparseTernary => primus_distr::sample_sparse_ternary_values(
                params.cipher_modulus_minus_one(),
                params.dimension(),
                rng,
            ),
            SecretKeyDistr::UniformTernary => primus_distr::sample_uniform_ternary_values(
                params.cipher_modulus_minus_one(),
                params.dimension(),
                rng,
            ),
            SecretKeyDistr::Ternary {
                negative_one_probability,
                one_probability,
            } => primus_distr::sample_ternary_values_with_probabilities(
                params.cipher_modulus_minus_one(),
                params.dimension(),
                negative_one_probability,
                one_probability,
                rng,
            ),
            SecretKeyDistr::FixedHammingWeightBinary { hamming_weight } => {
                primus_distr::sample_fixed_hamming_weight_binary_values(
                    params.dimension(),
                    hamming_weight,
                    rng,
                )
            }
            SecretKeyDistr::FixedHammingWeightTernary {
                negative_one_weight,
                one_weight,
            } => primus_distr::sample_fixed_hamming_weight_ternary_values(
                params.cipher_modulus_minus_one(),
                params.dimension(),
                negative_one_weight,
                one_weight,
                rng,
            ),
            SecretKeyDistr::Gaussian(_) => params
                .secret_key_gaussian()
                .expect("validated Gaussian LWE secret-key distribution")
                .sample_iter(rng)
                .take(params.dimension())
                .collect(),
        };
        Self { data: key, distr }
    }
}
