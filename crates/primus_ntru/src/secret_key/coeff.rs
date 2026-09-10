//! Canonical coefficient-domain NTRU secret key.

use primus_integer::FheUint;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{NtruParameters, SecretKeyDistr};

/// A small signed polynomial `f` shared by all NTRU transform backends.
///
/// Signed coefficients are intentionally stored independently of a ciphertext
/// modulus: `-1` is encoded as `q - 1` for NTT and as the native two's-complement
/// bit pattern for Fourier only when the key is converted to that backend.
/// Conversion to an explicit modulus `q` requires every coefficient's unsigned
/// magnitude to be strictly less than `q`. Generated keys satisfy this bound
/// for their parameter modulus; imported keys and conversions to another
/// modulus retain this caller obligation.
#[derive(Clone)]
pub struct NtruSecretKey<T: FheUint> {
    pub(crate) key: Vec<T::SignedInteger>,
    pub(crate) distr: SecretKeyDistr,
}

impl<T: FheUint> Zeroize for NtruSecretKey<T> {
    #[inline]
    fn zeroize(&mut self) {
        self.key.zeroize();
    }
}

impl<T: FheUint> ZeroizeOnDrop for NtruSecretKey<T> {}

impl<T: FheUint> NtruSecretKey<T> {
    /// Samples a binary prefix and pads the remaining coefficients with zero.
    ///
    /// This is kept internal because transform backends must still reject
    /// candidates that are not invertible in their ciphertext ring.
    pub(crate) fn generate_padded_binary<R>(
        poly_length: usize,
        active_length: usize,
        distr: SecretKeyDistr,
        rng: &mut R,
    ) -> Self
    where
        R: rand::Rng + rand::CryptoRng,
    {
        assert!((1..=poly_length).contains(&active_length));
        debug_assert!(distr.is_binary());
        let mut key = match distr {
            SecretKeyDistr::UniformBinary => {
                primus_distr::sample_uniform_binary_values(active_length, rng)
            }
            SecretKeyDistr::Binary { one_probability } => {
                primus_distr::sample_binary_values_with_probability(
                    active_length,
                    one_probability,
                    rng,
                )
            }
            SecretKeyDistr::FixedHammingWeightBinary { hamming_weight } => {
                primus_distr::sample_fixed_hamming_weight_binary_values(
                    active_length,
                    hamming_weight,
                    rng,
                )
            }
            _ => unreachable!("binary distribution checked above"),
        };
        key.resize(poly_length, T::ZERO.cast_to_signed());
        Self { key, distr }
    }

    /// Creates a coefficient-domain NTRU key from signed values.
    ///
    /// No modulus is attached to this key. Before using an explicit-modulus
    /// backend, the caller must ensure that every coefficient has unsigned
    /// magnitude less than the target modulus. `distr` records the sampling
    /// distribution; it does not validate the supplied coefficients.
    ///
    /// # Panics
    ///
    /// Panics if `key` is empty.
    #[inline]
    pub fn new(key: Vec<T::SignedInteger>, distr: SecretKeyDistr) -> Self {
        assert!(!key.is_empty(), "NTRU secret key must not be empty");
        Self { key, distr }
    }

    /// Returns the coefficient polynomial length.
    #[inline]
    pub fn poly_length(&self) -> usize {
        self.key.len()
    }

    /// Returns the distribution used to sample this key.
    #[inline]
    pub fn distr(&self) -> SecretKeyDistr {
        self.distr
    }

    /// Returns the canonical signed coefficients of `f`.
    #[inline]
    pub fn as_slice(&self) -> &[T::SignedInteger] {
        &self.key
    }

    /// Samples a coefficient key from `params`.
    ///
    /// This method does not impose backend-specific invertibility. Use
    /// [`crate::NttNtruSecretKey::generate`] or
    /// [`crate::FourierNtruSecretKey::generate`] when an immediately usable
    /// encryption key is required.
    pub fn generate<R, M>(params: &NtruParameters<T, M>, rng: &mut R) -> Self
    where
        R: rand::Rng + rand::CryptoRng,
        M: primus_reduce::RingContext<T>,
    {
        let poly_length = params.poly_length();
        let distr = params.secret_key_distr();
        let key = params.secret_key_sampler().sample(poly_length, rng);
        Self { key, distr }
    }
}
