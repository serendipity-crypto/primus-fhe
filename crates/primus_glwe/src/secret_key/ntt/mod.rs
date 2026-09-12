//! Single-modulus NTT-domain GLWE secret key with encryption and decryption.

use primus_integer::FheUint;
use primus_lattice::GlweSize;
use primus_modulus::UintModulus;
use primus_ntt::NttTable;
use primus_poly::NttPolynomialIter;
use primus_reduce::{EncodeSigned, FieldContext};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{GlweParameters, SecretKeyDistr};

use super::GlweSecretKey;

mod batch;
mod context;
mod decrypt;
mod encrypt;
mod gadget;
mod truncated;
pub use context::NttGadgetEncryptContext;

/// A single-modulus GLWE secret key in NTT form.
/// Key storage is securely erased on drop.
///
/// # Correctness
///
/// Key values must be canonical residues in the NTT representation used by
/// subsequent operations. The key does not store the modulus or table;
/// callers must preserve both when supplying parameters and transform tables.
#[derive(Clone)]
pub struct NttGlweSecretKey<T: FheUint> {
    key: Vec<T>,
    size: GlweSize,
    distr: SecretKeyDistr,
}

impl<T: FheUint> Zeroize for NttGlweSecretKey<T> {
    #[inline]
    fn zeroize(&mut self) {
        self.key.zeroize();
    }
}

impl<T: FheUint> ZeroizeOnDrop for NttGlweSecretKey<T> {}

impl<T: FheUint> Drop for NttGlweSecretKey<T> {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl<T: FheUint> NttGlweSecretKey<T> {
    /// Creates a new [`NttGlweSecretKey<T>`].
    ///
    /// # Correctness
    ///
    /// `key` must satisfy the representation contract of [`Self`] and encode
    /// a secret key with distribution `distr`; neither property is checked.
    ///
    /// # Panics
    ///
    /// Panics if `key.len()` differs from `size.mask_len()`.
    #[inline]
    #[must_use]
    pub fn new(key: Vec<T>, size: GlweSize, distr: SecretKeyDistr) -> Self {
        assert_eq!(key.len(), size.mask_len(), "NTT secret key layout mismatch");
        Self { key, size, distr }
    }

    /// Returns the coefficient-domain GLWE layout.
    #[inline]
    #[must_use]
    pub fn glwe_size(&self) -> GlweSize {
        self.size
    }

    /// Returns the coefficient polynomial length.
    #[inline]
    #[must_use]
    pub fn poly_length(&self) -> usize {
        self.size.poly_length()
    }

    /// Returns the GLWE dimension.
    #[inline]
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.size.dimension()
    }

    /// Returns the secret-key distribution.
    #[inline]
    #[must_use]
    pub fn distr(&self) -> SecretKeyDistr {
        self.distr
    }

    #[inline]
    /// Iterates over the NTT-domain secret polynomials in GLWE component order.
    #[must_use]
    pub fn iter(&self) -> NttPolynomialIter<'_, T> {
        NttPolynomialIter::new(self.key.as_slice(), self.size.poly_length())
    }

    /// Encodes and transforms a signed coefficient-domain secret key.
    ///
    /// # Correctness
    ///
    /// Every signed coefficient in `secret_key` must satisfy `s.unsigned_abs() < q`,
    /// where `q` is `ntt_table.modulus()`; see [`EncodeSigned::encode_signed`].
    ///
    /// # Panics
    ///
    /// Panics if the table polynomial length differs from the key layout.
    #[inline]
    #[must_use]
    pub fn from_coeff_secret_key<Table>(secret_key: &GlweSecretKey<T>, ntt_table: &Table) -> Self
    where
        Table: NttTable<ValueT = T>,
    {
        let size = secret_key.glwe_size();
        let poly_length = size.poly_length();
        assert_eq!(ntt_table.poly_length(), poly_length);

        let mut key = vec![T::ZERO; size.mask_len()];
        let modulus = UintModulus(ntt_table.modulus());
        for (coefficients, secret) in secret_key.iter().zip(key.chunks_exact_mut(poly_length)) {
            modulus.encode_signed_slice_to(coefficients, secret);
            ntt_table.transform_slice(secret);
        }

        Self::new(key, size, secret_key.distr)
    }

    /// Generates a new [`NttGlweSecretKey<T>`] from parameters.
    /// Reuses the parameter sampler to fill the final key storage with canonical
    /// residues, then transforms each polynomial in place. Fixed weights apply
    /// to the complete coefficient key, not to individual polynomials.
    ///
    /// # Panics
    ///
    /// Panics if the table's polynomial length or modulus differs from `params`,
    /// or a fixed secret-key weight exceeds the complete key length or its sum
    /// overflows. Key storage is erased on unwinding if sampling or NTT panics.
    #[must_use]
    #[inline]
    pub fn generate<R, M>(
        params: &GlweParameters<T, M>,
        ntt_table: &impl NttTable<ValueT = T>,
        rng: &mut R,
    ) -> Self
    where
        R: rand::Rng + rand::CryptoRng,
        M: FieldContext<T>,
    {
        assert_eq!(
            ntt_table.poly_length(),
            params.poly_length(),
            "NTT polynomial length mismatch"
        );
        assert_eq!(
            ntt_table.modulus(),
            params.cipher_modulus().value(),
            "NTT ciphertext modulus mismatch"
        );
        let size = params.size();
        // Own storage before sampling so Drop erases it even on unwinding.
        let mut result = Self {
            key: vec![T::ZERO; size.mask_len()],
            size,
            distr: params.secret_key_distr(),
        };
        params.secret_key_sampler().sample_encoded_to(
            &mut result.key,
            params.cipher_modulus_minus_one(),
            rng,
        );
        for polynomial in result.key.chunks_exact_mut(size.poly_length()) {
            ntt_table.transform_slice(polynomial);
        }
        result
    }
}
