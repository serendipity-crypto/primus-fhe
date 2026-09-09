use primus_data::{Data, DataMut};
use primus_decompose::primitive::ApproxSignedBasis;
use primus_integer::{FheUint, SignedInteger};
use primus_lattice::lwe::Lwe;
use primus_reduce::RingContext;

use crate::secret_key::encode_signed;

mod batch;

use crate::{LweParameters, LweSecretKey, LweSecretKeyRef};

/// An LWE key-switching key from one secret-key vector to another.
///
/// Storage is ordered by input secret coefficient, then decomposition level;
/// every entry is an LWE encryption under the output secret key.
#[derive(Clone)]
pub struct LweKeySwitchingKey<T: FheUint> {
    data: Vec<T>,
    input_dimension: usize,
    output_dimension: usize,
    basis: ApproxSignedBasis<T>,
}

impl<T: FheUint> LweKeySwitchingKey<T> {
    /// Generates a key switching from `input_secret_key` to
    /// `output_secret_key`. The output key uses encoded storage so entry
    /// encryption retains the modulus backend's slice dot-product kernel.
    /// Dimensions are taken from the keys; the generated key takes ownership of `basis`.
    ///
    /// # Correctness
    ///
    /// Encoded coefficients of either key must be canonical residues
    /// under the ciphertext modulus in `output_parameters`, satisfying
    /// [`primus_reduce::ReduceDotProduct`] and [`primus_reduce::ReduceMul`].
    /// Signed input coefficients must satisfy `-q < coefficient < q` for an
    /// explicit modulus `q`; every signed value is valid for the native modulus.
    /// These coefficient ranges are not checked in release builds. The input
    /// ciphertexts must use the same modulus; this operation does not switch moduli.
    ///
    /// # Panics
    ///
    /// Panics if the input key is empty, the output key dimension differs from
    /// the nonzero output parameter dimension, the basis modulus differs from
    /// the output modulus, or the key storage length overflows.
    pub fn generate<R, M>(
        input_secret_key: LweSecretKeyRef<'_, T>,
        output_secret_key: &LweSecretKey<T>,
        output_parameters: &LweParameters<T, M>,
        basis: ApproxSignedBasis<T>,
        rng: &mut R,
    ) -> Self
    where
        R: rand::Rng + rand::CryptoRng,
        M: RingContext<T>,
    {
        let input_dimension = input_secret_key.dimension();
        let output_dimension = output_secret_key.dimension();
        assert!(input_dimension != 0, "input LWE dimension must be non-zero");
        assert_eq!(
            output_dimension,
            output_parameters.dimension(),
            "output LWE key dimension mismatch"
        );
        assert_eq!(
            basis.modulus(),
            output_parameters.cipher_modulus().explicit_value(),
            "LWE key switching currently requires matching input and output ciphertext moduli"
        );

        let output_lwe_len = output_dimension + 1;
        let coefficient_len = output_lwe_len
            .checked_mul(basis.decompose_length())
            .expect("LWE key-switching coefficient block length overflow");
        let length = input_dimension
            .checked_mul(coefficient_len)
            .expect("LWE key-switching key length overflow");
        let mut data = vec![T::ZERO; length];

        let modulus = output_parameters.cipher_modulus();
        let uniform = output_parameters.cipher_modulus_uniform_distr();
        let gaussian = output_parameters.noise_distribution();
        let output_key = output_secret_key.as_view();
        // Each coefficient owns a contiguous block of decomposition-level entries.
        let blocks = data.chunks_exact_mut(coefficient_len);
        let encrypt_levels = |(secret, block): (T, &mut [T])| {
            for (scalar, entry) in basis
                .scalar_iter()
                .zip(block.chunks_exact_mut(output_lwe_len))
            {
                let plaintext = modulus.reduce_mul(secret, scalar);
                output_key.encrypt_encoded_to(
                    plaintext,
                    &mut Lwe::new(entry),
                    modulus,
                    uniform,
                    gaussian,
                    rng,
                );
            }
        };
        // Select representation and modulus shape once, outside both loops.
        match input_secret_key {
            LweSecretKeyRef::Encoded(coefficients) => coefficients
                .iter()
                .copied()
                .zip(blocks)
                .for_each(encrypt_levels),
            LweSecretKeyRef::Signed(coefficients) => match modulus.explicit_value() {
                Some(q) => coefficients
                    .iter()
                    .copied()
                    .map(|coefficient| encode_signed(coefficient, q))
                    .zip(blocks)
                    .for_each(encrypt_levels),
                None => coefficients
                    .iter()
                    .copied()
                    .map(SignedInteger::cast_to_unsigned)
                    .zip(blocks)
                    .for_each(encrypt_levels),
            },
        }

        Self {
            data,
            input_dimension,
            output_dimension,
            basis,
        }
    }

    /// Returns the input LWE dimension.
    #[inline]
    pub fn input_dimension(&self) -> usize {
        self.input_dimension
    }

    /// Returns the output LWE dimension.
    #[inline]
    pub fn output_dimension(&self) -> usize {
        self.output_dimension
    }

    /// Returns the key-switch decomposition basis.
    #[inline]
    pub fn basis(&self) -> &ApproxSignedBasis<T> {
        &self.basis
    }

    /// Returns the raw key data.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Key-switches `input`, overwriting all of `output` without allocating.
    ///
    /// # Correctness
    ///
    /// `input` must have canonical coefficients under the modulus used to
    /// generate this key. Decomposition approximates its mask coefficients;
    /// the resulting phase error includes both decomposition error weighted
    /// by the input secret and the accumulated key-entry noise. Parameters
    /// must leave sufficient decoding margin for the intended operation.
    ///
    /// # Panics
    ///
    /// Panics if either ciphertext lacks a body, its dimension differs from
    /// this key, or `modulus` differs from the basis modulus. Validating these
    /// conditions does not modify `output`.
    pub fn key_switch_to<M, A, B>(&self, input: &Lwe<A>, output: &mut Lwe<B>, modulus: M)
    where
        M: RingContext<T>,
        A: Data<Elem = T>,
        B: DataMut<Elem = T>,
    {
        let basis = &self.basis;
        let (input_mask, input_body) = input.a_b();
        let (output_mask, output_body) = output.a_b_mut();
        assert_eq!(
            input_mask.len(),
            self.input_dimension,
            "input LWE ciphertext dimension mismatch"
        );
        assert_eq!(
            output_mask.len(),
            self.output_dimension,
            "output LWE ciphertext dimension mismatch"
        );
        assert_eq!(
            basis.modulus(),
            modulus.explicit_value(),
            "LWE key-switching modulus mismatch"
        );

        output_mask.fill(T::ZERO);
        if modulus.explicit_value().is_none() {
            // Native multiplication is cheap: accumulate (0, b) - sum(digit * entry).
            *output_body = input_body;
            self.key_switch_kernel::<true, _>(input_mask, output.as_mut(), modulus);
        } else {
            // Keep positive digit multiply-adds for explicit moduli; negating
            // every general digit costs more than one final output negation.
            *output_body = modulus.reduce_neg(input_body);
            self.key_switch_kernel::<false, _>(input_mask, output.as_mut(), modulus);
            output.neg_assign(modulus);
        }
    }

    /// Accumulates mask contributions into an initialized output. Dimensions
    /// and modulus have been checked by the public boundary.
    fn key_switch_kernel<const NATIVE: bool, M: RingContext<T>>(
        &self,
        input_mask: &[T],
        output: &mut [T],
        modulus: M,
    ) {
        let output_lwe_len = self.output_dimension + 1;
        let coefficient_len = output_lwe_len * self.basis.decompose_length();
        for (&coefficient, block) in input_mask
            .iter()
            .zip(self.data.chunks_exact(coefficient_len))
        {
            let (adjusted, mut carry) = self.basis.init_value_carry(coefficient);
            for (decomposer, entry) in self
                .basis
                .decomposer_iter()
                .zip(block.chunks_exact(output_lwe_len))
            {
                let (digit, next_carry) = decomposer.decompose(adjusted, carry);
                carry = next_carry;
                accumulate_entry::<NATIVE, _, _>(output, entry, digit, modulus);
            }
        }
    }

    /// Key-switches `input` into a newly allocated ciphertext.
    ///
    /// The correctness and panic conditions of [`Self::key_switch_to`] apply;
    /// the output dimension is determined by this key.
    pub fn key_switch<M, A>(&self, input: &Lwe<A>, modulus: M) -> Lwe<Vec<T>>
    where
        M: RingContext<T>,
        A: Data<Elem = T>,
    {
        let mut output = Lwe::zero(self.output_dimension());
        self.key_switch_to(input, &mut output, modulus);
        output
    }
}

/// Accumulates one key entry into a validated output. Native kernels subtract
/// digit * entry directly. Explicit-modulus kernels accumulate its positive
/// contribution and negate the complete output afterwards; this retains cheap
/// additions/subtractions for small digits without negating general digits.
#[inline]
fn accumulate_entry<const NATIVE: bool, T: FheUint, M: RingContext<T>>(
    output: &mut [T],
    entry: &[T],
    digit: T,
    modulus: M,
) {
    if digit.is_zero() {
        return;
    }
    if NATIVE {
        modulus.reduce_add_mul_scalar_slice_assign(output, entry, modulus.reduce_neg(digit));
    } else if digit == T::ONE {
        modulus.reduce_add_slice_assign(output, entry);
    } else if digit == modulus.reduce_neg(T::ONE) {
        modulus.reduce_sub_slice_assign(output, entry);
    } else if digit == T::TWO {
        modulus.reduce_add_slice_assign(output, entry);
        modulus.reduce_add_slice_assign(output, entry);
    } else if digit == modulus.reduce_neg(T::TWO) {
        modulus.reduce_sub_slice_assign(output, entry);
        modulus.reduce_sub_slice_assign(output, entry);
    } else {
        modulus.reduce_add_mul_scalar_slice_assign(output, entry, digit);
    }
}
