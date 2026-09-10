use primus_integer::{AsInto, FheInt, FheUint, SignedInteger};
use rand::distr::Bernoulli;

use crate::{DiscreteGaussian, SecretKeyDistr, SignedDiscreteGaussian, ternary::TernarySampler};

/// Reusable secret-key sampler producing canonical ciphertext-modulus residues.
///
/// Construct with [`Self::new`]; Gaussian tables and the encoding of `-1` are
/// retained across calls, along with binary/ternary probability thresholds.
/// Fixed weights apply to the complete output slice.
pub type EncodedSecretKeySampler<T> = SecretKeySampler<T, DiscreteGaussian<T>>;

/// Reusable secret-key sampler producing signed integer coefficients.
///
/// Construct with [`Self::new`]; Gaussian tables and binary/ternary probability
/// thresholds are retained across calls.
/// This sampler is independent of a ciphertext modulus.
pub type SignedSecretKeySampler<T> = SecretKeySampler<T, SignedDiscreteGaussian<T>>;

/// Shared implementation of the encoded and signed secret-key samplers.
/// Constructors tie the distribution, output representation and Gaussian state
/// together; callers cannot replace individual fields. Use the encoded/signed
/// aliases to select the Gaussian output representation. Backend selection
/// and batch dispatch belong to the Gaussian sampler.
#[derive(Clone)]
pub struct SecretKeySampler<T, G> {
    distr: SecretKeyDistr,
    minus_one: T,
    method: SamplingMethod<G>,
}

/// Prepared state for one sampling algorithm. Dispatch stays outside coefficient
/// loops; probability thresholds and Gaussian tables are reused across calls.
#[derive(Clone)]
enum SamplingMethod<G> {
    UniformBinary,
    Binary(Bernoulli),
    SparseTernary,
    UniformTernary,
    Ternary(TernarySampler),
    FixedHammingWeightBinary {
        hamming_weight: usize,
    },
    FixedHammingWeightTernary {
        hamming_weight: usize,
    },
    FixedCompositionTernary {
        negative_one_weight: usize,
        one_weight: usize,
    },
    Gaussian(G),
}

impl<T: FheUint> EncodedSecretKeySampler<T> {
    /// Prepares a sampler under the modulus identified by `modulus_minus_one`.
    /// `T::MAX` denotes the native modulus `2^T::BITS`.
    ///
    /// # Panics
    ///
    /// Panics if the modulus is less than two, probabilities are
    /// invalid, or the Gaussian cannot be constructed under this modulus.
    /// Fixed weights are checked against the output length when sampling.
    #[must_use]
    pub fn new(distr: SecretKeyDistr, modulus_minus_one: T) -> Self {
        assert!(
            modulus_minus_one != T::ZERO,
            "secret-key modulus must exceed one"
        );
        Self::prepare(distr, modulus_minus_one, |sigma| {
            DiscreteGaussian::new(sigma, modulus_minus_one)
                .expect("invalid encoded secret-key Gaussian distribution")
        })
    }

    /// Samples a complete logical secret key into a newly allocated vector.
    ///
    /// # Panics
    ///
    /// Panics before sampling if a fixed weight exceeds `length` or its sum overflows.
    #[must_use]
    pub fn sample<R: rand::Rng + rand::CryptoRng>(&self, length: usize, rng: &mut R) -> Vec<T> {
        if let SamplingMethod::Gaussian(gaussian) = &self.method {
            return gaussian.sample_vec(length, rng);
        }
        let mut output = vec![T::ZERO; length];
        self.sample_to(&mut output, rng);
        output
    }

    /// Overwrites a complete logical secret key without allocating.
    /// Fixed weights apply to the whole slice, not independently to chunks.
    /// Randomness consumption follows the underlying vector samplers and may
    /// change when their algorithms change. It need not match repeated scalar
    /// sampling or be zero for empty slices. A panicking RNG may leave partially
    /// written output.
    ///
    /// # Panics
    ///
    /// Panics before writing or sampling if a fixed weight exceeds the output
    /// length or its sum overflows.
    #[inline]
    pub fn sample_to<R: rand::Rng + rand::CryptoRng>(&self, output: &mut [T], rng: &mut R) {
        self.sample_to_with(output, rng, DiscreteGaussian::sample_to);
    }
}

impl<T: FheInt + SignedInteger> SignedSecretKeySampler<T> {
    /// Prepares a signed sampler without a ciphertext modulus.
    ///
    /// # Panics
    ///
    /// Panics if probabilities are invalid, or the Gaussian
    /// cannot be constructed for `T`. Fixed weights are checked against the
    /// output length when sampling.
    #[must_use]
    pub fn new(distr: SecretKeyDistr) -> Self {
        Self::prepare(distr, -T::ONE, |sigma| {
            SignedDiscreteGaussian::new(sigma)
                .expect("invalid signed secret-key Gaussian distribution")
        })
    }

    /// Returns an inclusive bound on the unsigned magnitude of every sample.
    /// Gaussian sampling uses its truncated support; binary and ternary
    /// distributions return one, including configurations that only emit zero.
    /// Parameters can compare this bound with a target modulus once, before
    /// generating any keys.
    #[must_use]
    #[inline]
    pub fn maximum_magnitude(&self) -> T::UnsignedInteger {
        match &self.method {
            SamplingMethod::Gaussian(gaussian) => gaussian.maximum_magnitude().as_into(),
            _ => 1u8.as_into(),
        }
    }

    /// Samples a complete logical secret key into a newly allocated vector.
    ///
    /// # Panics
    ///
    /// Panics before sampling if a fixed weight exceeds `length` or its sum overflows.
    #[must_use]
    pub fn sample<R: rand::Rng + rand::CryptoRng>(&self, length: usize, rng: &mut R) -> Vec<T> {
        if let SamplingMethod::Gaussian(gaussian) = &self.method {
            return gaussian.sample_vec(length, rng);
        }
        let mut output = vec![T::ZERO; length];
        self.sample_to(&mut output, rng);
        output
    }

    /// Overwrites a complete logical secret key without allocating.
    /// Fixed weights apply to the whole slice, not independently to chunks.
    /// Randomness consumption follows the underlying vector samplers and may
    /// change when their algorithms change. It need not match repeated scalar
    /// sampling or be zero for empty slices. A panicking RNG may leave partially
    /// written output.
    ///
    /// # Panics
    ///
    /// Panics before writing or sampling if a fixed weight exceeds the output
    /// length or its sum overflows.
    #[inline]
    pub fn sample_to<R: rand::Rng + rand::CryptoRng>(&self, output: &mut [T], rng: &mut R) {
        self.sample_to_with(output, rng, SignedDiscreteGaussian::sample_to);
    }
}

impl<T: FheInt, G> SecretKeySampler<T, G> {
    fn prepare(distr: SecretKeyDistr, minus_one: T, gaussian: impl FnOnce(f64) -> G) -> Self {
        let method = match distr {
            SecretKeyDistr::UniformBinary => SamplingMethod::UniformBinary,
            SecretKeyDistr::Binary { one_probability } => SamplingMethod::Binary(
                Bernoulli::new(one_probability)
                    .expect("binary one probability must be finite and in [0, 1]"),
            ),
            SecretKeyDistr::SparseTernary => SamplingMethod::SparseTernary,
            SecretKeyDistr::UniformTernary => SamplingMethod::UniformTernary,
            SecretKeyDistr::Ternary {
                negative_one_probability,
                one_probability,
            } => SamplingMethod::Ternary(TernarySampler::new(
                negative_one_probability,
                one_probability,
            )),
            SecretKeyDistr::FixedHammingWeightBinary { hamming_weight } => {
                SamplingMethod::FixedHammingWeightBinary { hamming_weight }
            }
            SecretKeyDistr::FixedHammingWeightTernary { hamming_weight } => {
                SamplingMethod::FixedHammingWeightTernary { hamming_weight }
            }
            SecretKeyDistr::FixedCompositionTernary {
                negative_one_weight,
                one_weight,
            } => SamplingMethod::FixedCompositionTernary {
                negative_one_weight,
                one_weight,
            },
            SecretKeyDistr::Gaussian { standard_deviation } => {
                SamplingMethod::Gaussian(gaussian(standard_deviation))
            }
        };
        Self {
            distr,
            minus_one,
            method,
        }
    }

    /// Returns the configured coefficient distribution.
    #[must_use]
    #[inline]
    pub fn distr(&self) -> SecretKeyDistr {
        self.distr
    }

    /// Shared distribution dispatch. The representation-specific wrapper supplies
    /// the Gaussian batch method, so this layer does not name Gaussian backends.
    // Keep this large batch switch out of callers: inlining it regressed binary
    // and dense fixed-weight cases in the sample_secret_key benchmark.
    #[inline(never)]
    fn sample_to_with<R: rand::Rng + rand::CryptoRng>(
        &self,
        output: &mut [T],
        rng: &mut R,
        sample_gaussian: impl FnOnce(&G, &mut [T], &mut R),
    ) {
        match &self.method {
            SamplingMethod::UniformBinary => crate::sample_uniform_binary_values_to(output, rng),
            SamplingMethod::Binary(distribution) => {
                crate::binary::fill_binary(output, distribution, rng)
            }
            SamplingMethod::SparseTernary => {
                crate::sample_sparse_ternary_values_to(output, self.minus_one, rng)
            }
            SamplingMethod::UniformTernary => {
                crate::sample_uniform_ternary_values_to(output, self.minus_one, rng)
            }
            SamplingMethod::Ternary(distribution) => {
                distribution.sample_to(output, self.minus_one, rng)
            }
            SamplingMethod::FixedHammingWeightBinary { hamming_weight } => {
                crate::sample_fixed_hamming_weight_binary_values_to(output, *hamming_weight, rng)
            }
            SamplingMethod::FixedHammingWeightTernary { hamming_weight } => {
                crate::sample_fixed_hamming_weight_ternary_values_to(
                    output,
                    self.minus_one,
                    *hamming_weight,
                    rng,
                )
            }
            SamplingMethod::FixedCompositionTernary {
                negative_one_weight,
                one_weight,
            } => crate::sample_fixed_composition_ternary_values_to(
                output,
                self.minus_one,
                *negative_one_weight,
                *one_weight,
                rng,
            ),
            SamplingMethod::Gaussian(gaussian) => sample_gaussian(gaussian, output, rng),
        }
    }
}
