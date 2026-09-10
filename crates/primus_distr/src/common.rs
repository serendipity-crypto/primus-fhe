use std::slice::IterMut;

use num_traits::ConstZero;
use primus_integer::{FheInt, FheUint, Integer, SignedInteger, UnsignedInteger};
use rand::{
    distr::{Bernoulli, Distribution, Uniform},
    seq::SliceRandom,
};

use crate::{DiscreteGaussian, SignedDiscreteGaussian};

/// Sample a binary vector whose values are `T`.
pub fn sample_uniform_binary_values<T, R>(length: usize, rng: &mut R) -> Vec<T>
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let mut v = vec![T::ZERO; length];
    sample_uniform_binary_values_to(&mut v, rng);
    v
}

/// Sample a binary vector whose values are `T`.
pub fn sample_uniform_binary_values_to<T, R>(result: &mut [T], rng: &mut R)
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let s = [T::ZERO, T::ONE];

    let (chunks, remainder) = result.as_chunks_mut::<32>();
    for chunk in chunks {
        let mut r = rng.next_u32();
        for elem in chunk.iter_mut() {
            *elem = s[(r & 0b1) as usize];
            r >>= 1;
        }
    }
    let mut r = rng.next_u32();
    for elem in remainder {
        *elem = s[(r & 0b1) as usize];
        r >>= 1;
    }
}

/// Samples binary values with the configured probability of sampling `1`.
///
/// # Panics
///
/// Panics if `one_probability` is not finite or does not belong to `[0, 1]`.
pub fn sample_binary_values_with_probability<T, R>(
    length: usize,
    one_probability: f64,
    rng: &mut R,
) -> Vec<T>
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let mut result = vec![T::ZERO; length];
    sample_binary_values_with_probability_to(&mut result, one_probability, rng);
    result
}

/// Fills a binary slice with the configured probability of sampling `1`.
///
/// # Panics
///
/// Panics before writing if the probability is invalid.
pub fn sample_binary_values_with_probability_to<T, R>(
    result: &mut [T],
    one_probability: f64,
    rng: &mut R,
) where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let distribution = Bernoulli::new(one_probability)
        .expect("binary one probability must be finite and in [0, 1]");
    crate::binary::fill_binary(result, &distribution, rng);
}

/// Samples a binary vector containing exactly `hamming_weight` ones.
///
/// # Panics
///
/// Panics if `hamming_weight` exceeds `length`.
pub fn sample_fixed_hamming_weight_binary_values<T, R>(
    length: usize,
    hamming_weight: usize,
    rng: &mut R,
) -> Vec<T>
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let mut result = vec![T::ZERO; length];
    sample_fixed_hamming_weight_binary_values_to(&mut result, hamming_weight, rng);
    result
}

/// Overwrites a complete logical binary key with exactly `hamming_weight` ones.
///
/// # Panics
///
/// Panics before writing if the weight exceeds the slice length.
pub fn sample_fixed_hamming_weight_binary_values_to<T, R>(
    result: &mut [T],
    hamming_weight: usize,
    rng: &mut R,
) where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    assert!(
        hamming_weight <= result.len(),
        "binary Hamming weight must not exceed the output length"
    );
    let zero_weight = result.len() - hamming_weight;
    let (base, inserted, count) = if hamming_weight <= zero_weight {
        (T::ZERO, T::ONE, hamming_weight)
    } else {
        (T::ONE, T::ZERO, zero_weight)
    };
    result.fill(base);
    insert_random_values(result, result.len() - count, count, inserted, rng);
}

/// Samples sparse ternary values.
///
/// Values have probabilities `P(0) = 1/2` and `P(1) = P(-1) = 1/4`.
/// `minus_one` encodes `-1`. Uses [`sample_sparse_ternary_values_to`]'s
/// packed random-word order.
#[must_use]
pub fn sample_sparse_ternary_values<T, R>(minus_one: T, length: usize, rng: &mut R) -> Vec<T>
where
    T: Integer,
    R: rand::Rng + rand::CryptoRng,
{
    let mut v = vec![T::ZERO; length];
    sample_sparse_ternary_values_to(&mut v, minus_one, rng);
    v
}

/// Fills `result` with sparse ternary values.
///
/// Values have probabilities `P(0) = 1/2` and `P(1) = P(-1) = 1/4`.
/// `minus_one` encodes `-1`.
/// Each `next_u32()` supplies sixteen low-to-high two-bit groups, mapped to
/// `[0, 0, 1, minus_one]`. Consumes exactly `result.len().div_ceil(16)` words;
/// unused high bits are discarded, and empty output consumes no randomness.
pub fn sample_sparse_ternary_values_to<T, R>(result: &mut [T], minus_one: T, rng: &mut R)
where
    T: Integer,
    R: rand::Rng + rand::CryptoRng,
{
    let s = [T::ZERO, T::ZERO, T::ONE, minus_one];

    let (chunks, remainder) = result.as_chunks_mut::<16>();
    for chunk in chunks {
        let mut r = rng.next_u32();
        for elem in chunk.iter_mut() {
            *elem = s[(r & 0b11) as usize];
            r >>= 2;
        }
    }
    if !remainder.is_empty() {
        let mut r = rng.next_u32();
        for elem in remainder {
            *elem = s[(r & 0b11) as usize];
            r >>= 2;
        }
    }
}

/// Samples uniformly from `{-1, 0, 1}`.
pub fn sample_uniform_ternary_values<T, R>(minus_one: T, length: usize, rng: &mut R) -> Vec<T>
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let mut result = vec![T::ZERO; length];
    sample_uniform_ternary_values_to(&mut result, minus_one, rng);
    result
}

/// Fills `result` uniformly from `{-1, 0, 1}`.
pub fn sample_uniform_ternary_values_to<T, R>(result: &mut [T], minus_one: T, rng: &mut R)
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let values = [minus_one, T::ZERO, T::ONE];
    let mut random = 0;
    let mut remaining_bytes = 0;
    // A uniform byte below 3^5 contains five independent uniform trits.
    // Reject 243..=255, rather than reducing a byte modulo 243 with bias.
    for chunk in result.chunks_mut(5) {
        let mut trits = loop {
            if remaining_bytes == 0 {
                random = rng.next_u32();
                remaining_bytes = 4;
            }
            let byte = random & 0xff;
            random >>= 8;
            remaining_bytes -= 1;
            if byte < 243 {
                break byte;
            }
        };
        for output in chunk {
            *output = values[(trits % 3) as usize];
            trits /= 3;
        }
    }
}

/// Samples ternary values with explicit probabilities for `-1` and `1`.
/// Each probability is rounded down to a multiple of `2^-64`; the total
/// nonzero probability is capped at one to handle floating-point boundary rounding.
///
/// # Panics
///
/// Panics if either probability is invalid or their sum exceeds one.
pub fn sample_ternary_values_with_probabilities<T, R>(
    minus_one: T,
    length: usize,
    minus_one_probability: f64,
    one_probability: f64,
    rng: &mut R,
) -> Vec<T>
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let mut result = vec![T::ZERO; length];
    sample_ternary_values_with_probabilities_to(
        &mut result,
        minus_one,
        minus_one_probability,
        one_probability,
        rng,
    );
    result
}

/// Fills a ternary slice with explicit probabilities of `minus_one` and `1`.
///
/// `minus_one` is the caller's encoding of `-1`.
/// Probabilities use the quantization described by
/// [`sample_ternary_values_with_probabilities`].
///
/// # Panics
///
/// Panics before writing if a probability is invalid or their sum exceeds one.
pub fn sample_ternary_values_with_probabilities_to<T, R>(
    result: &mut [T],
    minus_one: T,
    minus_one_probability: f64,
    one_probability: f64,
    rng: &mut R,
) where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    crate::ternary::TernarySampler::new(minus_one_probability, one_probability)
        .sample_to(result, minus_one, rng);
}

/// Samples exactly `hamming_weight` nonzero coefficients at uniformly selected
/// positions, with independent uniform signs. `minus_one` encodes `-1`.
///
/// # Panics
///
/// Panics if the weight exceeds `length`.
#[must_use]
pub fn sample_fixed_hamming_weight_ternary_values<T, R>(
    minus_one: T,
    length: usize,
    hamming_weight: usize,
    rng: &mut R,
) -> Vec<T>
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let mut output = vec![T::ZERO; length];
    sample_fixed_hamming_weight_ternary_values_to(&mut output, minus_one, hamming_weight, rng);
    output
}

/// Overwrites a complete logical key with exactly `hamming_weight` nonzero
/// coefficients and independent uniform signs. `minus_one` encodes `-1`.
///
/// # Panics
///
/// Panics before writing or sampling if the weight exceeds the output length.
pub fn sample_fixed_hamming_weight_ternary_values_to<T, R>(
    output: &mut [T],
    minus_one: T,
    hamming_weight: usize,
    rng: &mut R,
) where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    assert!(
        hamming_weight <= output.len(),
        "ternary Hamming weight must not exceed the output length"
    );
    let values = [minus_one, T::ONE];
    if hamming_weight == output.len() {
        for chunk in output.chunks_mut(32) {
            let mut signs = rng.next_u32();
            for out in chunk {
                *out = values[(signs & 1) as usize];
                signs >>= 1;
            }
        }
        return;
    }
    output.fill(T::ZERO);
    let mut signs = 0;
    let mut remaining_signs = 0;
    // Random insertion extends the initially all-zero prefix, as in
    // insert_random_values; the sign of each inserted value is independent.
    for index in output.len() - hamming_weight..output.len() {
        if remaining_signs == 0 {
            signs = rng.next_u32();
            remaining_signs = 32;
        }
        let selected = Uniform::new_inclusive(0, index)
            .expect("nonempty insertion range")
            .sample(rng);
        output[index] = output[selected];
        output[selected] = values[(signs & 1) as usize];
        signs >>= 1;
        remaining_signs -= 1;
    }
}

/// Samples a ternary vector with exact counts of `-1` and `1`.
///
/// # Panics
///
/// Panics if the weights overflow or their sum exceeds `length`.
#[must_use]
pub fn sample_fixed_composition_ternary_values<T, R>(
    minus_one: T,
    length: usize,
    minus_one_weight: usize,
    one_weight: usize,
    rng: &mut R,
) -> Vec<T>
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let mut result = vec![T::ZERO; length];
    sample_fixed_composition_ternary_values_to(
        &mut result,
        minus_one,
        minus_one_weight,
        one_weight,
        rng,
    );
    result
}

/// Overwrites a complete logical ternary key with exact negative/positive weights.
///
/// `minus_one` is the caller's encoding of `-1`.
///
/// # Panics
///
/// Panics before writing if the weights overflow or exceed the slice length.
pub fn sample_fixed_composition_ternary_values_to<T, R>(
    result: &mut [T],
    minus_one: T,
    minus_one_weight: usize,
    one_weight: usize,
    rng: &mut R,
) where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let nonzero_weight = minus_one_weight
        .checked_add(one_weight)
        .expect("ternary Hamming weights must fit in usize");
    assert!(
        nonzero_weight <= result.len(),
        "ternary Hamming weights must not exceed the output length"
    );
    let counts = [result.len() - nonzero_weight, minus_one_weight, one_weight];
    let values = [T::ZERO, minus_one, T::ONE];
    let mut base = 0;
    for index in 1..3 {
        if counts[index] > counts[base] {
            base = index;
        }
    }
    result.fill(values[base]);
    let mut prefix_length = counts[base];
    for index in 0..3 {
        if index != base {
            insert_random_values(result, prefix_length, counts[index], values[index], rng);
            prefix_length += counts[index];
        }
    }
}

/// Extends a uniformly shuffled prefix by `count` copies of `value`.
/// Moving a uniformly selected old element to the new last position and putting
/// the new value in its place preserves the uniform multiset permutation.
/// The prefix must already have the required distribution and the extension
/// must fit in `output`.
fn insert_random_values<T: Copy, R: rand::Rng + rand::CryptoRng>(
    output: &mut [T],
    prefix_length: usize,
    count: usize,
    value: T,
    rng: &mut R,
) {
    let end = prefix_length + count;
    // For dense insertions, rand batches the changing-range draws. Shuffle
    // only the new suffix into the existing uniform prefix; shuffling that
    // prefix again would do redundant work.
    if count > end / 4 {
        output[prefix_length..end].fill(value);
        let _ = output[..end].partial_shuffle(rng, count);
        return;
    }
    for index in prefix_length..end {
        let selected = Uniform::new_inclusive(0, index)
            .expect("nonempty insertion range")
            .sample(rng);
        output[index] = output[selected];
        output[selected] = value;
    }
}

/// Sample a vector of `length` values from a uniform distribution.
pub fn sample_uniform_values<T, R>(length: usize, distr: &Uniform<T>, rng: &mut R) -> Vec<T>
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    distr.sample_iter(rng).take(length).collect()
}

/// Fill `result` with samples from a uniform distribution (in-place).
pub fn sample_uniform_values_to<T, R>(result: &mut [T], distr: &Uniform<T>, rng: &mut R)
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    result
        .iter_mut()
        .zip(distr.sample_iter(rng))
        .for_each(|(a, b)| *a = b);
}

/// Sample a vector of `length` values from a discrete Gaussian distribution.
pub fn sample_gaussian_values<T, R>(
    length: usize,
    distr: &DiscreteGaussian<T>,
    rng: &mut R,
) -> Vec<T>
where
    T: FheUint,
    R: rand::Rng + rand::CryptoRng,
{
    distr.sample_vec(length, rng)
}

/// Fill `result` with samples from a discrete Gaussian distribution (in-place).
pub fn sample_gaussian_values_to<T, R>(result: &mut [T], distr: &DiscreteGaussian<T>, rng: &mut R)
where
    T: FheUint,
    R: rand::Rng + rand::CryptoRng,
{
    distr.sample_to(result, rng);
}

/// Samples CRT-layout uniform binary values.
///
/// Generates `length` binary values and replicates them across `moduli_count`
/// slots, producing a vector of `length * moduli_count` elements in
/// modulus-major CRT order.
pub fn sample_crt_uniform_binary_values<T, R>(
    length: usize,
    moduli_count: usize,
    rng: &mut R,
) -> Vec<T>
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let mut result = vec![T::ZERO; length * moduli_count];

    sample_crt_uniform_binary_values_to(&mut result, length, rng);

    result
}

/// Fills `result` with CRT-layout uniform binary values.
///
/// Samples `length` binary values into the first chunk, then copies them
/// into each subsequent chunk of `length` elements.
pub fn sample_crt_uniform_binary_values_to<T, R>(result: &mut [T], length: usize, rng: &mut R)
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let (v, w) = result.split_at_mut(length);

    sample_uniform_binary_values_to(v, rng);

    w.chunks_exact_mut(length)
        .for_each(|s| s.copy_from_slice(v));
}

/// Samples CRT-layout sparse ternary values.
///
/// Each logical value is shared by every modulus component and has
/// probabilities `P(0) = 1/2` and `P(1) = P(-1) = 1/4`.
/// `moduli_minus_one[i]` encodes `-1` in component `i`.
pub fn sample_crt_sparse_ternary_values<T, R>(
    length: usize,
    moduli_minus_one: &[T],
    rng: &mut R,
) -> Vec<T>
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let moduli_count = moduli_minus_one.len();
    let mut result = vec![T::ZERO; length * moduli_count];

    sample_crt_sparse_ternary_values_to(&mut result, length, moduli_minus_one, rng);

    result
}

/// Fills `result` with CRT-layout sparse ternary values.
///
/// Each logical value is shared by every modulus component and has
/// probabilities `P(0) = 1/2` and `P(1) = P(-1) = 1/4`.
/// `moduli_minus_one[i]` encodes `-1` in component `i`.
///
/// # Correctness
///
/// `result.len()` must equal `length * moduli_minus_one.len()`; this layout
/// requirement is checked only in debug builds. Empty output is supported.
pub fn sample_crt_sparse_ternary_values_to<T, R>(
    result: &mut [T],
    length: usize,
    moduli_minus_one: &[T],
    rng: &mut R,
) where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    debug_assert_eq!(result.len(), moduli_minus_one.len() * length);

    if result.is_empty() {
        return;
    }

    // Keep random words in a small stack tile, then write each CRT limb
    // contiguously. Every limb decodes the same words with its encoding of -1.
    const TILE_LENGTH: usize = 256;
    let mut words = [0u32; TILE_LENGTH / 16];
    for offset in (0..length).step_by(TILE_LENGTH) {
        let tile_length = (length - offset).min(TILE_LENGTH);
        let words = &mut words[..tile_length.div_ceil(16)];
        for word in words.iter_mut() {
            *word = rng.next_u32();
        }
        for (limb, &minus_one) in result.chunks_exact_mut(length).zip(moduli_minus_one) {
            let values = [T::ZERO, T::ZERO, T::ONE, minus_one];
            for (chunk, &word) in limb[offset..offset + tile_length]
                .chunks_mut(16)
                .zip(words.iter())
            {
                let mut random = word;
                for out in chunk {
                    *out = values[(random & 3) as usize];
                    random >>= 2;
                }
            }
        }
    }
}

/// Samples a uniform vector in modulus-major CRT layout.
pub fn sample_crt_uniform_values<T, R>(
    poly_length: usize,
    uniform_distrs: &[Uniform<T>],
    rng: &mut R,
) -> Vec<T>
where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    let mut result = vec![T::ZERO; poly_length * uniform_distrs.len()];

    sample_crt_uniform_values_to(&mut result, poly_length, uniform_distrs, rng);

    result
}

/// Fills a slice with uniform samples in modulus-major CRT layout.
///
/// In debug builds, this function checks that `poly_length` is nonzero and
/// that `result.len()` equals `poly_length * uniform_distrs.len()`.
pub fn sample_crt_uniform_values_to<T, R>(
    result: &mut [T],
    poly_length: usize,
    uniform_distrs: &[Uniform<T>],
    rng: &mut R,
) where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    debug_assert!(poly_length > 0, "CRT polynomial length must be nonzero");
    debug_assert_eq!(
        result.len(),
        poly_length * uniform_distrs.len(),
        "CRT uniform output length must equal polynomial length times the distribution count"
    );

    result
        .chunks_exact_mut(poly_length)
        .zip(uniform_distrs)
        .for_each(|(s, u)| {
            s.iter_mut()
                .zip(u.sample_iter(&mut *rng))
                .for_each(|(a, b)| {
                    *a = b;
                });
        });
}

/// Sample a uniform vector whose values are `T`.
pub fn sample_crt_uniform_values_iter_mut<T, R>(
    iters: Vec<IterMut<'_, T>>,
    uniform_distrs: &[Uniform<T>],
    rng: &mut R,
) where
    T: FheInt,
    R: rand::Rng + rand::CryptoRng,
{
    iters.into_iter().zip(uniform_distrs).for_each(|(s, u)| {
        s.zip(u.sample_iter(&mut *rng)).for_each(|(a, b)| {
            *a = b;
        });
    });
}

/// Samples a Gaussian vector in modulus-major CRT layout.
///
/// Every modulus must canonically encode the complete truncated support. This
/// precondition is not checked and must be established by the caller's
/// parameter construction.
pub fn sample_crt_gaussian_values<T, R>(
    length: usize,
    moduli: &[T],
    gaussian: &SignedDiscreteGaussian<<T as UnsignedInteger>::SignedInteger>,
    rng: &mut R,
) -> Vec<T>
where
    T: FheUint,
    R: rand::Rng + rand::CryptoRng,
{
    let moduli_count = moduli.len();
    let mut result = vec![T::ZERO; length * moduli_count];
    sample_crt_gaussian_values_to(&mut result, length, moduli, gaussian, rng);

    result
}

/// Fills a slice with Gaussian samples in modulus-major CRT layout.
///
/// Every modulus must canonically encode the complete truncated support. This
/// precondition is not checked and must be established by the caller's
/// parameter construction.
///
/// In debug builds, this function checks that `length` is nonzero and that
/// `result.len()` equals `length * moduli.len()`.
pub fn sample_crt_gaussian_values_to<T, R>(
    result: &mut [T],
    length: usize,
    moduli: &[T],
    gaussian: &SignedDiscreteGaussian<<T as UnsignedInteger>::SignedInteger>,
    rng: &mut R,
) where
    T: FheUint,
    R: rand::Rng + rand::CryptoRng,
{
    debug_assert!(length > 0, "CRT polynomial length must be nonzero");
    debug_assert_eq!(
        result.len(),
        length * moduli.len(),
        "CRT Gaussian output length must equal length times the modulus count"
    );
    if result.is_empty() {
        return;
    }

    for coefficient in 0..length {
        let sample = gaussian.sample(rng);
        if sample >= <<T as UnsignedInteger>::SignedInteger as ConstZero>::ZERO {
            let residue: T = sample.cast_to_unsigned();
            for value in result.iter_mut().skip(coefficient).step_by(length) {
                *value = residue;
            }
        } else {
            for (value, &modulus) in result
                .iter_mut()
                .skip(coefficient)
                .step_by(length)
                .zip(moduli)
            {
                *value = <T as UnsignedInteger>::wrapping_add_signed(modulus, sample);
            }
        }
    }
}
