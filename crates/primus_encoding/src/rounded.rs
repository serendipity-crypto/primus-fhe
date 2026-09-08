use super::decode::DecodeStrategy;
use super::helpers;
use super::helpers::{checked_message, lift_centered_from_raw, mul_div_round};
use super::integer_scale::IntegerScale;
use crate::PlaintextEmbedding;
use primus_integer::FheUint;
use primus_modulus::common::uint::reduce_add_assign;

/// Per-message rounding: `round(lift(m) * q / t) mod q`, returned in `[0,q)`,
/// with ties away from zero before modular reduction.
/// Decoding rounds `c * t / q` to the nearest integer, ties upward, modulo `t`.
/// Inputs to decoding and accumulators must be canonical residues in `[0,q)`.
///
/// This keeps modulus-shape checks and shift/mask computation out of hot
/// coefficient loops while hiding strategy-specific precomputation.
#[derive(Clone, Copy, Debug)]
pub struct RoundedCodec<T: FheUint> {
    t: T,
    centered_half: T,
    strategy: RoundedEncoding<T>,
    decoder: DecodeStrategy<T>,
}

impl<T: FheUint> RoundedCodec<T> {
    /// Creates a codec for plaintext modulus `t` and ciphertext modulus `q`.
    ///
    /// `q = None` selects the native wrapping modulus `2^T::BITS`.
    ///
    /// # Panics
    ///
    /// Panics if `t <= 1` or an explicit `q` is not greater than `t`.
    #[must_use]
    #[inline]
    pub fn new(t: T, q: Option<T>) -> Self {
        let (floor, remainder) = helpers::modulus_div_rem(t, q);
        let decoder = DecodeStrategy::from_ratio(t, q, floor, remainder);
        let strategy = if remainder == T::ZERO {
            RoundedEncoding::Exact(IntegerScale::new(floor, q))
        } else {
            let biased_remainder_fits = (t - T::ONE)
                .checked_mul(remainder)
                .and_then(|product| product.checked_add(t >> 1u32))
                .is_some();
            let ratio = match (biased_remainder_fits, q) {
                (true, None) => RatioEncoding::DecomposedNative(DecomposedRatio {
                    floor,
                    remainder,
                    modulus: NativeRatio,
                }),
                (true, Some(q)) => RatioEncoding::DecomposedExplicit(DecomposedRatio {
                    floor,
                    remainder,
                    modulus: ExplicitRatio { q },
                }),
                (false, None) => RatioEncoding::Native(NativeRatio),
                (false, Some(q)) => RatioEncoding::Wide(ExplicitRatio { q }),
            };
            RoundedEncoding::Ratio(ratio)
        };

        Self {
            t,
            centered_half: helpers::centered_half(t),
            strategy,
            decoder,
        }
    }

    /// Returns the plaintext modulus `t` used by this codec.
    #[must_use]
    #[inline]
    pub fn t(&self) -> T {
        self.t
    }
}

impl<T: FheUint> RoundedCodec<T> {
    /// Decodes a ciphertext residue into a canonical plaintext residue in `[0,t)`.
    ///
    /// # Correctness
    ///
    /// `value` must be in `[0,q)`. This range is not checked.
    ///
    /// # Panics
    ///
    /// Panics if conversion of the decoded residue to `M` fails.
    #[must_use]
    #[inline]
    pub fn decode_value<M: TryFrom<T>>(&self, value: T) -> M {
        self.decoder.value(value, self.t)
    }
    /// Replaces ciphertext residues with canonical plaintext residues in `[0,t)`.
    ///
    /// # Correctness
    ///
    /// Every input must be in `[0,q)`. This range is not checked.
    #[inline]
    pub fn decode_slice_assign(&self, values: &mut [T]) {
        self.decoder.assign(values, self.t);
    }
    /// Decodes into an equally sized output slice of canonical residues in `[0,t)`.
    ///
    /// # Correctness
    ///
    /// Every input must be in `[0,q)`. This range is not checked.
    ///
    /// # Panics
    ///
    /// Panics if the slices differ in length or conversion of a decoded residue
    /// to `M` fails. Length is checked before writing; conversion failure may
    /// leave earlier output elements modified.
    #[inline]
    pub fn decode_slice_to<M: TryFrom<T>>(&self, input: &[T], output: &mut [M]) {
        self.decoder.to(input, output, self.t);
    }
}

/// Exact divisibility uses the shared integer kernel; other ratios require
/// per-message rounding. The constructor selects these mutually exclusive cases.
#[derive(Clone, Copy, Debug)]
enum RoundedEncoding<T: FheUint> {
    Exact(IntegerScale<T>),
    Ratio(RatioEncoding<T>),
}

/// Arithmetic for non-integral q/t, selected before coefficient loops.
#[derive(Clone, Copy, Debug)]
enum RatioEncoding<T: FheUint> {
    DecomposedNative(DecomposedRatio<T, NativeRatio>),
    DecomposedExplicit(DecomposedRatio<T, ExplicitRatio<T>>),
    Native(NativeRatio),
    Wide(ExplicitRatio<T>),
}

#[derive(Clone, Copy, Debug)]
struct NativeRatio;

#[derive(Clone, Copy, Debug)]
struct ExplicitRatio<T: FheUint> {
    q: T,
}

/// Decomposes `q = floor*t + remainder` for single-word nearest rounding.
///
/// The constructor proves `(t-1)*remainder + floor(t/2)` fits one word.
/// For magnitude `m < t`, `round(m*q/t) = m*floor + round(m*remainder/t) < q`
/// because `q > t`; both the integer product and the final sum also fit,
/// including when `q = 2^T::BITS`.
#[derive(Clone, Copy, Debug)]
struct DecomposedRatio<T, M> {
    floor: T,
    remainder: T,
    modulus: M,
}

impl<T: FheUint, M> DecomposedRatio<T, M> {
    /// Requires `magnitude < t` and the same `t` used to establish the stored
    /// decomposition and biased-product bound.
    #[inline]
    fn encode_magnitude(&self, magnitude: T, t: T) -> T {
        magnitude * self.floor + (magnitude * self.remainder + (t >> 1u32)) / t
    }
}

impl<T: FheUint> DecomposedRatio<T, NativeRatio> {
    #[inline]
    fn neg_nonzero(&self, value: T) -> T {
        self.modulus.neg_nonzero(value)
    }
    #[inline]
    fn add_assign(&self, acc: &mut T, value: T) {
        self.modulus.add_assign(acc, value);
    }
}

impl<T: FheUint> DecomposedRatio<T, ExplicitRatio<T>> {
    #[inline]
    fn neg_nonzero(&self, value: T) -> T {
        self.modulus.neg_nonzero(value)
    }
    #[inline]
    fn add_assign(&self, acc: &mut T, value: T) {
        self.modulus.add_assign(acc, value);
    }
}

macro_rules! dispatch_ratio {
    ($ratio:expr,$codec:ident => $body:expr) => {
        match $ratio {
            RatioEncoding::DecomposedNative($codec) => $body,
            RatioEncoding::DecomposedExplicit($codec) => $body,
            RatioEncoding::Native($codec) => $body,
            RatioEncoding::Wide($codec) => $body,
        }
    };
}

impl NativeRatio {
    /// Rounds `(magnitude*2^T::BITS)/t` with ties upward.
    ///
    /// # Correctness
    ///
    /// Requires `magnitude < t` and `t >= 2`. The numerator's high word is
    /// `magnitude`, so it is below the divisor as required by `div_wide`.
    #[inline]
    fn encode_magnitude<T: FheUint>(&self, magnitude: T, t: T) -> T {
        T::div_wide(t >> 1u32, magnitude, t)
    }
    #[inline]
    fn neg_nonzero<T: FheUint>(&self, value: T) -> T {
        value.wrapping_neg()
    }
    #[inline]
    fn add_assign<T: FheUint>(&self, acc: &mut T, value: T) {
        *acc = acc.wrapping_add(value);
    }
}

impl<T: FheUint> ExplicitRatio<T> {
    /// Requires `magnitude < t` and `q > t >= 2`, so the rounded result fits
    /// in one word and is canonical modulo `q`.
    #[inline]
    fn encode_magnitude(&self, magnitude: T, t: T) -> T {
        mul_div_round(magnitude, self.q, t)
    }
    #[inline]
    fn neg_nonzero(&self, value: T) -> T {
        // A negative lift has magnitude >= 1; q > t makes its encoding nonzero.
        debug_assert!(value != T::ZERO);
        self.q - value
    }
    #[inline]
    fn add_assign(&self, acc: &mut T, value: T) {
        reduce_add_assign(self.q, acc, value);
    }
}

impl<T: FheUint> RoundedCodec<T> {
    /// Encodes a message in `[0,t)` into a canonical ciphertext residue in `[0,q)`.
    ///
    /// # Panics
    ///
    /// Panics if the message cannot be represented by `T` or lies outside the
    /// plaintext domain `[0,t)`.
    #[must_use]
    #[inline]
    pub fn encode_value<M>(&self, message: M, embedding: PlaintextEmbedding) -> T
    where
        M: TryInto<T>,
    {
        let message = checked_message(message, self.t);
        let (magnitude, is_negative) = match embedding {
            PlaintextEmbedding::Unsigned => (message, false),
            PlaintextEmbedding::Centered => {
                lift_centered_from_raw(message, self.t, self.centered_half)
            }
        };

        match &self.strategy {
            RoundedEncoding::Exact(scale) => {
                let encoded = scale.encode_magnitude(magnitude);
                if is_negative {
                    scale.neg_nonzero(encoded)
                } else {
                    encoded
                }
            }
            RoundedEncoding::Ratio(ratio) => dispatch_ratio!(ratio, codec => {
                let encoded = codec.encode_magnitude(magnitude, self.t);
                if is_negative {
                    codec.neg_nonzero(encoded)
                } else {
                    encoded
                }
            }),
        }
    }

    /// Encodes `messages` into canonical residues in `output` using the selected embedding.
    /// The previous output is overwritten; validation completes before any writes.
    ///
    /// # Panics
    ///
    /// Panics if the slices differ in length or a message lies outside the
    /// plaintext domain `[0,t)`.
    #[inline]
    pub fn encode_slice_to(&self, messages: &[T], output: &mut [T], embedding: PlaintextEmbedding) {
        assert_eq!(
            messages.len(),
            output.len(),
            "encoding slice length mismatch"
        );
        assert!(
            messages.iter().copied().max().is_none_or(|m| m < self.t),
            "message outside plaintext domain"
        );
        match &self.strategy {
            RoundedEncoding::Exact(scale) => {
                scale.apply::<false, _>(
                    output.iter_mut().zip(messages.iter().copied()),
                    self.t,
                    embedding,
                );
            }
            RoundedEncoding::Ratio(ratio) => {
                let t = self.t;
                dispatch_ratio!(ratio, codec => {
                    match embedding {
                        PlaintextEmbedding::Unsigned => {
                            for (&message, output) in messages.iter().zip(output) {
                                *output = codec.encode_magnitude(message, t);
                            }
                        }
                        PlaintextEmbedding::Centered => {
                            for (&message, output) in messages.iter().zip(output) {
                                let (magnitude, is_negative) =
                                    lift_centered_from_raw(message, t, self.centered_half);
                                let encoded = codec.encode_magnitude(magnitude, t);
                                *output = if is_negative {
                                    codec.neg_nonzero(encoded)
                                } else {
                                    encoded
                                };
                            }
                        }
                    }
                });
            }
        }
    }

    /// Replaces plaintext residues with canonical ciphertext residues using the
    /// selected embedding. Validation completes before any writes.
    ///
    /// # Panics
    ///
    /// Panics if a value lies outside the plaintext domain `[0,t)`.
    #[inline]
    pub fn encode_slice_assign(&self, values: &mut [T], embedding: PlaintextEmbedding) {
        assert!(
            values.iter().copied().max().is_none_or(|m| m < self.t),
            "message outside plaintext domain"
        );
        match &self.strategy {
            RoundedEncoding::Exact(scale) => {
                scale.apply::<false, _>(
                    values.iter_mut().map(|out| {
                        let m = *out;
                        (out, m)
                    }),
                    self.t,
                    embedding,
                );
            }
            RoundedEncoding::Ratio(ratio) => {
                let t = self.t;
                dispatch_ratio!(ratio, codec => {
                    match embedding {
                        PlaintextEmbedding::Unsigned => {
                            for value in values {
                                *value = codec.encode_magnitude(*value, t);
                            }
                        }
                        PlaintextEmbedding::Centered => {
                            for value in values {
                                let (magnitude, is_negative) =
                                    lift_centered_from_raw(*value, t, self.centered_half);
                                let encoded = codec.encode_magnitude(magnitude, t);
                                *value = if is_negative {
                                    codec.neg_nonzero(encoded)
                                } else {
                                    encoded
                                };
                            }
                        }
                    }
                });
            }
        }
    }
}

impl<T: FheUint> RoundedCodec<T> {
    /// Encodes `message` and adds it modulo `q` to `accumulator` without clearing it.
    ///
    /// # Correctness
    ///
    /// `accumulator` must be in `[0,q)` and remains canonical after the addition.
    /// Its range is not checked.
    ///
    /// # Panics
    ///
    /// Panics if the message cannot be represented by `T` or is outside `[0,t)`.
    #[inline]
    pub fn add_encode_value_assign<M>(
        &self,
        accumulator: &mut T,
        message: M,
        embedding: PlaintextEmbedding,
    ) where
        M: TryInto<T>,
    {
        let message = checked_message(message, self.t);
        let (magnitude, is_negative) = match embedding {
            PlaintextEmbedding::Unsigned => (message, false),
            PlaintextEmbedding::Centered => {
                lift_centered_from_raw(message, self.t, self.centered_half)
            }
        };

        match &self.strategy {
            RoundedEncoding::Exact(scale) => {
                let encoded = scale.encode_magnitude(magnitude);
                let encoded = if is_negative {
                    scale.neg_nonzero(encoded)
                } else {
                    encoded
                };
                scale.add_assign(accumulator, encoded);
            }
            RoundedEncoding::Ratio(ratio) => dispatch_ratio!(ratio, codec => {
                let encoded = codec.encode_magnitude(magnitude, self.t);
                let encoded = if is_negative {
                    codec.neg_nonzero(encoded)
                } else {
                    encoded
                };
                codec.add_assign(accumulator, encoded);
            }),
        }
    }

    /// Encodes each message and adds it modulo `q` to the corresponding accumulator.
    /// The accumulator is not cleared; message validation completes before any writes.
    ///
    /// # Correctness
    ///
    /// Every accumulator must be in `[0,q)` and remains canonical after addition.
    /// Accumulator ranges are not checked.
    ///
    /// # Panics
    ///
    /// Panics on a length mismatch or a message that cannot be represented by
    /// `T` or is outside `[0,t)`.
    #[inline]
    pub fn add_encode_slice_assign<M>(
        &self,
        accumulator: &mut [T],
        messages: &[M],
        embedding: PlaintextEmbedding,
    ) where
        M: Copy + TryInto<T>,
    {
        assert_eq!(
            accumulator.len(),
            messages.len(),
            "encoding slice length mismatch"
        );
        for &message in messages {
            let _: T = checked_message(message, self.t);
        }
        match &self.strategy {
            RoundedEncoding::Exact(scale) => {
                scale.apply::<true, _>(
                    accumulator.iter_mut().zip(
                        messages
                            .iter()
                            .copied()
                            .map(super::helpers::convert_message),
                    ),
                    self.t,
                    embedding,
                );
            }
            RoundedEncoding::Ratio(ratio) => {
                let t = self.t;
                dispatch_ratio!(ratio, codec => {
                    match embedding {
                        PlaintextEmbedding::Unsigned => {
                            for (accumulator, &message) in accumulator.iter_mut().zip(messages) {
                                let magnitude = super::helpers::convert_message(message);
                                let encoded = codec.encode_magnitude(magnitude, t);
                                codec.add_assign(accumulator, encoded);
                            }
                        }
                        PlaintextEmbedding::Centered => {
                            for (accumulator, &message) in accumulator.iter_mut().zip(messages) {
                                let message = super::helpers::convert_message(message);
                                let (magnitude, is_negative) =
                                    lift_centered_from_raw(message, t, self.centered_half);
                                let encoded = codec.encode_magnitude(magnitude, t);
                                let encoded = if is_negative {
                                    codec.neg_nonzero(encoded)
                                } else {
                                    encoded
                                };
                                codec.add_assign(accumulator, encoded);
                            }
                        }
                    }
                });
            }
        }
    }
}
