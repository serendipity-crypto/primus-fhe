use super::decode::DecodeStrategy;
use super::helpers;
use super::helpers::{
    checked_message, lift_centered_from_raw, mul_div_round, narrow_mul_div_round,
};
use super::integer_scale::IntegerScale;
use crate::PlaintextEmbedding;
use primus_integer::FheUint;
use primus_modulus::common::uint::{reduce_add_assign, reduce_neg};

/// Per-message rounding: `round(lift(m) * q / t)` with ties away from zero.
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
            let ratio = match q {
                None => RatioEncoding::Native(NativeRatio),
                Some(q) if q.checked_mul(t).is_some() => RatioEncoding::Narrow(ExplicitRatio { q }),
                Some(q) => RatioEncoding::Wide(ExplicitRatio { q }),
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
    /// Decodes a canonical ciphertext residue. Panics if M cannot hold the result.
    #[must_use]
    #[inline]
    pub fn decode_value<M: TryFrom<T>>(&self, value: T) -> M {
        self.decoder.value(value, self.t)
    }
    /// Decodes canonical ciphertext residues in place.
    #[inline]
    pub fn decode_slice_assign(&self, values: &mut [T]) {
        self.decoder.assign(values, self.t);
    }
    /// Decodes into an equally sized output slice.
    /// Panics on length mismatch or if M cannot hold the result.
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
    Native(NativeRatio),
    Narrow(ExplicitRatio<T, false>),
    Wide(ExplicitRatio<T, true>),
}

#[derive(Clone, Copy, Debug)]
struct NativeRatio;

/// WIDE selects the product width at monomorphization, outside coefficient loops.
#[derive(Clone, Copy, Debug)]
struct ExplicitRatio<T: FheUint, const WIDE: bool> {
    q: T,
}

macro_rules! dispatch_ratio {
    ($ratio:expr,$codec:ident => $body:expr) => {
        match $ratio {
            RatioEncoding::Native($codec) => $body,
            RatioEncoding::Narrow($codec) => $body,
            RatioEncoding::Wide($codec) => $body,
        }
    };
}

impl NativeRatio {
    #[inline]
    fn encode_magnitude<T: FheUint>(&self, magnitude: T, t: T) -> T {
        T::div_wide(t >> 1u32, magnitude, t)
    }
    #[inline]
    fn neg<T: FheUint>(&self, value: T) -> T {
        value.wrapping_neg()
    }
    #[inline]
    fn add_assign<T: FheUint>(&self, acc: &mut T, value: T) {
        *acc = acc.wrapping_add(value);
    }
}

impl<T: FheUint, const WIDE: bool> ExplicitRatio<T, WIDE> {
    #[inline]
    fn encode_magnitude(&self, magnitude: T, t: T) -> T {
        if WIDE {
            mul_div_round(magnitude, self.q, t)
        } else {
            narrow_mul_div_round(magnitude, self.q, t)
        }
    }
    #[inline]
    fn neg(&self, value: T) -> T {
        reduce_neg(self.q, value)
    }
    #[inline]
    fn add_assign(&self, acc: &mut T, value: T) {
        reduce_add_assign(self.q, acc, value);
    }
}

impl<T: FheUint> RoundedCodec<T> {
    /// Encodes one message using the selected embedding.
    ///
    /// # Panics
    ///
    /// Panics if the message cannot be represented by `T` or lies outside the
    /// plaintext domain.
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
                    scale.neg(encoded)
                } else {
                    encoded
                }
            }
            RoundedEncoding::Ratio(ratio) => dispatch_ratio!(ratio, codec => {
                let encoded = codec.encode_magnitude(magnitude, self.t);
                if is_negative {
                    codec.neg(encoded)
                } else {
                    encoded
                }
            }),
        }
    }

    /// Encodes `messages` into `output` using the selected embedding.
    ///
    /// # Panics
    ///
    /// Panics if the slices differ in length or a message lies outside the
    /// plaintext domain.
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
                                    codec.neg(encoded)
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

    /// Encodes all values in place using the selected embedding.
    ///
    /// # Panics
    ///
    /// Panics if a value lies outside the plaintext domain.
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
                                    codec.neg(encoded)
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
    /// Encodes `message` and modular-adds into canonical `accumulator`.
    ///
    /// # Panics
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
                    scale.neg(encoded)
                } else {
                    encoded
                };
                scale.add_assign(accumulator, encoded);
            }
            RoundedEncoding::Ratio(ratio) => dispatch_ratio!(ratio, codec => {
                let encoded = codec.encode_magnitude(magnitude, self.t);
                let encoded = if is_negative {
                    codec.neg(encoded)
                } else {
                    encoded
                };
                codec.add_assign(accumulator, encoded);
            }),
        }
    }

    /// Encodes each message and adds into the corresponding canonical accumulator.
    ///
    /// # Panics
    /// Panics on a length mismatch or a message that cannot be represented by
    /// `T` or is outside `[0,t)`. Validation completes before any writes.
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
                                    codec.neg(encoded)
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
