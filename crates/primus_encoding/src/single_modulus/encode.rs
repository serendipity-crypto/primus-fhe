use primus_integer::FheUint;

use super::{
    RoundedCodec,
    helpers::{checked_message, lift_centered_from_raw},
    rounded::{RoundedEncoding, dispatch_strategy},
};
use crate::PlaintextEmbedding;

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

        dispatch_strategy!(&self.strategy, codec => {
            let encoded = codec.encode_magnitude(magnitude, self.t);
            if is_negative {
                codec.neg(encoded)
            } else {
                encoded
            }
        })
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
        if let RoundedEncoding::Integer(scale) = &self.strategy {
            scale.apply::<false, _>(
                output.iter_mut().zip(messages.iter().copied()),
                self.t,
                embedding,
            );
            return;
        }
        let t = self.t;

        dispatch_strategy!(&self.strategy, codec => {
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
        if let RoundedEncoding::Integer(scale) = &self.strategy {
            scale.apply::<false, _>(
                values.iter_mut().map(|out| {
                    let m = *out;
                    (out, m)
                }),
                self.t,
                embedding,
            );
            return;
        }
        let t = self.t;

        dispatch_strategy!(&self.strategy, codec => {
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
