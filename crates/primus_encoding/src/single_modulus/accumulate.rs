use primus_integer::FheUint;

use super::{
    RoundedCodec,
    helpers::{checked_message, lift_centered_from_raw},
    rounded::{RoundedEncoding, dispatch_strategy},
};
use crate::PlaintextEmbedding;

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

        dispatch_strategy!(&self.strategy, codec => {
            let encoded = codec.encode_magnitude(magnitude, self.t);
            let encoded = if is_negative {
                codec.neg(encoded)
            } else {
                encoded
            };
            codec.add_assign(accumulator, encoded);
        });
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
        if let RoundedEncoding::Integer(scale) = &self.strategy {
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
            return;
        }
        let t = self.t;

        dispatch_strategy!(&self.strategy, codec => {
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
