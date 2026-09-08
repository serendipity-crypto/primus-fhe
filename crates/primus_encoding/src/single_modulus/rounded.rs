use super::decode::DecodeParams;
use super::helpers;
use super::helpers::{mul_div_round, narrow_mul_div_round};
use super::scale::IntegerScale;
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
    pub(super) t: T,
    pub(super) centered_half: T,
    pub(super) strategy: RoundedEncoding<T>,
    decoder: DecodeParams<T>,
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
        let decoder = DecodeParams::from_ratio(t, q, floor, remainder);
        let strategy = if remainder == T::ZERO {
            RoundedEncoding::Integer(IntegerScale::new(floor, q))
        } else {
            match q {
                None => RoundedEncoding::Native(NativeRatio),
                Some(q) if q.checked_mul(t).is_some() => {
                    RoundedEncoding::Narrow(ExplicitRatio { q })
                }
                Some(q) => RoundedEncoding::Wide(ExplicitRatio { q }),
            }
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
        self.decoder.value(value)
    }
    /// Decodes canonical ciphertext residues in place.
    #[inline]
    pub fn decode_slice_assign(&self, values: &mut [T]) {
        self.decoder.assign(values);
    }
    /// Decodes into an equally sized output slice.
    /// Panics on length mismatch or if M cannot hold the result.
    #[inline]
    pub fn decode_slice_to<M: TryFrom<T>>(&self, input: &[T], output: &mut [M]) {
        self.decoder.to(input, output);
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum RoundedEncoding<T: FheUint> {
    Integer(IntegerScale<T>),
    Native(NativeRatio),
    Narrow(ExplicitRatio<T, false>),
    Wide(ExplicitRatio<T, true>),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct NativeRatio;

/// WIDE selects the product width at monomorphization, outside coefficient loops.
#[derive(Clone, Copy, Debug)]
pub(super) struct ExplicitRatio<T: FheUint, const WIDE: bool> {
    pub(super) q: T,
}

macro_rules! dispatch_strategy {
    ($strategy:expr,$codec:ident => $body:expr) => {
        match $strategy {
            RoundedEncoding::Integer($codec) => $body,
            RoundedEncoding::Native($codec) => $body,
            RoundedEncoding::Narrow($codec) => $body,
            RoundedEncoding::Wide($codec) => $body,
        }
    };
}
pub(super) use dispatch_strategy;

impl NativeRatio {
    #[inline]
    pub(super) fn encode_magnitude<T: FheUint>(&self, magnitude: T, t: T) -> T {
        T::div_wide(t >> 1u32, magnitude, t)
    }
    #[inline]
    pub(super) fn neg<T: FheUint>(&self, value: T) -> T {
        value.wrapping_neg()
    }
    #[inline]
    pub(super) fn add_assign<T: FheUint>(&self, acc: &mut T, value: T) {
        *acc = acc.wrapping_add(value);
    }
}

impl<T: FheUint, const WIDE: bool> ExplicitRatio<T, WIDE> {
    #[inline]
    pub(super) fn encode_magnitude(&self, magnitude: T, t: T) -> T {
        if WIDE {
            mul_div_round(magnitude, self.q, t)
        } else {
            narrow_mul_div_round(magnitude, self.q, t)
        }
    }
    #[inline]
    pub(super) fn neg(&self, value: T) -> T {
        reduce_neg(self.q, value)
    }
    #[inline]
    pub(super) fn add_assign(&self, acc: &mut T, value: T) {
        reduce_add_assign(self.q, acc, value);
    }
}
