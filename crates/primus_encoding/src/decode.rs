//! Shared nearest-cell decoding for `RoundedCodec` and `ScaledCodec`.

use super::helpers::{centered_half, mul_div_round, narrow_mul_div_round, try_from_decoded};
use primus_integer::FheUint;

/// Arithmetic selected for `round(c*t/q) mod t`, with canonical `c` and ties upward.
/// Exact divisibility takes precedence over native/explicit modulus dispatch.
/// The owning codec supplies the same `t` used to construct this strategy to
/// every decoding operation, independently of its encoding convention.
#[derive(Clone, Copy, Debug)]
pub(super) enum DecodeStrategy<T: FheUint> {
    /// `t` divides `q` and `q/t = 2^shift`; stores `shift >= 1`.
    /// Decodes by rounding `c / 2^shift`, then reducing modulo `t`.
    /// Both moduli being powers of two is the common case, but not required
    /// (e.g. `q=18, t=9`). A rounded scale being a power of two is insufficient.
    Shift(u32),
    /// `t` divides explicit `q`, but the exact scale `delta=q/t` is not a power
    /// of two (e.g. `q=45, t=9`). Stores `delta` and rounds `c/delta` using
    /// quotient and remainder, then reduces modulo `t`.
    Divide(T),
    /// Native `q=2^T::BITS` with `t` not dividing `q` (equivalently, `t` is not
    /// a power of two). Rounds `c*t/q` using the high word of the product.
    Native,
    /// Explicit `q` with `t` not dividing `q` and `q*t <= T::MAX`.
    /// Stores `q`; the canonical input guarantees `c*t` fits one word, so
    /// quotient/remainder rounding needs no wide product.
    Narrow(T),
    /// Explicit `q` with `t` not dividing `q` and `q*t > T::MAX`.
    /// Stores `q`; rounds `c*t/q` with a two-word product and wide division.
    /// This is selected by the product bound, not merely by the size of `q`.
    Wide(T),
}

impl<T: FheUint> DecodeStrategy<T> {
    /// Uses a validated modulus pair and its exact quotient/remainder.
    pub(super) fn from_ratio(t: T, q: Option<T>, delta: T, remainder: T) -> Self {
        if remainder == T::ZERO {
            if delta.is_power_of_two() {
                Self::Shift(delta.trailing_zeros())
            } else {
                Self::Divide(delta)
            }
        } else {
            match q {
                None => Self::Native,
                Some(q) if q.checked_mul(t).is_some() => Self::Narrow(q),
                Some(q) => Self::Wide(q),
            }
        }
    }

    #[inline]
    pub(super) fn value<M: TryFrom<T>>(&self, value: T, t: T) -> M {
        let mut output = T::ZERO;
        self.apply(core::iter::once((value, &mut output)), t);
        try_from_decoded(output)
    }

    #[inline]
    pub(super) fn assign(&self, values: &mut [T], t: T) {
        self.apply(values.iter_mut().map(|out| (*out, out)), t);
    }

    #[inline]
    pub(super) fn to<M: TryFrom<T>>(&self, input: &[T], output: &mut [M], t: T) {
        assert_eq!(input.len(), output.len(), "decoding slice length mismatch");
        self.apply(input.iter().copied().zip(output), t);
    }

    // All phases are canonical. Raw rounding lies in [0,t]; select reduction
    // outside the loop and avoid an overflowing c + delta/2 intermediate.
    #[inline]
    fn apply<'a, M: TryFrom<T> + 'a, I: Iterator<Item = (T, &'a mut M)>>(&self, input: I, t: T) {
        match *self {
            Self::Shift(shift) if t.is_power_of_two() => map(input, |c| {
                ((c >> shift) + ((c >> (shift - 1)) & T::ONE)) & (t - T::ONE)
            }),
            Self::Shift(shift) => map(input, |c| {
                canonical((c >> shift) + ((c >> (shift - 1)) & T::ONE), t)
            }),
            Self::Divide(delta) => map(input, |c| {
                let (d, r) = c.div_rem(delta);
                canonical(
                    d + if r >= centered_half(delta) {
                        T::ONE
                    } else {
                        T::ZERO
                    },
                    t,
                )
            }),
            Self::Native => map(input, |c| {
                canonical(c.carrying_mul_hw(t, T::ONE << (T::BITS - 1)), t)
            }),
            Self::Narrow(q) => map(input, |c| canonical(narrow_mul_div_round(c, t, q), t)),
            Self::Wide(q) => map(input, |c| canonical(mul_div_round(c, t, q), t)),
        }
    }
}

#[inline]
fn canonical<T: FheUint>(value: T, t: T) -> T {
    if value >= t { value - t } else { value }
}

#[inline]
fn map<'a, T: FheUint, M: TryFrom<T> + 'a, I: Iterator<Item = (T, &'a mut M)>, F: Fn(T) -> T>(
    input: I,
    decode: F,
) {
    for (value, out) in input {
        *out = try_from_decoded(decode(value));
    }
}
