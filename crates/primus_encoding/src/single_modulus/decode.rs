use super::helpers::{centered_half, mul_div_round, narrow_mul_div_round, try_from_decoded};
use primus_integer::FheUint;

/// Parameters for nearest-cell decoding, independent of the encoding convention.
#[derive(Clone, Copy, Debug)]
pub(super) struct DecodeParams<T: FheUint> {
    t: T,
    strategy: DecodeStrategy<T>,
}

/// Arithmetic selected for `round(c*t/q) mod t`, with canonical `c` and ties upward.
/// Exact divisibility takes precedence over native/explicit modulus dispatch.
#[derive(Clone, Copy, Debug)]
enum DecodeStrategy<T: FheUint> {
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

impl<T: FheUint> DecodeParams<T> {
    /// Uses a validated modulus pair and its exact quotient/remainder.
    pub(super) fn from_ratio(t: T, q: Option<T>, delta: T, remainder: T) -> Self {
        let strategy = if remainder == T::ZERO {
            if delta.is_power_of_two() {
                DecodeStrategy::Shift(delta.trailing_zeros())
            } else {
                DecodeStrategy::Divide(delta)
            }
        } else {
            match q {
                None => DecodeStrategy::Native,
                Some(q) if q.checked_mul(t).is_some() => DecodeStrategy::Narrow(q),
                Some(q) => DecodeStrategy::Wide(q),
            }
        };
        Self { t, strategy }
    }

    #[inline]
    pub(super) fn value<M: TryFrom<T>>(&self, value: T) -> M {
        let mut output = T::ZERO;
        self.apply(core::iter::once((value, &mut output)));
        try_from_decoded(output)
    }

    #[inline]
    pub(super) fn assign(&self, values: &mut [T]) {
        self.apply(values.iter_mut().map(|out| (*out, out)));
    }

    #[inline]
    pub(super) fn to<M: TryFrom<T>>(&self, input: &[T], output: &mut [M]) {
        assert_eq!(input.len(), output.len(), "decoding slice length mismatch");
        self.apply(input.iter().copied().zip(output));
    }

    // All phases are canonical. Raw rounding lies in [0,t]; select reduction
    // outside the loop and avoid an overflowing c + delta/2 intermediate.
    #[inline]
    fn apply<'a, M: TryFrom<T> + 'a, I: Iterator<Item = (T, &'a mut M)>>(&self, input: I) {
        let t = self.t;
        match self.strategy {
            DecodeStrategy::Shift(shift) if t.is_power_of_two() => map(input, |c| {
                ((c >> shift) + ((c >> (shift - 1)) & T::ONE)) & (t - T::ONE)
            }),
            DecodeStrategy::Shift(shift) => map(input, |c| {
                canonical((c >> shift) + ((c >> (shift - 1)) & T::ONE), t)
            }),
            DecodeStrategy::Divide(delta) => map(input, |c| {
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
            DecodeStrategy::Native => map(input, |c| {
                canonical(c.carrying_mul_hw(t, T::ONE << (T::BITS - 1)), t)
            }),
            DecodeStrategy::Narrow(q) => {
                map(input, |c| canonical(narrow_mul_div_round(c, t, q), t))
            }
            DecodeStrategy::Wide(q) => map(input, |c| canonical(mul_div_round(c, t, q), t)),
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
