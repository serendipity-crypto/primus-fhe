use super::{
    decode::DecodeStrategy,
    helpers::{centered_half, checked_message, lift_centered_from_raw, modulus_div_rem},
    integer_scale::IntegerScale,
};
use crate::PlaintextEmbedding;
use primus_integer::FheUint;

/// Fixed rounded scaling: `lift(m) * delta mod q`, returned in `[0,q)`, where
/// `delta = round(q/t)` with ties upward.
///
/// This preserves the coefficient scaling used by single-modulus GLWE/NTRU.
/// It is distinct from BFV's `floor(Q/t)` scaling and per-message rounding.
/// Decoding rounds `c*t/q` with ties upward, modulo `t`. For integer lift `m` and noise `e`,
/// recovery is guaranteed when `abs((t*delta-q)*m + t*e) < q/2`.
/// Accumulators and decoding inputs must be canonical ciphertext residues.
#[derive(Clone, Copy, Debug)]
pub struct ScaledCodec<T: FheUint> {
    t: T,
    decoder: DecodeStrategy<T>,
    scale: IntegerScale<T>,
}

impl<T: FheUint> ScaledCodec<T> {
    /// Constructs a fixed-scale codec; `None` denotes `q = 2^T::BITS`.
    ///
    /// The scale-recovery bound below is a conservative sufficient condition
    /// for noiseless recovery with either lift.
    ///
    /// # Panics
    ///
    /// Panics unless `t >= 2`, `q > t`, and
    /// `abs(t*round(q/t)-q)*(t-1) < q/2`.
    #[must_use]
    pub fn new(t: T, q: Option<T>) -> Self {
        let (floor, remainder) = modulus_div_rem(t, q);
        let decoder = DecodeStrategy::from_ratio(t, q, floor, remainder);
        let delta = floor
            + if remainder >= centered_half(t) {
                T::ONE
            } else {
                T::ZERO
            };
        let drift = remainder.min(t - remainder);
        let (lo, hi) = drift.carrying_mul(t - T::ONE, T::ZERO);
        let fits = hi == T::ZERO
            && match q {
                Some(q) => lo <= (q - T::ONE) / T::TWO,
                None => lo < (T::ONE << (T::BITS - 1)),
            };
        assert!(
            fits,
            "ciphertext modulus too small for fixed rounded scaling"
        );
        // If epsilon=t*delta-q <= 0, (t-1)*delta < q immediately.
        // Otherwise epsilon*(t-1)<q implies epsilon<delta, hence
        // (t-1)*delta=q+epsilon-delta<q. Products need no modular reduction.
        let scale = IntegerScale::new(delta, q);
        Self { t, decoder, scale }
    }

    /// Returns the plaintext modulus.
    #[must_use]
    #[inline]
    pub fn t(&self) -> T {
        self.t
    }

    /// Encodes a residue in `[0,t)` into a canonical residue in `[0,q)` with the selected lift.
    ///
    /// # Panics
    ///
    /// Panics if the message cannot be represented by `T` or is outside `[0,t)`.
    #[must_use]
    #[inline]
    pub fn encode_value<M: TryInto<T>>(&self, message: M, embedding: PlaintextEmbedding) -> T {
        let message = checked_message(message, self.t());
        self.encode_raw(message, embedding)
    }

    /// Encodes a validated plaintext residue `message < t`.
    /// The positive scale and `(t-1)*delta < q` ensure that negative lifts have
    /// nonzero encodings, permitting explicit negation as `q - value`.
    #[inline]
    fn encode_raw(&self, message: T, embedding: PlaintextEmbedding) -> T {
        let (m, negative) = match embedding {
            PlaintextEmbedding::Unsigned => (message, false),
            PlaintextEmbedding::Centered => {
                lift_centered_from_raw(message, self.t(), super::helpers::centered_half(self.t()))
            }
        };
        let value = self.scale.encode_magnitude(m);
        if negative {
            self.scale.neg_nonzero(value)
        } else {
            value
        }
    }

    /// Encodes messages into an equally sized output slice of canonical residues in `[0,q)`.
    /// The previous output is overwritten; validation completes before any writes.
    ///
    /// # Panics
    ///
    /// Panics on a length mismatch or messages outside `[0,t)`.
    #[inline]
    pub fn encode_slice_to(&self, messages: &[T], output: &mut [T], embedding: PlaintextEmbedding) {
        self.validate(messages, output.len());
        self.scale.apply::<false, _>(
            output.iter_mut().zip(messages.iter().copied()),
            self.t,
            embedding,
        );
    }

    /// Replaces plaintext residues with canonical ciphertext residues in `[0,q)`.
    /// Validation completes before any writes.
    ///
    /// # Panics
    ///
    /// Panics on messages outside `[0,t)`.
    #[inline]
    pub fn encode_slice_assign(&self, values: &mut [T], embedding: PlaintextEmbedding) {
        self.validate(values, values.len());
        self.scale.apply::<false, _>(
            values.iter_mut().map(|out| {
                let m = *out;
                (out, m)
            }),
            self.t,
            embedding,
        );
    }

    /// Adds encoded messages modulo `q` to the accumulator without clearing it.
    /// Validation of messages and lengths completes before any writes.
    ///
    /// # Correctness
    ///
    /// Every accumulator must be in `[0,q)` and remains canonical after addition.
    /// Accumulator ranges are not checked.
    ///
    /// # Panics
    ///
    /// Panics on a length mismatch or messages outside `[0,t)`.
    #[inline]
    pub fn add_encode_slice_assign(
        &self,
        accumulator: &mut [T],
        messages: &[T],
        embedding: PlaintextEmbedding,
    ) {
        self.validate(messages, accumulator.len());
        self.scale.apply::<true, _>(
            accumulator.iter_mut().zip(messages.iter().copied()),
            self.t,
            embedding,
        );
    }

    #[inline]
    fn validate(&self, messages: &[T], output_len: usize) {
        assert_eq!(messages.len(), output_len, "encoding slice length mismatch");
        assert!(
            messages.iter().copied().max().is_none_or(|m| m < self.t()),
            "message outside plaintext domain"
        );
    }

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
    pub fn decode_slice_to<M: TryFrom<T>>(&self, input: &[T], output: &mut [M]) {
        self.decoder.to(input, output, self.t);
    }
}
