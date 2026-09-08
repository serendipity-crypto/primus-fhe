use super::helpers::{centered_half, lift_centered_from_raw};
use crate::PlaintextEmbedding;
use primus_integer::FheUint;
use primus_modulus::common::uint::{reduce_add, reduce_neg};

/// Integer scaling shared by fixed-scale encoding and exact q/t encoding.
/// Constructors of the owning codec guarantee (t-1)*delta < q, so magnitude
/// products fit one word and are canonical without modular multiplication.
#[derive(Clone, Copy, Debug)]
pub(super) struct IntegerScale<T: FheUint> {
    q: Option<T>,
    op: ScaleOp<T>,
}

/// Magnitude scaling selected at construction, independent of the modulus.
#[derive(Clone, Copy, Debug)]
enum ScaleOp<T> {
    Shift { shift: u32 },
    Multiply { delta: T },
}

impl<T: FheUint> IntegerScale<T> {
    pub(super) fn new(delta: T, q: Option<T>) -> Self {
        let op = if delta.is_power_of_two() {
            ScaleOp::Shift {
                shift: delta.trailing_zeros(),
            }
        } else {
            ScaleOp::Multiply { delta }
        };
        Self { op, q }
    }

    #[inline]
    pub(super) fn encode_magnitude(&self, m: T) -> T {
        match self.op {
            ScaleOp::Shift { shift } => m << shift,
            ScaleOp::Multiply { delta } => m * delta,
        }
    }
    #[inline]
    pub(super) fn neg(&self, value: T) -> T {
        match self.q {
            None => value.wrapping_neg(),
            Some(q) => reduce_neg(q, value),
        }
    }
    #[inline]
    pub(super) fn add_assign(&self, acc: &mut T, value: T) {
        *acc = match self.q {
            None => acc.wrapping_add(value),
            Some(q) => reduce_add(q, *acc, value),
        };
    }

    /// Applies scaling to validated messages, optionally adding to canonical output.
    /// The iterator also permits in-place encoding without allocating a copy.
    // Keep dispatch inline so callers can specialize the scale and modulus.
    #[inline(always)]
    pub(super) fn apply<'a, const ADD: bool, I>(
        &self,
        input: I,
        t: T,
        embedding: PlaintextEmbedding,
    ) where
        I: Iterator<Item = (&'a mut T, T)>,
        T: 'a,
    {
        match self.q {
            None => self.apply_scale::<ADD, _, _, _>(
                input,
                t,
                embedding,
                |x| x.wrapping_neg(),
                |a, b| a.wrapping_add(b),
            ),
            Some(q) => self.apply_scale::<ADD, _, _, _>(
                input,
                t,
                embedding,
                |x| reduce_neg(q, x),
                |a, b| reduce_add(q, a, b),
            ),
        }
    }

    #[inline]
    fn apply_scale<'a, const ADD: bool, I, N, A>(
        &self,
        input: I,
        t: T,
        embedding: PlaintextEmbedding,
        neg: N,
        add: A,
    ) where
        I: Iterator<Item = (&'a mut T, T)>,
        N: Fn(T) -> T,
        A: Fn(T, T) -> T,
        T: 'a,
    {
        match self.op {
            ScaleOp::Shift { shift } => {
                apply_kernel::<T, ADD, _, _, _, _>(input, t, embedding, |m| m << shift, neg, add);
            }
            ScaleOp::Multiply { delta } => {
                apply_kernel::<T, ADD, _, _, _, _>(input, t, embedding, |m| m * delta, neg, add);
            }
        }
    }
}

/// Modulus, scale and embedding dispatch is performed before entering the loop.
#[inline]
fn apply_kernel<'a, T: FheUint + 'a, const ADD: bool, I, E, N, A>(
    input: I,
    t: T,
    embedding: PlaintextEmbedding,
    encode: E,
    neg: N,
    add: A,
) where
    I: Iterator<Item = (&'a mut T, T)>,
    E: Fn(T) -> T,
    N: Fn(T) -> T,
    A: Fn(T, T) -> T,
{
    match embedding {
        PlaintextEmbedding::Unsigned => {
            for (out, m) in input {
                let value = encode(m);
                *out = if ADD { add(*out, value) } else { value };
            }
        }
        PlaintextEmbedding::Centered => {
            let half = centered_half(t);
            for (out, m) in input {
                let (magnitude, negative) = lift_centered_from_raw(m, t, half);
                let value = encode(magnitude);
                let value = if negative { neg(value) } else { value };
                *out = if ADD { add(*out, value) } else { value };
            }
        }
    }
}
