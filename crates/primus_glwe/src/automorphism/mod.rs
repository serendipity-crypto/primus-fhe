//! Homomorphic automorphisms with backend-specific key switching.
use primus_integer::{FheUint, WrappingNeg};
use primus_modulus::PowOf2Modulus;
use primus_reduce::{ReduceMul, RingContext};

mod fourier;
mod ntt;
pub use fourier::{FourierGlweAutomorphismContext, FourierGlweAutomorphismKey};
pub use ntt::{NttGlweAutomorphismContext, NttGlweAutomorphismKey};

#[derive(Clone, Copy)]
struct CoefficientSource {
    index: u32,
    negate: bool,
}

/// Precomputed coefficient permutation for `X -> X^degree` in
/// `Z_q[X]/(X^N + 1)`.
#[derive(Clone)]
pub(super) struct CoeffAutoPermutation {
    sources: Vec<CoefficientSource>,
}

impl CoeffAutoPermutation {
    pub(super) fn new(degree: usize, poly_length: usize) -> Self {
        assert!(
            degree < 2 * poly_length && degree % 2 == 1,
            "GLWE automorphism degree must be odd and less than 2N"
        );
        let modulus = PowOf2Modulus::new(2 * poly_length);
        let mut sources = vec![
            CoefficientSource {
                index: 0,
                negate: false,
            };
            poly_length
        ];
        for source in 0..poly_length {
            let mapped = modulus.reduce_mul(source, degree);
            let (destination, negate) = if mapped < poly_length {
                (mapped, false)
            } else {
                (mapped - poly_length, true)
            };
            sources[destination] = CoefficientSource {
                index: source as u32,
                negate,
            };
        }
        Self { sources }
    }

    #[inline]
    fn poly_length(&self) -> usize {
        self.sources.len()
    }

    pub(super) fn apply_secret<T: FheUint>(
        &self,
        input: &[T::SignedInteger],
        output: &mut [T::SignedInteger],
    ) {
        debug_assert_eq!(input.len(), self.poly_length());
        debug_assert_eq!(output.len(), self.poly_length());
        for (output, source) in output.iter_mut().zip(&self.sources) {
            let value = input[source.index as usize];
            *output = if source.negate {
                value.wrapping_neg()
            } else {
                value
            };
        }
    }

    pub(super) fn apply_residues<T, M>(&self, input: &[T], output: &mut [T], modulus: M)
    where
        T: FheUint,
        M: RingContext<T>,
    {
        debug_assert_eq!(input.len(), self.poly_length());
        debug_assert_eq!(output.len(), self.poly_length());
        for (output, source) in output.iter_mut().zip(&self.sources) {
            let value = input[source.index as usize];
            *output = if source.negate {
                modulus.reduce_neg(value)
            } else {
                value
            };
        }
    }
}
