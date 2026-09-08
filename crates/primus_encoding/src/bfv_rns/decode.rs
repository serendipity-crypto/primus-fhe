use primus_data::DataMut;
use primus_factor::FactorMul;
use primus_integer::FheUint;
use primus_poly::{CrtPolynomial, Polynomial};
use primus_reduce::FieldContext;

use super::BfvRnsCodec;

impl<T, M> BfvRnsCodec<T, M>
where
    T: FheUint,
    M: FieldContext<T>,
{
    /// Exact scratch length for decoding `poly_length` coefficients.
    /// Returns zero for a single-modulus basis, otherwise
    /// `poly_length * moduli_count()` elements.
    ///
    /// # Panics
    ///
    /// Panics if the required scratch length overflows `usize`.
    #[must_use]
    pub fn decode_scratch_len(&self, poly_length: usize) -> usize {
        self.converter_q_to_t_gamma
            .fast_convert_array_scratch_len(poly_length)
    }

    /// Decodes coefficient-domain CRT residues into canonical coefficients in `[0,t)`.
    ///
    /// `msg_mod_q` is overwritten as workspace; `msg` receives the decoded
    /// coefficients. `fast_convert_buffer` must contain exactly
    /// `decode_scratch_len(msg.len())` elements and need not be zeroed.
    ///
    /// # Correctness
    ///
    /// `msg_mod_q` must use modulus-major coefficient layout in [`Self::base_q`]
    /// order: each chunk of `msg.len()` elements contains canonical residues
    /// modulo its corresponding `q_i`. Residue ranges and basis/domain are not
    /// checked; NTT data must be inverse-transformed before this call.
    ///
    /// With multiple ciphertext moduli, the destination moduli `t` and `gamma`
    /// must support the additional dot-product input ranges documented by
    /// [`primus_rns::BaseConverter::fast_convert`]. [`FieldContext`] alone does
    /// not guarantee this. A single-modulus basis uses direct reduction.
    ///
    /// For phase `c = delta*m + e (mod Q)`, where `delta = floor(Q/t)` and `m`
    /// is the chosen integer lift, a sufficient recovery condition is
    /// `abs(t*e - (Q % t)*m)/Q + k/gamma < 1/2`, with `k = moduli_count()`.
    /// This includes encoding drift and the fast base-conversion error.
    ///
    /// # Panics
    ///
    /// Panics if `msg_mod_q.len()` differs from `msg.len() * moduli_count()`,
    /// that product overflows `usize`, or `fast_convert_buffer.len()` differs
    /// from `decode_scratch_len(msg.len())`. Lengths are checked before any writes.
    pub fn decode_coeffs_to<A, B>(
        &self,
        msg_mod_q: &mut CrtPolynomial<A>,
        msg: &mut Polynomial<B>,
        fast_convert_buffer: &mut [T],
    ) where
        A: DataMut<Elem = T>,
        B: DataMut<Elem = T>,
    {
        let poly_length = msg.as_ref().len();
        let rns_poly_len = self.rns_poly_len(poly_length);
        assert_eq!(
            msg_mod_q.as_ref().len(),
            rns_poly_len,
            "RNS input length mismatch"
        );
        assert_eq!(
            fast_convert_buffer.len(),
            self.decode_scratch_len(poly_length),
            "RNS scratch length mismatch"
        );
        if poly_length == 0 {
            return;
        }

        let t = self.t;
        let gamma = self.gamma;
        let t_modulus = self.t_modulus;
        let decode_factor_mod_t = self.decode_factor_mod_t_gamma[0];
        let decode_factor_mod_gamma = self.decode_factor_mod_t_gamma[1];
        let inv_gamma_mod_t = self.inv_gamma_mod_t;
        let msg = msg.as_mut();

        // Let x = t*gamma*c mod Q. Fast conversion yields x + alpha*Q,
        // 0 <= alpha < k. Multiplying by -Q^-1 gives
        // z = floor(t*gamma*c/Q) - alpha modulo t and gamma.
        // Subtract centered(z mod gamma) in Z_t, then divide by gamma.
        msg_mod_q.mul_factor_assign(
            self.t_gamma_factor_mod_q.as_ref(),
            poly_length,
            &self.moduli_values,
        );

        self.converter_q_to_t_gamma
            .fast_convert_array_to_pair_iter(msg_mod_q.as_ref(), poly_length, fast_convert_buffer)
            .zip(msg.iter_mut())
            .for_each(|((y_t, y_gamma), m)| {
                // Distribute γ^-1 over both terms so the converted coefficient
                // can be finalized here, without a second output pass. Shoup
                // accepts the full-width correction; reducing it to t first
                // is unnecessary.
                let y_t = decode_factor_mod_t.factor_mul_modulo(y_t, t);
                let y_gamma = decode_factor_mod_gamma.factor_mul_modulo(y_gamma, gamma);

                *m = if y_gamma > (gamma >> 1u32) {
                    t_modulus.reduce_add(y_t, inv_gamma_mod_t.factor_mul_modulo(gamma - y_gamma, t))
                } else {
                    t_modulus.reduce_sub(y_t, inv_gamma_mod_t.factor_mul_modulo(y_gamma, t))
                };
            });
    }
}
