use primus_data::DataMut;
use primus_factor::FactorSliceOps;
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
    ///
    /// # Panics
    /// Panics if the RNS shape cannot be represented by `usize`.
    #[must_use]
    pub fn decode_scratch_len(&self, poly_length: usize) -> usize {
        self.rns_poly_len(poly_length)
    }

    /// Decodes canonical coefficient-domain CRT residues into coefficients modulo `t`.
    ///
    /// The caller must inverse-transform NTT data before creating the CRT view.
    /// For phase `c = delta*m + e` (with the chosen integer lift of `m`), a
    /// sufficient recovery condition is
    /// `abs(t*e - (Q % t)*m)/Q + k/gamma < 1/2`, where `k = moduli_count()`.
    /// This includes encoding drift and the fast base-conversion error.
    ///
    /// `msg_mod_q` is used as mutable workspace and is overwritten. The
    /// conversion buffer must contain one RNS polynomial.
    ///
    /// # Panics
    ///
    /// Panics unless input and scratch each contain exactly
    /// `output_length * moduli_count()` elements.
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
            rns_poly_len,
            "RNS scratch length mismatch"
        );
        if poly_length == 0 {
            return;
        }

        let t = self.t;
        let gamma = self.gamma;
        let t_modulus = self.t_gamma[0];
        let gamma_modulus = self.t_gamma[1];
        let minus_inv_q_mod_t = self.minus_inv_q_mod_t_gamma[0];
        let minus_inv_q_mod_gamma = self.minus_inv_q_mod_t_gamma[1];
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

        let conversion_scratch_len = self
            .converter_q_to_t_gamma
            .fast_convert_array_scratch_len(poly_length);
        self.converter_q_to_t_gamma
            .fast_convert_array_to_pair_iter(
                msg_mod_q.as_ref(),
                poly_length,
                &mut fast_convert_buffer[..conversion_scratch_len],
            )
            .zip(msg.iter_mut())
            .for_each(|((y_t, y_gamma), m)| {
                let y_t = t_modulus.reduce_mul(y_t, minus_inv_q_mod_t);
                let y_gamma = gamma_modulus.reduce_mul(y_gamma, minus_inv_q_mod_gamma);

                *m = if y_gamma > (gamma >> 1u32) {
                    t_modulus.reduce_add(y_t, t_modulus.reduce(gamma - y_gamma))
                } else {
                    t_modulus.reduce_sub(y_t, t_modulus.reduce(y_gamma))
                };
            });
        inv_gamma_mod_t.factor_mul_slice_assign(msg, t);
    }
}
