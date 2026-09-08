use primus_data::{Data, DataMut};
use primus_factor::{FactorMul, FactorSliceOps, ShoupFactor};
use primus_integer::{BigUint, DivRemScalar, FheUint, multiply_many_values};
use primus_poly::{CrtPolynomial, Polynomial};
use primus_reduce::FieldContext;

use crate::PlaintextEmbedding;
use primus_rns::{BaseConverter, RNSBase, ResidueFactors, Residues};

/// BFV-style RNS coefficient codec.
///
/// Encodes `m ∈ Z_t` as RNS residues of `m · Δ mod Q` where `Δ = floor(Q/t)`,
/// and decodes via the HPS / Bajard et al. fast base extension to `{t, γ}`.
#[derive(Clone)]
pub struct BfvRnsCodec<T, M>
where
    T: FheUint,
    M: FieldContext<T>,
{
    // Plaintext / ciphertext bases
    t: T,
    base_q: RNSBase<T, M>,
    moduli_values: Vec<T>,

    // For encoding: Δ = floor(Q / t) as BigUint, pre-decomposed into Shoup factors mod each q_i
    delta: BigUint<Vec<T>>,
    delta_factor_mod_q: ResidueFactors<Vec<ShoupFactor<T>>>,

    // For decoding: HPS γ-trick, produces m mod t
    gamma: T,
    base_t_gamma: RNSBase<T, M>,                               // {t, γ}
    t_gamma_factor_mod_q: ResidueFactors<Vec<ShoupFactor<T>>>, // [(t·γ) mod q_i]
    minus_inv_q_mod_t_gamma: Residues<Vec<T>>,                 // [(−Q^{-1}) mod m_j], m_j ∈ {t, γ}
    inv_gamma_mod_t: ShoupFactor<T>,                           // (γ^{-1}) mod t
    converter_q_to_t_gamma: BaseConverter<T, M>,
}

impl<T, M> BfvRnsCodec<T, M>
where
    T: FheUint,
    M: FieldContext<T>,
{
    /// Builds a BFV-style RNS codec.
    ///
    /// Establishes conservative noiseless-recovery bounds:
    /// `Q > 4*(Q % t)*(t-1)` and `gamma > 4*k` for `k` ciphertext moduli.
    /// These are sufficient bounds, not the full set of usable BFV parameters.
    ///
    /// # Panics
    /// Panics unless `t >= 2`, `t < gamma <= T::MAX/2`, each `t < q_i <= T::MAX/2`,
    /// and `t`, `gamma`, and each ciphertext modulus are pairwise coprime
    /// as required by the conversions. Also panics if the recovery bounds fail.
    #[must_use]
    pub fn new(t_modulus: M, base_q: RNSBase<T, M>, gamma_modulus: M) -> Self {
        let t = t_modulus.value();
        let gamma = gamma_modulus.value();

        let moduli_values: Vec<T> = base_q.moduli().iter().map(|m| m.value()).collect();

        Self::validate_moduli(t, &moduli_values, gamma);

        let cipher_modulus = base_q.moduli_product();

        let mut delta = BigUint(vec![T::ZERO; cipher_modulus.len()]);

        let rem = DivRemScalar::div_rem_scalar(cipher_modulus.digits(), t, delta.digits_mut());
        // Bound the unsigned encoding drift by 1/4 of a decoding cell.
        // Combined with k/gamma < 1/4 this guarantees noiseless recovery.
        let mut drift_bound = multiply_many_values(&[(T::TWO + T::TWO), rem, t - T::ONE]);
        assert!(
            {
                let fits = drift_bound
                    .digits()
                    .get(cipher_modulus.len()..)
                    .is_none_or(|high| high.iter().all(|&x| x == T::ZERO));
                drift_bound.0.resize(cipher_modulus.len(), T::ZERO);
                fits && cipher_modulus.cmp(&drift_bound).is_gt()
            },
            "BFV modulus product too small for the plaintext scaling error"
        );

        let delta_factor_mod_q = base_q.decompose_factors(delta.view());

        let t_gamma = [t_modulus, gamma_modulus];
        let base_t_gamma = RNSBase::new(&t_gamma).unwrap();
        let q_mod_t_gamma = base_t_gamma.decompose(cipher_modulus.view());
        let minus_inv_q_mod_t_gamma = Residues(
            q_mod_t_gamma
                .iter()
                .zip(&t_gamma)
                .map(|(&x, modulus)| modulus.reduce_neg(modulus.reduce_inv(x)))
                .collect::<Vec<T>>(),
        );
        let inv_gamma_mod_t = ShoupFactor::new(t_modulus.reduce_inv(t_modulus.reduce(gamma)), t);
        let t_gamma_value = multiply_many_values(&[t, gamma]);
        let t_gamma_factor_mod_q = base_q.decompose_factors(t_gamma_value.view());

        let converter_q_to_t_gamma = BaseConverter::new(&base_q, &base_t_gamma);

        Self {
            t,
            base_q,
            moduli_values,
            delta,
            delta_factor_mod_q,
            gamma,
            base_t_gamma,
            t_gamma_factor_mod_q,
            minus_inv_q_mod_t_gamma,
            inv_gamma_mod_t,
            converter_q_to_t_gamma,
        }
    }

    fn validate_moduli(plain_modulus_value: T, cipher_moduli_value: &[T], gamma: T) {
        let limbs = T::try_from(cipher_moduli_value.len())
            .ok()
            .and_then(|k| k.checked_mul(T::TWO + T::TWO))
            .expect("RNS modulus count is too large");
        assert!(
            !cipher_moduli_value.is_empty(),
            "RNS basis must not be empty"
        );
        assert!(
            gamma > limbs,
            "gamma must exceed four times the RNS modulus count"
        );
        assert!(
            plain_modulus_value >= T::TWO,
            "plain modulus must be at least 2"
        );
        assert!(
            gamma > plain_modulus_value,
            "gamma modulus must be greater than the plain modulus for HPS decoding"
        );
        assert!(
            plain_modulus_value.is_coprime(gamma),
            "plain modulus and gamma modulus must be coprime"
        );
        assert!(
            gamma <= T::MAX / T::TWO,
            "gamma too large for Shoup arithmetic"
        );

        for &qi in cipher_moduli_value {
            assert!(
                qi <= T::MAX / T::TWO,
                "cipher modulus too large for Shoup arithmetic"
            );
            assert!(
                qi.is_coprime(plain_modulus_value),
                "cipher moduli must be coprime with the plain modulus"
            );
            assert!(
                qi > plain_modulus_value,
                "each RNS ciphertext modulus must be greater than the plain modulus for centered coefficient lifting"
            );
            assert!(
                qi.is_coprime(gamma),
                "cipher moduli must be coprime with the gamma modulus"
            );
        }
    }

    /// Returns the plaintext modulus `t`.
    #[must_use]
    pub fn t(&self) -> T {
        self.t
    }

    /// Returns the ordered ciphertext RNS basis Q.
    #[must_use]
    pub fn base_q(&self) -> &RNSBase<T, M> {
        &self.base_q
    }

    /// Returns the number of ciphertext moduli.
    #[must_use]
    pub fn moduli_count(&self) -> usize {
        self.base_q.moduli_count()
    }

    /// Returns the ordered numeric values of the ciphertext moduli.
    #[must_use]
    pub fn moduli_values(&self) -> &[T] {
        &self.moduli_values
    }

    /// Returns the Shoup factors for `floor(Q / t)` in every Q limb.
    #[must_use]
    pub fn delta_factor_mod_q(&self) -> ResidueFactors<&[ShoupFactor<T>]> {
        self.delta_factor_mod_q.view()
    }

    /// Returns `floor(Q / t)` as a multi-limb integer.
    #[must_use]
    pub fn delta(&self) -> BigUint<&[T]> {
        self.delta.view()
    }

    /// Validates a complete coefficient-domain plaintext/output boundary.
    fn validate_plaintext(&self, input: &[T], output_len: usize) {
        assert_eq!(
            output_len,
            self.decode_scratch_len(input.len()),
            "RNS output length mismatch"
        );
        assert!(
            input.iter().all(|&m| m < self.t),
            "message must be less than plaintext modulus"
        );
    }

    /// Encodes coefficients as `lift(input) * floor(Q/t)` in the ordered CRT basis.
    ///
    /// Centered embedding uses `[-floor(t/2), ceil(t/2))`, including `1 -> -1`
    /// for `t = 2`. Output is in the coefficient domain, ready for a separate NTT.
    ///
    /// # Panics
    /// Panics for messages outside `[0,t)` or an output length other than
    /// `input.len() * moduli_count()`.
    pub fn encode_coeffs_to<A, B>(
        &self,
        input: &Polynomial<A>,
        output: &mut CrtPolynomial<B>,
        embedding: PlaintextEmbedding,
    ) where
        A: Data<Elem = T>,
        B: DataMut<Elem = T>,
    {
        self.validate_plaintext(input.as_ref(), output.as_ref().len());
        if input.as_ref().is_empty() {
            return;
        }
        let half = (self.t >> 1u32) + (self.t & T::ONE);
        for ((chunk, &q), &factor) in output
            .as_mut()
            .chunks_exact_mut(input.as_ref().len())
            .zip(&self.moduli_values)
            .zip(self.delta_factor_mod_q.iter())
        {
            match embedding {
                PlaintextEmbedding::Unsigned => {
                    factor.factor_mul_slice_to(input.as_ref(), chunk, q)
                }
                PlaintextEmbedding::Centered => {
                    for (&m, out) in input.as_ref().iter().zip(chunk) {
                        *out = if m < half {
                            factor.factor_mul_modulo(m, q)
                        } else {
                            let value = factor.factor_mul_modulo(self.t - m, q);
                            if value == T::ZERO { value } else { q - value }
                        };
                    }
                }
            }
        }
    }

    /// Adds scaled plaintext coefficients to a canonical CRT accumulator.
    ///
    /// Uses the same embedding as [`Self::encode_coeffs_to`]. The accumulator
    /// is not cleared; its residues must be canonical under their respective moduli.
    ///
    /// # Panics
    /// Panics for messages outside `[0,t)` or an accumulator length other than
    /// `input.len() * moduli_count()`.
    pub fn add_encode_coeffs_assign<A, B>(
        &self,
        input: &Polynomial<A>,
        acc: &mut CrtPolynomial<B>,
        embedding: PlaintextEmbedding,
    ) where
        A: Data<Elem = T>,
        B: DataMut<Elem = T>,
    {
        self.validate_plaintext(input.as_ref(), acc.as_ref().len());
        if input.as_ref().is_empty() {
            return;
        }
        let half = (self.t >> 1u32) + (self.t & T::ONE);
        for (((chunk, &q), modulus), &factor) in acc
            .as_mut()
            .chunks_exact_mut(input.as_ref().len())
            .zip(&self.moduli_values)
            .zip(self.base_q.moduli())
            .zip(self.delta_factor_mod_q.iter())
        {
            match embedding {
                PlaintextEmbedding::Unsigned => {
                    factor.add_factor_mul_slice_assign(chunk, input.as_ref(), q)
                }
                PlaintextEmbedding::Centered => {
                    for (&m, out) in input.as_ref().iter().zip(chunk) {
                        if m < half {
                            modulus.reduce_add_assign(out, factor.factor_mul_modulo(m, q));
                        } else {
                            modulus.reduce_sub_assign(out, factor.factor_mul_modulo(self.t - m, q));
                        }
                    }
                }
            }
        }
    }

    /// Exact scratch length for decoding `poly_length` coefficients.
    ///
    /// # Panics
    /// Panics if the RNS shape cannot be represented by `usize`.
    #[must_use]
    pub fn decode_scratch_len(&self, poly_length: usize) -> usize {
        self.moduli_count()
            .checked_mul(poly_length)
            .expect("RNS polynomial length overflow")
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
        let rns_poly_len = self.decode_scratch_len(poly_length);
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
        let t_modulus = self.base_t_gamma.moduli()[0];
        let gamma_modulus = self.base_t_gamma.moduli()[1];
        let minus_inv_q_mod_t = self.minus_inv_q_mod_t_gamma.as_ref()[0];
        let minus_inv_q_mod_gamma = self.minus_inv_q_mod_t_gamma.as_ref()[1];
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
