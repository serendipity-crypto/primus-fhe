//! BFV coefficient scaling and decoding over an RNS ciphertext basis.

mod decode;
mod encode;

use primus_factor::{FactorMul, ShoupFactor};
use primus_integer::{BigUint, DivRemScalar, FheUint, multiply_many_values};
use primus_reduce::FieldContext;
use primus_rns::{BaseConverter, RNSBase, ResidueFactors};

/// BFV-style RNS coefficient codec.
///
/// Encodes `m ∈ Z_t` as canonical RNS residues of `lift(m) * delta mod Q`,
/// where `delta = floor(Q/t)`. Decoding uses fast base conversion to `{t, gamma}`
/// and an auxiliary-modulus correction to recover canonical residues modulo `t`.
/// CRT buffers use coefficient-domain, modulus-major layout in [`Self::base_q`] order.
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

    // For decoding: auxiliary-modulus correction, producing m mod t.
    gamma: T,
    t_modulus: M,
    t_gamma_factor_mod_q: ResidueFactors<Vec<ShoupFactor<T>>>, // [(t·γ) mod q_i]
    // The t factor includes the final division by γ; the γ factor recovers
    // the centered correction before that division.
    decode_factor_mod_t_gamma: [ShoupFactor<T>; 2], // {−(Qγ)^−1 mod t, −Q^−1 mod γ}
    inv_gamma_mod_t: ShoupFactor<T>,                // γ^−1 mod t
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
    /// The supplied [`RNSBase`] already guarantees a nonempty, pairwise-coprime basis.
    ///
    /// # Panics
    ///
    /// Panics unless `t >= 2`, `t < gamma <= T::MAX/2`, each `t < q_i <= T::MAX/2`,
    /// `gcd(t,gamma) = 1`, and each `q_i` is coprime with both `t` and `gamma`.
    /// Also panics if either recovery bound above fails.
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
        let minus_inv_q_mod_t_gamma: [T; 2] = core::array::from_fn(|i| {
            t_gamma[i].reduce_neg(t_gamma[i].reduce_inv(q_mod_t_gamma.as_ref()[i]))
        });
        let inv_gamma_mod_t = ShoupFactor::new(t_modulus.reduce_inv(t_modulus.reduce(gamma)), t);
        let decode_factor_mod_t_gamma = [
            ShoupFactor::new(
                inv_gamma_mod_t.factor_mul_modulo(minus_inv_q_mod_t_gamma[0], t),
                t,
            ),
            ShoupFactor::new(minus_inv_q_mod_t_gamma[1], gamma),
        ];
        let t_gamma_value = multiply_many_values(&[t, gamma]);
        let t_gamma_factor_mod_q = base_q.decompose_factors(t_gamma_value.view());

        let converter_q_to_t_gamma = BaseConverter::from_owned_bases(base_q.clone(), base_t_gamma);

        Self {
            t,
            base_q,
            moduli_values,
            delta,
            delta_factor_mod_q,
            gamma,
            t_modulus,
            t_gamma_factor_mod_q,
            decode_factor_mod_t_gamma,
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

    /// Returns the ordered ciphertext RNS basis whose product is `Q`.
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

    /// Returns the Shoup factors for `floor(Q/t) mod q_i` in [`Self::base_q`] order.
    #[must_use]
    pub fn delta_factor_mod_q(&self) -> ResidueFactors<&[ShoupFactor<T>]> {
        self.delta_factor_mod_q.view()
    }

    /// Returns `floor(Q/t)` as a borrowed integer with little-endian limbs.
    #[must_use]
    pub fn delta(&self) -> BigUint<&[T]> {
        self.delta.view()
    }

    /// Shared RNS shape for coefficient buffers and decoding workspace.
    ///
    /// # Panics
    ///
    /// Panics if `poly_length * moduli_count()` overflows `usize`.
    fn rns_poly_len(&self, poly_length: usize) -> usize {
        self.moduli_count()
            .checked_mul(poly_length)
            .expect("RNS polynomial length overflow")
    }
}
