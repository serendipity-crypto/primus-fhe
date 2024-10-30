//! Interactive Proof Protocol used for Multilinear Sumcheck
// It is derived from https://github.com/arkworks-rs/sumcheck/blob/master/src/ml_sumcheck/protocol/mod.rs.

use algebra::{utils::Transcript, Field, ListOfProductsOfPolynomials, PolynomialInfo};
use prover::{ProverMsg, ProverState};
use serde::Serialize;
use std::marker::PhantomData;
use verifier::SubClaim;
pub mod prover;
pub mod verifier;

/// IP for MLSumcheck   
pub struct IPForMLSumcheck<F: Field> {
    _marker: PhantomData<F>,
}

/// Sumcheck for products of multilinear polynomial
pub struct MLSumcheck<F: Field>(PhantomData<F>);

/// proof generated by prover
pub type Proof<F> = Vec<ProverMsg<F>>;

/// This is a wrapper for prover claiming sumcheck protocol
pub struct SumcheckKit<F: Field> {
    /// claimed sum
    pub claimed_sum: F,
    /// poly info of the polynomial proved in the sumcheck
    pub info: PolynomialInfo,
    /// random point used to instantiate the sumcheck protocol
    pub u: Vec<F>,
    /// sumcheck proof
    pub proof: Proof<F>,
    /// random point returned from sumcheck protocol
    pub randomness: Vec<F>,
}

/// This is a wrapper for verifier checking the sumcheck protocol
pub struct ProofWrapper<F: Field> {
    /// claimed sum
    pub claimed_sum: F,
    /// poly info of the polynomial proved in the sumcheck
    pub info: PolynomialInfo,
    /// sumcheck proof
    pub proof: Proof<F>,
}

impl<F: Field> SumcheckKit<F> {
    /// Extract the proof wrapper used by verifier
    pub fn extract(&self) -> ProofWrapper<F> {
        ProofWrapper::<F> {
            claimed_sum: self.claimed_sum,
            info: self.info,
            proof: self.proof.clone(),
        }
    }
}

impl<F: Field + Serialize> MLSumcheck<F> {
    /// Extract sum from the proof
    pub fn extract_sum(proof: &Proof<F>) -> F {
        proof[0].evaluations[0] + proof[0].evaluations[1]
    }

    /// Generate proof of the sum of polynomial over {0, 1}^`num_vars`
    ///
    /// The polynomial is represented by a list of products of polynomials along with its coefficient that is meant to be added together.
    ///
    /// This data structure of the polynomial is a list of list of `(coefficient, DenseMultilinearExtension)`.
    /// * Number of products n = `polynomial.products.len()`,
    /// * Number of multiplicands of ith product m_i = `polynomial.products[i].1.len()`,
    /// * Coefficient of ith product c_i = `polynomial.products[i].0`
    ///
    /// The resulting polynomial is
    ///
    /// $$\sum_{i=0}^{n}C_i\cdot\prod_{j=0}^{m_i}P_{ij}$$
    pub fn prove(
        trans: &mut Transcript<F>,
        polynomial: &ListOfProductsOfPolynomials<F>,
    ) -> Result<(Proof<F>, ProverState<F>), crate::error::Error> {
        trans.append_message(b"polynomial info", &polynomial.info());
        println!("[sumcheck] The polynomial (degree = {}) to be proved consists of {} MLEs (#vars = {}) in the form of {} products.", polynomial.max_multiplicands, polynomial.flattened_ml_extensions.len(), polynomial.num_variables, polynomial.products.len());
        let mut prover_state = IPForMLSumcheck::prover_init(polynomial);
        let mut verifier_msg = None;
        let mut prover_msgs = Vec::with_capacity(polynomial.num_variables);
        for _ in 0..polynomial.num_variables {
            let prover_msg = IPForMLSumcheck::prove_round(&mut prover_state, &verifier_msg);
            trans.append_message(b"sumcheck msg", &prover_msg);
            prover_msgs.push(prover_msg);
            verifier_msg = Some(IPForMLSumcheck::sample_round(trans));
        }
        prover_state
            .randomness
            .push(verifier_msg.unwrap().randomness);
        Ok((prover_msgs, prover_state))
    }

    /// verify the proof using `polynomial_info` as the verifier key
    pub fn verify(
        trans: &mut Transcript<F>,
        polynomial_info: &PolynomialInfo,
        claimed_sum: F,
        proof: &Proof<F>,
    ) -> Result<SubClaim<F>, crate::Error> {
        trans.append_message(b"polynomial info", polynomial_info);
        let mut verifier_state = IPForMLSumcheck::verifier_init(polynomial_info);
        for i in 0..polynomial_info.num_variables {
            let prover_msg = proof.get(i).expect("proof is incomplete");
            trans.append_message(b"sumcheck msg", prover_msg);

            IPForMLSumcheck::verify_round(prover_msg, &mut verifier_state, trans);
        }

        IPForMLSumcheck::check_and_generate_subclaim(verifier_state, claimed_sum)
    }
}
