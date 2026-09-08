#![cfg(feature = "rns")]

use primus_encoding::BfvRnsCodec;
use primus_modulus::BarrettModulus;
use primus_poly::{CrtPolynomial, Polynomial};
use primus_rns::RNSBase;

#[test]
fn decodes_without_t_gamma_workspace() {
    type ValueT = u64;

    let moduli_value: [ValueT; 2] = [1125899906826241, 1125899906629633];
    let moduli = moduli_value.map(BarrettModulus::new);
    let base_q = RNSBase::new(&moduli).unwrap();
    let t = 12289;
    let gamma = 2305843009213554689;
    let codec = BfvRnsCodec::new(BarrettModulus::new(t), base_q, BarrettModulus::new(gamma));
    let poly_length = 32;
    let rns_poly_len = codec.moduli_count() * poly_length;
    let input_values: Vec<ValueT> = (0..poly_length)
        .map(|i| [0, 1, t / 2, t.div_ceil(2), t - 1][i % 5])
        .collect();
    let input = Polynomial::new(input_values.clone());
    let mut encoded: CrtPolynomial<Vec<ValueT>> = CrtPolynomial::<Vec<u64>>::zero(rns_poly_len);

    codec.encode_coeffs_to(
        &input,
        &mut encoded,
        primus_encoding::PlaintextEmbedding::Centered,
    );

    let mut fused_q = CrtPolynomial::new(encoded.into_owned());
    let mut fused_decoded: Polynomial<Vec<ValueT>> = Polynomial::<Vec<u64>>::zero(poly_length);
    let mut fused_fast_convert_buffer = vec![0; rns_poly_len];
    codec.decode_coeffs_to(
        &mut fused_q,
        &mut fused_decoded,
        &mut fused_fast_convert_buffer,
    );
    assert_eq!(fused_decoded.as_ref(), input_values);
}

#[test]
fn floor_scaling_and_noisy_decode_match_integer_oracle() {
    use primus_encoding::PlaintextEmbedding::{Centered, Unsigned};
    for (moduli, t, gamma) in [
        ([17u64, 19], 3, 65537),
        ([17, 19], 2, 65537),
        ([97, 193], 7, 65537),
    ] {
        let q = u128::from(moduli[0]) * u128::from(moduli[1]);
        let delta = q / u128::from(t);
        let base = RNSBase::new(&moduli.map(BarrettModulus::new)).unwrap();
        let codec = BfvRnsCodec::new(BarrettModulus::new(t), base, BarrettModulus::new(gamma));
        let input = Polynomial::new((0..t).collect::<Vec<_>>());
        let n = t as usize;
        for embedding in [Unsigned, Centered] {
            let expected: Vec<u64> = moduli
                .iter()
                .flat_map(|&qi| {
                    (0..t).map(move |m| {
                        let lift = if embedding == Centered && m >= t.div_ceil(2) {
                            i128::from(m) - i128::from(t)
                        } else {
                            i128::from(m)
                        };
                        (lift * delta as i128).rem_euclid(i128::from(qi)) as u64
                    })
                })
                .collect();
            let mut encoded = CrtPolynomial::<Vec<u64>>::zero(n * 2);
            codec.encode_coeffs_to(&input, &mut encoded, embedding);
            assert_eq!(encoded.as_ref(), expected);
            let mut acc = CrtPolynomial::new(
                moduli
                    .iter()
                    .flat_map(|&qi| vec![qi - 1; n])
                    .collect::<Vec<_>>(),
            );
            codec.add_encode_coeffs_assign(&input, &mut acc, embedding);
            for (i, &value) in acc.as_ref().iter().enumerate() {
                assert_eq!(value, (expected[i] + moduli[i / n] - 1) % moduli[i / n]);
            }
            for noise in [-1i128, 0, 1] {
                let mut phase = CrtPolynomial::new(
                    expected
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| {
                            (i128::from(x) + noise).rem_euclid(i128::from(moduli[i / n])) as u64
                        })
                        .collect::<Vec<_>>(),
                );
                let mut output = Polynomial::<Vec<u64>>::zero(n);
                codec.decode_coeffs_to(
                    &mut phase,
                    &mut output,
                    &mut vec![0; codec.decode_scratch_len(n)],
                );
                assert_eq!(
                    output.as_ref(),
                    input.as_ref(),
                    "t={t}, embedding={embedding:?}, noise={noise}"
                );
            }
        }
    }
}

#[test]
fn rejects_invalid_rns_boundaries_before_writing() {
    use primus_encoding::PlaintextEmbedding::{Centered, Unsigned};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let base = RNSBase::new(&[17u64, 19].map(BarrettModulus::new)).unwrap();
    let codec = BfvRnsCodec::new(BarrettModulus::new(3), base, BarrettModulus::new(65537));
    for embedding in [Unsigned, Centered] {
        for (input, len) in [(vec![0], 1), (vec![0], 3), (vec![0, 3], 4), (vec![0, 4], 4)] {
            let mut output = CrtPolynomial::new(vec![9; len]);
            assert!(
                catch_unwind(AssertUnwindSafe(|| codec.encode_coeffs_to(
                    &Polynomial::new(input.clone()),
                    &mut output,
                    embedding
                )))
                .is_err()
            );
            assert_eq!(output.as_ref(), vec![9; len]);
            assert!(
                catch_unwind(AssertUnwindSafe(|| codec.add_encode_coeffs_assign(
                    &Polynomial::new(input),
                    &mut output,
                    embedding
                )))
                .is_err()
            );
            assert_eq!(output.as_ref(), vec![9; len]);
        }
    }
    for (input_len, scratch_len) in [(1, 2), (3, 2), (2, 1), (2, 3)] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| codec.decode_coeffs_to(
                &mut CrtPolynomial::<Vec<u64>>::zero(input_len),
                &mut Polynomial::<Vec<u64>>::zero(1),
                &mut vec![0; scratch_len]
            )))
            .is_err()
        );
    }
}
