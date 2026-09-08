#![cfg(feature = "rns")]

use primus_encoding::{
    BfvRnsCodec,
    PlaintextEmbedding::{Centered, Unsigned},
};
use primus_modulus::BarrettModulus;
use primus_poly::{CrtPolynomial, Polynomial};
use primus_rns::RNSBase;

#[test]
fn floor_scaling_and_noisy_decode_match_integer_oracle() {
    for (moduli, t, gamma) in [
        (&[97u64][..], 3, 65537),
        (&[17u64, 19][..], 3, 65537),
        (&[17, 19][..], 2, 65537),
        (&[97, 193][..], 7, 65537),
        (
            &[1125899906826241, 1125899906629633][..],
            12289,
            2305843009213554689,
        ), // Q exceeds u64; the independent oracle still fits i128.
    ] {
        let q: u128 = moduli.iter().map(|&qi| u128::from(qi)).product();
        let delta = q / u128::from(t);
        let base = RNSBase::new(
            &moduli
                .iter()
                .copied()
                .map(BarrettModulus::new)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let codec = BfvRnsCodec::new(BarrettModulus::new(t), base, BarrettModulus::new(gamma));
        let input = Polynomial::new(if t <= 7 {
            (0..t).collect::<Vec<_>>()
        } else {
            // Include full SIMD vectors and a scalar tail when SIMD is enabled.
            (0..17)
                .map(|i| [0, 1, t / 2, t.div_ceil(2), t - 1][i % 5])
                .collect()
        });
        let messages = input.as_ref();
        let n = messages.len();
        assert_eq!(
            codec.decode_scratch_len(n),
            if moduli.len() == 1 {
                0
            } else {
                n * moduli.len()
            }
        );
        let mut scratch = vec![0; codec.decode_scratch_len(n)];
        let mut output = Polynomial::<Vec<u64>>::zero(n);
        for embedding in [Unsigned, Centered] {
            let expected: Vec<u64> = moduli
                .iter()
                .flat_map(|&qi| {
                    messages.iter().map(move |&m| {
                        let lift = if embedding == Centered && m >= t.div_ceil(2) {
                            i128::from(m) - i128::from(t)
                        } else {
                            i128::from(m)
                        };
                        (lift * delta as i128).rem_euclid(i128::from(qi)) as u64
                    })
                })
                .collect();
            let mut encoded = CrtPolynomial::<Vec<u64>>::zero(n * moduli.len());
            codec.encode_coeffs_to(&input, &mut encoded, embedding);
            assert_eq!(
                encoded.as_ref(),
                expected,
                "t={t}, Q={q}, embedding={embedding:?}"
            );
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
                codec.decode_coeffs_to(&mut phase, &mut output, &mut scratch);
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
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let base = RNSBase::new(&[17u64, 19].map(BarrettModulus::new)).unwrap();
    let codec = BfvRnsCodec::new(BarrettModulus::new(3), base, BarrettModulus::new(65537));
    for (input, len) in [(&[0][..], 1), (&[0][..], 3), (&[0, 3][..], 4)] {
        let mut output = CrtPolynomial::new(vec![9; len]);
        assert!(
            catch_unwind(AssertUnwindSafe(|| codec.encode_coeffs_to(
                &Polynomial::new(input),
                &mut output,
                Centered
            )))
            .is_err()
        );
        assert_eq!(output.as_ref(), vec![9; len]);
        assert!(
            catch_unwind(AssertUnwindSafe(|| codec.add_encode_coeffs_assign(
                &Polynomial::new(input),
                &mut output,
                Centered
            )))
            .is_err()
        );
        assert_eq!(output.as_ref(), vec![9; len]);
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
