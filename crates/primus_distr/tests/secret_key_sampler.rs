use primus_distr::{EncodedSecretKeySampler, SecretKeyDistr, SignedSecretKeySampler};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[test]
fn representations_preserve_distribution_and_logical_weights() {
    for distr in [
        SecretKeyDistr::UniformBinary,
        SecretKeyDistr::binary(0.3),
        SecretKeyDistr::SparseTernary,
        SecretKeyDistr::UniformTernary,
        SecretKeyDistr::ternary(0.2, 0.4),
        SecretKeyDistr::fixed_hamming_weight_binary(67, 7),
        SecretKeyDistr::fixed_hamming_weight_binary(67, 33),
        SecretKeyDistr::fixed_hamming_weight_binary(67, 60),
        SecretKeyDistr::fixed_hamming_weight_ternary(67, 13),
        SecretKeyDistr::fixed_hamming_weight_ternary(67, 67),
        SecretKeyDistr::fixed_composition_ternary(67, 5, 8),
        SecretKeyDistr::fixed_composition_ternary(67, 22, 22),
        SecretKeyDistr::fixed_composition_ternary(67, 60, 4),
        SecretKeyDistr::fixed_composition_ternary(67, 4, 60),
        SecretKeyDistr::gaussian(3.2),
        SecretKeyDistr::gaussian(30.0),
    ] {
        let signed = SignedSecretKeySampler::<i32>::new(distr);
        let expected_bound = match distr {
            SecretKeyDistr::Gaussian {
                standard_deviation: 3.2,
            } => 38,
            SecretKeyDistr::Gaussian { .. } => 360,
            _ => 1,
        };
        assert_eq!(signed.maximum_magnitude(), expected_bound);
        for modulus in [1_009i64, 1i64 << 32] {
            let encoded = EncodedSecretKeySampler::<u32>::new(distr, (modulus - 1) as u32);
            let mut signed_rng = StdRng::seed_from_u64(201);
            let mut encoded_rng = StdRng::seed_from_u64(201);
            let input = signed.sample(67, &mut signed_rng);
            let mut output = [u32::MAX; 67];
            encoded.sample_to(&mut output, &mut encoded_rng);
            let expected: Vec<u32> = input
                .iter()
                .map(|&v| i64::from(v).rem_euclid(modulus) as u32)
                .collect();
            assert_eq!(output.as_slice(), expected, "{distr:?}");
            assert_eq!(signed_rng.next_u64(), encoded_rng.next_u64());
            match distr {
                SecretKeyDistr::FixedHammingWeightBinary { hamming_weight } => {
                    assert_eq!(input.iter().filter(|&&v| v == 1).count(), hamming_weight);
                    assert!(input.iter().all(|&v| v == 0 || v == 1));
                }
                SecretKeyDistr::FixedCompositionTernary {
                    negative_one_weight,
                    one_weight,
                } => {
                    assert_eq!(
                        input.iter().filter(|&&v| v == -1).count(),
                        negative_one_weight
                    );
                    assert_eq!(input.iter().filter(|&&v| v == 1).count(), one_weight);
                    assert!(input.iter().all(|&v| (-1..=1).contains(&v)));
                }
                SecretKeyDistr::FixedHammingWeightTernary { hamming_weight } => {
                    assert_eq!(input.iter().filter(|&&v| v != 0).count(), hamming_weight);
                    assert!(input.iter().all(|&v| (-1..=1).contains(&v)));
                }
                _ => {}
            }
        }
    }
}

#[test]
fn invalid_weight_is_rejected_before_sampling_or_writing() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    for distr in [
        SecretKeyDistr::fixed_hamming_weight_binary(7, 7),
        SecretKeyDistr::fixed_hamming_weight_ternary(7, 7),
        SecretKeyDistr::fixed_composition_ternary(7, 3, 4),
        SecretKeyDistr::FixedCompositionTernary {
            negative_one_weight: usize::MAX,
            one_weight: 1,
        },
    ] {
        let sampler = SignedSecretKeySampler::<i32>::new(distr);
        let mut output = [19; 6];
        let mut rng = StdRng::seed_from_u64(202);
        let mut expected = StdRng::seed_from_u64(202);
        assert!(
            catch_unwind(AssertUnwindSafe(|| sampler.sample_to(&mut output, &mut rng))).is_err()
        );
        assert_eq!(output, [19; 6]);
        assert_eq!(rng.next_u64(), expected.next_u64());
    }
}

#[test]
fn sampler_rejects_invalid_probabilities_in_raw_variants() {
    for distr in [
        SecretKeyDistr::Binary {
            one_probability: f64::NAN,
        },
        SecretKeyDistr::Ternary {
            negative_one_probability: -0.5,
            one_probability: 0.5,
        },
    ] {
        assert!(std::panic::catch_unwind(|| SignedSecretKeySampler::<i32>::new(distr)).is_err());
    }
}
