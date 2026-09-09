use primus_decompose::primitive::ApproxSignedBasis;
use primus_lwe::{
    LweKeySwitchingKey, LweParameters, LweSecretKey, LweSecretKeyRef, SecretKeyDistr,
};
use primus_modulus::{BarrettModulus, NativeModulus};
use primus_reduce::RingContext;
use rand::{Rng, SeedableRng, rngs::StdRng};

#[test]
fn key_switches_between_lwe_secret_keys() {
    let modulus = NativeModulus::<u32>::new();
    let input_parameters = LweParameters::new(8, 4u32, modulus, SecretKeyDistr::UniformBinary, 3.2);
    let output_parameters =
        LweParameters::new(5, 4u32, modulus, SecretKeyDistr::UniformBinary, 3.2);
    let basis = ApproxSignedBasis::new(None, 4, None);
    let mut rng = StdRng::seed_from_u64(0x1_ee05);
    let input_secret_key = LweSecretKey::generate(&input_parameters, &mut rng);
    let output_secret_key = LweSecretKey::generate(&output_parameters, &mut rng);
    let key_switching_key = LweKeySwitchingKey::generate(
        input_secret_key.as_view(),
        &output_secret_key,
        &output_parameters,
        basis,
        &mut rng,
    );

    for message in 0u32..4 {
        let input = input_secret_key.encrypt(message, &input_parameters, &mut rng);
        let output = key_switching_key.key_switch(&input, modulus);
        assert_eq!(
            output_secret_key.decrypt::<_, u32>(&output, &output_parameters),
            message
        );
    }
}

#[test]
fn key_switches_with_base_four_signed_digits() {
    check_base_four_signed_digits(NativeModulus::new());
    check_base_four_signed_digits(BarrettModulus::new(132_120_577));
}

fn check_base_four_signed_digits<M: RingContext<u32>>(modulus: M) {
    let q = modulus.explicit_value().map_or(1i64 << 32, i64::from);
    let input_parameters =
        LweParameters::new(8, 4u32, modulus, SecretKeyDistr::UniformTernary, 0.7);
    let output_parameters =
        LweParameters::new(5, 4u32, modulus, SecretKeyDistr::UniformTernary, 0.7);
    let basis = ApproxSignedBasis::new(
        modulus.explicit_value(),
        2,
        modulus.explicit_value().map(|_| 13),
    );
    let mut rng = StdRng::seed_from_u64(0x1_ee06);
    let signed_key = [-1, 0, 1, -1, 1, 0, -1, 1];
    let input_secret_key = LweSecretKey::new(
        signed_key
            .iter()
            .map(|&value: &i32| i64::from(value).rem_euclid(q) as u32)
            .collect(),
        SecretKeyDistr::UniformTernary,
    );
    let signed_output_key = [1, -1, 0, 1, -1];
    let output_secret_key = LweSecretKey::new(
        signed_output_key
            .iter()
            .map(|&value: &i32| i64::from(value).rem_euclid(q) as u32)
            .collect(),
        SecretKeyDistr::UniformTernary,
    );
    let mut encoded_rng = StdRng::seed_from_u64(0x1_ee08);
    let key_switching_key = LweKeySwitchingKey::generate(
        input_secret_key.as_view(),
        &output_secret_key,
        &output_parameters,
        basis.clone(),
        &mut encoded_rng,
    );
    let expected_next_random = encoded_rng.next_u64();
    let mut signed_rng = StdRng::seed_from_u64(0x1_ee08);
    let signed_key_switching_key = LweKeySwitchingKey::generate(
        LweSecretKeyRef::Signed(&signed_key),
        &output_secret_key,
        &output_parameters,
        basis,
        &mut signed_rng,
    );
    // Input representation preserves entry layout and RNG consumption.
    assert_eq!(
        key_switching_key.as_slice(),
        signed_key_switching_key.as_slice()
    );
    assert_eq!(signed_rng.next_u64(), expected_next_random);

    let mut output = primus_lwe::LweCiphertext::new(vec![17u32; output_parameters.dimension() + 1]);
    *output.b_mut() = (q / 2) as u32;
    for message in 0u32..4 {
        let input = input_secret_key.encrypt(message, &input_parameters, &mut rng);
        key_switching_key.key_switch_to(&input, &mut output, modulus);
        assert_eq!(
            output_secret_key.decrypt::<_, u32>(&output, &output_parameters),
            message
        );
    }
}

#[test]
fn key_switch_validates_before_writing() {
    use primus_lwe::LweCiphertext;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let modulus = BarrettModulus::new(97u32);
    let output_parameters = LweParameters::new(2, 4, modulus, SecretKeyDistr::UniformBinary, 0.7);
    let basis = ApproxSignedBasis::new(Some(97), 2, None);
    let output_key = LweSecretKey::new(vec![1, 0], SecretKeyDistr::UniformBinary);
    let mut rng = StdRng::seed_from_u64(0x1_ee09);
    let key = LweKeySwitchingKey::generate(
        LweSecretKeyRef::Encoded(&[1, 0, 1]),
        &output_key,
        &output_parameters,
        basis,
        &mut rng,
    );
    // Missing bodies, wrong dimensions and wrong modulus must all leave output intact.
    for (input_len, output_len, q) in [(0, 3, 97), (4, 0, 97), (3, 3, 97), (4, 2, 97), (4, 3, 101)]
    {
        let input = LweCiphertext::new(vec![7u32; input_len]);
        let mut storage = vec![11u32; output_len];
        let expected = storage.clone();
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                key.key_switch_to(
                    &input,
                    &mut primus_lattice::lwe::Lwe::new(storage.as_mut_slice()),
                    BarrettModulus::new(q),
                );
            }))
            .is_err()
        );
        assert_eq!(storage, expected);
    }
}

#[test]
fn generation_validates_before_sampling() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let parameters = LweParameters::new(
        2,
        4u32,
        BarrettModulus::new(97),
        SecretKeyDistr::UniformBinary,
        0.7,
    );
    for (input_key, output_dimension, basis_modulus) in [
        (LweSecretKeyRef::Encoded(&[]), 2, 97),
        (LweSecretKeyRef::Signed(&[]), 2, 97),
        (LweSecretKeyRef::Encoded(&[1]), 0, 97),
        (LweSecretKeyRef::Encoded(&[1]), 3, 97),
        (LweSecretKeyRef::Encoded(&[1]), 2, 101),
    ] {
        let output_key =
            LweSecretKey::new(vec![1; output_dimension], SecretKeyDistr::UniformBinary);
        let basis = ApproxSignedBasis::new(Some(basis_modulus), 2, None);
        let mut rng = StdRng::seed_from_u64(0x1_ee0a);
        let mut expected_rng = StdRng::seed_from_u64(0x1_ee0a);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                LweKeySwitchingKey::generate(input_key, &output_key, &parameters, basis, &mut rng)
            }))
            .is_err()
        );
        assert_eq!(rng.next_u64(), expected_rng.next_u64());
    }
}
