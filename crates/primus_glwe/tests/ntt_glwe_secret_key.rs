use primus_encoding::PlaintextEmbedding;
use primus_glwe::{
    GlweParameters, GlweSecretKey, NttGlweCiphertext, NttGlweSecretKey, SecretKeyDistr,
};
use primus_integer::FheUint;
use primus_modulus::BarrettModulus;
use primus_ntt::{NttTable, PrimitiveRoot, UintNttTable};
use primus_poly::{Polynomial, PolynomialOwned};
use primus_reduce::ReduceAdd;
use rand::{Rng, SeedableRng, rngs::StdRng};

const DIMENSION: usize = 2;
const POLY_LENGTH: usize = 256;
const PLAIN_MODULUS: usize = 16;

fn assert_roundtrip<T>(cipher_modulus: T)
where
    T: FheUint + PrimitiveRoot,
{
    let modulus = BarrettModulus::new(cipher_modulus);
    let ntt = UintNttTable::new(POLY_LENGTH.trailing_zeros(), modulus).unwrap();
    let mut rng = StdRng::seed_from_u64(42);
    let messages: Vec<T> = (0..POLY_LENGTH)
        .map(|index| T::try_from(index % PLAIN_MODULUS).unwrap())
        .collect();
    let message = Polynomial::new(messages.clone());

    for secret_key_distr in [
        SecretKeyDistr::UniformBinary,
        SecretKeyDistr::SparseTernary,
        SecretKeyDistr::gaussian(3.2),
    ] {
        let params = GlweParameters::new(
            DIMENSION,
            POLY_LENGTH,
            T::try_from(PLAIN_MODULUS).unwrap(),
            modulus,
            secret_key_distr,
            0.7,
        );
        let coeff_secret_key =
            GlweSecretKey::generate(params.size(), params.secret_key_sampler(), &mut rng);
        let secret_key = NttGlweSecretKey::from_coeff_secret_key(&coeff_secret_key, &ntt);
        let mut cipher = secret_key.encrypt(&message, &params, &ntt, &mut rng);
        assert_eq!(
            secret_key.decrypt(&cipher, &params, &ntt).as_ref(),
            messages
        );

        secret_key.encrypt_centered_to(&message, &mut cipher, &params, &ntt, &mut rng);
        assert_eq!(
            secret_key.decrypt(&cipher, &params, &ntt).as_ref(),
            messages
        );

        secret_key.encrypt_zeros_to(&mut cipher, &params, &ntt, &mut rng);
        assert_eq!(
            secret_key.decrypt(&cipher, &params, &ntt).as_ref(),
            vec![T::ZERO; POLY_LENGTH]
        );

        let mut encoded = vec![T::ZERO; POLY_LENGTH];
        params.plaintext_codec().add_encode_slice_assign(
            &mut encoded,
            &messages,
            PlaintextEmbedding::Unsigned,
        );
        secret_key.encrypt_encoded_to(
            &Polynomial::new(encoded),
            &mut cipher,
            &params,
            &ntt,
            &mut rng,
        );

        let mut reused_output = PolynomialOwned::new(vec![T::MAX; POLY_LENGTH]);
        secret_key.decrypt_to(&cipher, &mut reused_output, &params, &ntt);
        assert_eq!(reused_output.as_ref(), messages);
    }
}

#[test]
fn ntt_glwe_secret_key_roundtrip_u32() {
    assert_roundtrip(132_120_577u32);
}

#[test]
fn ntt_glwe_secret_key_roundtrip_u64() {
    assert_roundtrip(1_125_899_906_826_241u64);
}

#[test]
fn direct_generation_matches_coefficient_conversion() {
    fn check<T: FheUint + PrimitiveRoot>(cipher_modulus: T) {
        let modulus = BarrettModulus::new(cipher_modulus);
        let ntt = UintNttTable::new(POLY_LENGTH.trailing_zeros(), modulus).unwrap();
        let key_len = DIMENSION * POLY_LENGTH;
        for distr in [
            SecretKeyDistr::UniformBinary,
            SecretKeyDistr::binary(0.3),
            SecretKeyDistr::SparseTernary,
            SecretKeyDistr::UniformTernary,
            SecretKeyDistr::ternary(0.2, 0.4),
            SecretKeyDistr::fixed_hamming_weight_binary(key_len, POLY_LENGTH + 7),
            SecretKeyDistr::fixed_hamming_weight_ternary(key_len, POLY_LENGTH + 7),
            SecretKeyDistr::fixed_composition_ternary(key_len, 17, POLY_LENGTH + 7),
            SecretKeyDistr::gaussian(3.2),
            SecretKeyDistr::gaussian(30.0),
        ] {
            let params = GlweParameters::new(
                DIMENSION,
                POLY_LENGTH,
                T::try_from(PLAIN_MODULUS).unwrap(),
                modulus,
                distr,
                0.7,
            );
            let mut coeff_rng = StdRng::seed_from_u64(0x4e54_5453_414d_504c);
            let mut direct_rng = StdRng::seed_from_u64(0x4e54_5453_414d_504c);
            let coeff_key =
                GlweSecretKey::generate(params.size(), params.secret_key_sampler(), &mut coeff_rng);
            let expected = NttGlweSecretKey::from_coeff_secret_key(&coeff_key, &ntt);
            let actual = NttGlweSecretKey::generate(&params, &ntt, &mut direct_rng);
            assert_eq!(actual.glwe_size(), expected.glwe_size());
            assert_eq!(actual.distr(), expected.distr());
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert_eq!(actual.as_ref(), expected.as_ref(), "{distr:?}");
            }
            assert_eq!(direct_rng.next_u64(), coeff_rng.next_u64(), "{distr:?}");
        }
    }

    check(132_120_577u32);
    check(1_125_899_906_826_241u64);
}

#[test]
fn noise_diagnostics_report_signed_phase_distance() {
    let modulus = BarrettModulus::new(1_125_899_906_826_241u64);
    let params = GlweParameters::new(
        DIMENSION,
        POLY_LENGTH,
        PLAIN_MODULUS as u64,
        modulus,
        SecretKeyDistr::UniformBinary,
        0.7,
    );
    let table = UintNttTable::new(POLY_LENGTH.trailing_zeros(), modulus).unwrap();
    let secret_key = NttGlweSecretKey::new(
        vec![0; params.secret_key_len()],
        params.size(),
        SecretKeyDistr::UniformBinary,
    );
    let message: Vec<u64> = (0..POLY_LENGTH)
        .map(|i| (i % PLAIN_MODULUS) as u64)
        .collect();
    let expected_noise: Vec<u64> = (0..POLY_LENGTH).map(|i| [0, 3, 5][i % 3]).collect();
    let mut ciphertext = NttGlweCiphertext::<Vec<u64>>::zero(params.glwe_len());
    for embedding in [PlaintextEmbedding::Unsigned, PlaintextEmbedding::Centered] {
        // A zero mask lets us specify exact positive/negative phase errors.
        let (_, body) = ciphertext.a_b_mut_slices(POLY_LENGTH);
        params
            .plaintext_codec()
            .encode_slice_to(&message, body, embedding);
        for (index, value) in body.iter_mut().enumerate() {
            let error = [0, 3, 1_125_899_906_826_241 - 5][index % 3];
            *value = modulus.reduce_add(*value, error);
        }
        table.transform_slice(body);
        let (decoded, noise) =
            secret_key.decrypt_with_noise_and_embedding(&ciphertext, &params, &table, embedding);
        assert_eq!(decoded.as_ref(), message);
        assert_eq!(noise.as_ref(), expected_noise);
    }
}

#[test]
fn truncated_decryption_returns_only_retained_coefficients() {
    let modulus = BarrettModulus::new(1_125_899_906_826_241u64);
    let params = GlweParameters::new(
        DIMENSION,
        POLY_LENGTH,
        PLAIN_MODULUS as u64,
        modulus,
        SecretKeyDistr::UniformBinary,
        0.7,
    );
    let table = UintNttTable::new(POLY_LENGTH.trailing_zeros(), modulus).unwrap();
    let mut rng = StdRng::seed_from_u64(42);
    let secret_key = NttGlweSecretKey::generate(&params, &table, &mut rng);
    for count in [0, 32, POLY_LENGTH] {
        let mut ciphertext = secret_key.encrypt_truncated_zeros(count, &params, &table, &mut rng);
        let message: Vec<_> = (0..count)
            .map(|i| ((3 * i + 1) % PLAIN_MODULUS) as u64)
            .collect();
        // A nonzero prefix checks coefficient contents/order as well as truncation.
        params.plaintext_codec().add_encode_slice_assign(
            &mut ciphertext.as_mut()[params.size().mask_len()..],
            &message,
            PlaintextEmbedding::Unsigned,
        );
        assert_eq!(
            secret_key.decrypt_truncated(&ciphertext, &params, &table),
            message
        );
    }
}
