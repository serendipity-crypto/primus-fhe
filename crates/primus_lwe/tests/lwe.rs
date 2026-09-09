use primus_encoding::PlaintextEmbedding;
use primus_integer::FheUint;
use primus_lattice::lwe::{Lwe, MultiMsgLwe};
use primus_lwe::{LweParameters, LweSecretKey, MultiMsgLweCiphertext, SecretKeyDistr};
use primus_modulus::{BarrettModulus, NativeModulus, PowOf2Modulus};
use primus_reduce::RingContext;
use rand::{SeedableRng, rngs::StdRng};

const DIMENSION: usize = 670;
const NOISE_ALPHA: f64 = 2.980_232_238_769_531_2e-8; // 2^-25
const PLAIN_MODULI: [usize; 2] = [256, 257];
const SECRET_KEY_TYPES: [SecretKeyDistr; 3] = [
    SecretKeyDistr::UniformBinary,
    SecretKeyDistr::SparseTernary,
    SecretKeyDistr::Gaussian(3.2),
];

fn noise_standard_deviation_from_q(q: f64) -> f64 {
    (q * NOISE_ALPHA).max(0.7)
}

fn from_usize<T: FheUint>(value: usize) -> T {
    T::try_from(value).unwrap()
}

fn noise_standard_deviation<T, M>(cipher_modulus: M) -> f64
where
    T: FheUint,
    M: RingContext<T>,
{
    let q = cipher_modulus
        .explicit_value()
        .map_or(2.0_f64.powi(T::BITS as i32), |q| q.as_into());
    noise_standard_deviation_from_q(q)
}

fn assert_lwe_secret_key_enc_dec<T, M>(cipher_modulus: M)
where
    T: FheUint,
    M: RingContext<T>,
{
    let mut rng = StdRng::seed_from_u64(0x1_ee02);
    let noise_standard_deviation = noise_standard_deviation(cipher_modulus);

    for secret_key_distr in SECRET_KEY_TYPES {
        for plain_modulus_usize in PLAIN_MODULI {
            let plain_modulus = from_usize(plain_modulus_usize);
            let params = LweParameters::new(
                DIMENSION,
                plain_modulus,
                cipher_modulus,
                secret_key_distr,
                noise_standard_deviation,
            );
            let secret_key = LweSecretKey::generate(&params, &mut rng);

            for message in (0..plain_modulus_usize).map(from_usize::<T>) {
                let ciphertext = secret_key.encrypt(message, &params, &mut rng);

                let decrypted: T = secret_key.decrypt(&ciphertext, &params);
                assert_eq!(decrypted, message);

                let ciphertext = secret_key.encrypt_centered(message, &params, &mut rng);

                let decrypted: T = secret_key.decrypt(&ciphertext, &params);
                assert_eq!(decrypted, message);
            }

            let messages: Vec<T> = (0..DIMENSION)
                .map(|index| from_usize(index % plain_modulus_usize))
                .collect();
            let ciphertext = secret_key.encrypt_multi_messages(&messages, &params, &mut rng);

            let decrypted: Vec<T> = secret_key.decrypt_multi_messages(&ciphertext, &params);
            assert_eq!(decrypted, messages);

            let ciphertext =
                secret_key.encrypt_multi_messages_centered(&messages, &params, &mut rng);

            let decrypted: Vec<T> = secret_key.decrypt_multi_messages(&ciphertext, &params);
            assert_eq!(decrypted, messages);

            let ciphertext = secret_key.encrypt_multi_zeros(DIMENSION, &params, &mut rng);
            let decrypted: Vec<T> = secret_key.decrypt_multi_messages(&ciphertext, &params);
            assert_eq!(decrypted, vec![T::ZERO; DIMENSION]);
        }
    }
}

#[test]
fn test_lwe_secret_key_enc_dec_native_modulus() {
    assert_lwe_secret_key_enc_dec::<u64, _>(NativeModulus::<u64>::new());
    assert_lwe_secret_key_enc_dec::<u32, _>(NativeModulus::<u32>::new());
}

#[test]
fn test_lwe_secret_key_enc_dec_power_of_two_modulus() {
    assert_lwe_secret_key_enc_dec::<u64, _>(PowOf2Modulus::new(1u64 << 50));
    assert_lwe_secret_key_enc_dec::<u32, _>(PowOf2Modulus::new(1u32 << 30));
}

#[test]
fn test_lwe_secret_key_enc_dec_prime_modulus() {
    assert_lwe_secret_key_enc_dec::<u64, _>(BarrettModulus::new(1_125_899_906_826_241u64));
    assert_lwe_secret_key_enc_dec::<u32, _>(BarrettModulus::new(536_813_569u32));
}

#[test]
#[should_panic(expected = "LWE dimension must be non-zero")]
fn parameters_reject_zero_dimension() {
    LweParameters::new(
        0,
        4u32,
        NativeModulus::new(),
        SecretKeyDistr::UniformBinary,
        0.7,
    );
}

#[test]
fn packed_capacity_is_enforced() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let params = LweParameters::new(
        4,
        4u32,
        NativeModulus::new(),
        SecretKeyDistr::UniformBinary,
        0.7,
    );
    let mut rng = StdRng::seed_from_u64(0x1_ee03);
    let key = LweSecretKey::generate(&params, &mut rng);

    // n+1 exposes opposite masks; n+2 also used to underflow the rotation index.
    for count in [5, 6] {
        for embedding in [PlaintextEmbedding::Unsigned, PlaintextEmbedding::Centered] {
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    key.encrypt_multi_messages_with_embedding(
                        &vec![0u32; count],
                        &params,
                        &mut rng,
                        embedding,
                    )
                }))
                .is_err()
            );
        }
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                key.encrypt_multi_zeros(count, &params, &mut rng)
            }))
            .is_err()
        );
        let malformed = MultiMsgLweCiphertext::new(vec![0u32; params.dimension() + count]);
        assert!(
            catch_unwind(|| key.decrypt_multi_messages::<_, u32>(&malformed, &params)).is_err()
        );
    }
}

#[test]
fn packed_samples_decrypt_independently() {
    check_packed_samples(NativeModulus::new());
    check_packed_samples(BarrettModulus::new(132_120_577));
}

fn check_packed_samples<M: RingContext<u32>>(modulus: M) {
    // Exercise both vector blocks and tails as the rotation splits move through the mask.
    let dimension = 513;
    let params = LweParameters::new(dimension, 4, modulus, SecretKeyDistr::UniformTernary, 0.7);
    let pattern = [1, modulus.reduce_neg(1), 0, 1];
    let key = LweSecretKey::new(
        (0..dimension).map(|i| pattern[i % pattern.len()]).collect(),
        SecretKeyDistr::UniformTernary,
    );
    let mut rng = StdRng::seed_from_u64(0x1_ee04);
    let messages: Vec<u8> = (0..dimension).map(|i| (i % 4) as u8).collect();

    for count in [0, 3, dimension] {
        for embedding in [PlaintextEmbedding::Unsigned, PlaintextEmbedding::Centered] {
            let ciphertext = key.encrypt_multi_messages_with_embedding(
                &messages[..count],
                &params,
                &mut rng,
                embedding,
            );
            assert_eq!(ciphertext.0.len(), params.dimension() + count);
            let borrowed = MultiMsgLwe::new(ciphertext.0.as_slice());
            assert_eq!(
                key.decrypt_multi_messages::<_, u8>(&borrowed, &params),
                messages[..count]
            );
            // Extraction uses the lattice layout rather than the shared packed
            // encryption/decryption helper, exposing common rotation errors.
            for (index, &message) in messages[..count].iter().enumerate() {
                let sample = ciphertext.extract_lwe_at(index, params.dimension(), modulus);
                assert_eq!(key.decrypt::<_, u8>(&sample, &params), message);
            }
        }
    }
}

#[test]
fn noise_is_distance_to_the_decoded_message_embedding() {
    let params = LweParameters::new(
        1,
        4u32,
        BarrettModulus::new(97),
        SecretKeyDistr::UniformBinary,
        0.7,
    );
    let key = LweSecretKey::new(vec![1u32], SecretKeyDistr::UniformBinary);

    // For m=2, q=97, t=4, ties away from zero give unsigned E(m)=49
    // and centered E(m)=-49 mod 97=48. Use an independent, known phase.
    for (embedding, encoded) in [
        (PlaintextEmbedding::Unsigned, 49i32),
        (PlaintextEmbedding::Centered, 48),
    ] {
        for noise in [-2i32, 0, 2] {
            let body = (11 + encoded + noise).rem_euclid(97) as u32;
            let storage = [11u32, body];
            let ciphertext = Lwe::new(&storage[..]);
            let result: (u32, u32) = match embedding {
                PlaintextEmbedding::Unsigned => key.decrypt_with_noise(&ciphertext, &params),
                PlaintextEmbedding::Centered => {
                    key.decrypt_centered_with_noise(&ciphertext, &params)
                }
            };
            assert_eq!(result, (2, noise.unsigned_abs()));
        }
    }
}
