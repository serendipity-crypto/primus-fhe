use primus_encoding::PlaintextEmbedding;
use primus_integer::FheUint;
use primus_lattice::lwe::LweIter;
use primus_lwe::{LweParameters, LwePublicKey, LweSecretKey, SecretKeyDistr};
use primus_modulus::{BarrettModulus, NativeModulus, PowOf2Modulus};
use primus_reduce::RingContext;
use rand::{Rng, SeedableRng, rngs::StdRng};

#[test]
fn public_batch_matches_independent_matrix_arithmetic() {
    use rand::distr::Distribution;
    fn check<M: RingContext<u32>>(modulus: M) {
        let q = modulus.explicit_value().map_or(1i128 << 32, i128::from);
        let signed = [-1i128, 0, 1, 2, -2, 1, -1];
        let params = LweParameters::new(7, 4, modulus, SecretKeyDistr::gaussian(2.0), 3.2);
        let secret = LweSecretKey::new(
            signed.iter().map(|s| s.rem_euclid(q) as u32).collect(),
            SecretKeyDistr::gaussian(2.0),
        );
        let mut rng = StdRng::seed_from_u64(0x1_ee26);
        let public = LwePublicKey::generate(&secret, &params, &mut rng);
        let plaintexts = [0, (q - 1) as u32, (q / 4) as u32];
        let mut rng = StdRng::seed_from_u64(0x1_ee27);
        let mut oracle_rng = StdRng::seed_from_u64(0x1_ee27);
        let batch = public.encrypt_encoded_batch(
            &plaintexts,
            modulus,
            params.noise_distribution(),
            &mut rng,
        );
        let mut expected = [[0i128; 8]; 3];
        for (row, &plaintext) in expected.iter_mut().zip(&plaintexts) {
            for value in row.iter_mut() {
                *value = i128::from(params.noise_distribution().sample(&mut oracle_rng));
            }
            row[7] += i128::from(plaintext);
        }
        // Integer matrix multiplication checks the row/body layout and gives each
        // output its own independent coefficient for every public-key row.
        for public_row in public.as_slice().as_chunks::<8>().0 {
            for row in &mut expected {
                let r = [0, 0, 1, -1][(oracle_rng.next_u32() & 3) as usize];
                for (out, &value) in row.iter_mut().zip(public_row) {
                    *out += i128::from(value) * r;
                }
            }
        }
        for (ciphertext, expected) in LweIter::new(&batch, 8).zip(expected) {
            for (&actual, expected) in ciphertext.as_ref().iter().zip(expected) {
                assert_eq!(i128::from(actual), expected.rem_euclid(q));
            }
            let dot: i128 = expected[..7].iter().zip(signed).map(|(&a, s)| a * s).sum();
            assert_eq!(
                i128::from(secret.as_view().decrypt_phase(&ciphertext, modulus)),
                (expected[7] - dot).rem_euclid(q)
            );
        }
        assert_eq!(rng.next_u64(), oracle_rng.next_u64());
    }
    check(NativeModulus::new());
    check(BarrettModulus::new(132_120_577));
}

fn check_batches<T: FheUint, M: RingContext<T>>(modulus: M) {
    let params = LweParameters::new(
        7,
        T::as_from(4u32),
        modulus,
        SecretKeyDistr::UniformTernary,
        3.2,
    );
    let mut rng = StdRng::seed_from_u64(0x1_ee21);
    let secret = LweSecretKey::generate(&params, &mut rng);
    let public = LwePublicKey::generate(&secret, &params, &mut rng);
    for count in [0, 1, 7, 8, 17] {
        let messages: Vec<T> = (0..count).map(|i| T::as_from(i % 4)).collect();
        for embedding in [PlaintextEmbedding::Unsigned, PlaintextEmbedding::Centered] {
            for use_public in [false, true] {
                let mut rng = StdRng::seed_from_u64(0x1_ee22);
                let mut raw_rng = StdRng::seed_from_u64(0x1_ee22);
                let mut storage = vec![T::MAX; count * 8 + 2];
                let batch = &mut storage[1..count * 8 + 1];
                if use_public {
                    public.encrypt_batch_with_embedding_to(
                        &messages, batch, &params, &mut rng, embedding,
                    );
                } else {
                    secret.encrypt_batch_with_embedding_to(
                        &messages, batch, &params, &mut rng, embedding,
                    );
                }
                let encoded: Vec<T> = messages
                    .iter()
                    .map(|&m| params.plaintext_codec().encode_value(m, embedding))
                    .collect();
                let raw = if use_public {
                    public.encrypt_encoded_batch(
                        &encoded,
                        modulus,
                        params.noise_distribution(),
                        &mut raw_rng,
                    )
                } else {
                    secret.encrypt_encoded_batch(
                        &encoded,
                        modulus,
                        params.cipher_modulus_uniform_distr(),
                        params.noise_distribution(),
                        &mut raw_rng,
                    )
                };
                assert_eq!(raw.as_slice(), &*batch);
                assert_eq!(secret.decrypt_batch::<_, T>(batch, &params), messages);
                let mut decrypted = vec![T::MAX; count];
                secret.decrypt_batch_to(batch, &mut decrypted, &params);
                assert_eq!(decrypted, messages);
                let phases = secret.decrypt_phase_batch(batch, modulus);
                secret.decrypt_phase_batch_to(batch, &mut decrypted, modulus);
                assert_eq!(decrypted, phases);
                for (ciphertext, &phase) in LweIter::new(batch, 8).zip(&phases) {
                    assert_eq!(secret.as_view().decrypt_phase(&ciphertext, modulus), phase);
                    assert!(
                        ciphertext
                            .as_ref()
                            .iter()
                            .all(|&c| c <= modulus.minus_one())
                    );
                }
                assert_eq!(storage[0], T::MAX);
                assert_eq!(storage[count * 8 + 1], T::MAX);
                if count == 0 {
                    let mut untouched = StdRng::seed_from_u64(0x1_ee22);
                    assert_eq!(rng.next_u64(), untouched.next_u64());
                }
            }
        }
    }
}

#[test]
fn batches_decrypt_across_moduli_embeddings_and_tail_sizes() {
    check_batches(NativeModulus::<u32>::new());
    check_batches(NativeModulus::<u64>::new());
    check_batches(PowOf2Modulus::new(1u32 << 30));
    check_batches(BarrettModulus::new(132_120_577u32));
    check_batches(BarrettModulus::new(1_125_899_906_826_241u64));
}

#[test]
fn raw_secret_batch_accepts_zero_dimension() {
    use rand::distr::Distribution;
    let modulus = BarrettModulus::new(97u32);
    let key = LweSecretKey::new(vec![], SecretKeyDistr::UniformTernary);
    // Message-level parameters disallow dimension zero; raw APIs permit it.
    let params = LweParameters::new(1, 4, modulus, SecretKeyDistr::UniformTernary, 0.7);
    let plaintexts = [0, 96];
    let mut rng = StdRng::seed_from_u64(0x1_ee23);
    let mut oracle_rng = StdRng::seed_from_u64(0x1_ee23);
    let batch = key.encrypt_encoded_batch(
        &plaintexts,
        modulus,
        params.cipher_modulus_uniform_distr(),
        params.noise_distribution(),
        &mut rng,
    );
    let expected: Vec<u32> = plaintexts
        .iter()
        .map(|&plaintext| (plaintext + params.noise_distribution().sample(&mut oracle_rng)) % 97)
        .collect();
    // Without a mask each stored coefficient is already its own phase.
    assert_eq!(batch, expected);
    assert_eq!(key.decrypt_phase_batch(&batch, modulus), expected);
    assert_eq!(rng.next_u64(), oracle_rng.next_u64());
}

#[test]
fn batch_boundaries_validate_layout_and_reject_invalid_messages() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let params = LweParameters::new(
        3,
        4u32,
        NativeModulus::new(),
        SecretKeyDistr::UniformBinary,
        0.7,
    );
    let mut rng = StdRng::seed_from_u64(0x1_ee24);
    let secret = LweSecretKey::generate(&params, &mut rng);
    let public = LwePublicKey::generate(&secret, &params, &mut rng);
    for use_public in [false, true] {
        for length in [0, 7, 12] {
            let mut storage = vec![11; length];
            let mut rng = StdRng::seed_from_u64(31);
            let mut expected = StdRng::seed_from_u64(31);
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    if use_public {
                        public.encrypt_batch_to(&[0u32, 1], &mut storage, &params, &mut rng);
                    } else {
                        secret.encrypt_batch_to(&[0u32, 1], &mut storage, &params, &mut rng);
                    }
                }))
                .is_err()
            );
            assert!(storage.iter().all(|&c| c == 11));
            assert_eq!(rng.next_u64(), expected.next_u64());
        }
        // First-message failures occur before sampling; later failures may
        // leave completed ciphertexts or a partially initialized public-key tile.
        for invalid in [4u64, u64::MAX] {
            for index in [0, 9] {
                let mut messages = [1u64; 10];
                messages[index] = invalid;
                let mut output = vec![17u32; 40];
                let mut rng = StdRng::seed_from_u64(32);
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        if use_public {
                            public.encrypt_batch_to(&messages, &mut output, &params, &mut rng);
                        } else {
                            secret.encrypt_batch_to(&messages, &mut output, &params, &mut rng);
                        }
                    }))
                    .is_err()
                );
                if index == 0 {
                    assert!(output.iter().all(|&c| c == 17));
                    assert_eq!(rng.next_u64(), StdRng::seed_from_u64(32).next_u64());
                }
            }
        }
    }
    let batch = public.encrypt_batch(&[1u32, 2], &params, &mut rng);
    let mut messages = [17u32; 1];
    assert!(
        catch_unwind(AssertUnwindSafe(|| secret.decrypt_batch_to(
            &batch,
            &mut messages,
            &params
        )))
        .is_err()
    );
    assert_eq!(messages, [17]);

    // Slice-based APIs must reject incomplete tails before the chunk iterator
    // can omit them, including a tail with no complete ciphertext at all.
    for length in [1, 7] {
        let input = vec![0u32; length];
        assert!(catch_unwind(|| secret.decrypt_batch::<_, u32>(&input, &params)).is_err());
        assert!(
            catch_unwind(|| secret.decrypt_phase_batch(&input, params.cipher_modulus())).is_err()
        );
        let mut output = vec![17u32; length / 4];
        assert!(
            catch_unwind(AssertUnwindSafe(|| secret.decrypt_batch_to(
                &input,
                &mut output,
                &params
            )))
            .is_err()
        );
        assert!(output.iter().all(|&value| value == 17));
    }
}
