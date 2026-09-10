// cargo bench -p primus_lwe --bench secret_key
// Retain separate allocating/in-place kernels, encoded/signed dot products,
// and the packed rotated-dot kernel; sampler/codec microbenchmarks live below LWE.

use criterion::{Criterion, criterion_group, criterion_main};
use primus_lwe::{LweCiphertext, LweParameters, LweSecretKey, LweSecretKeyRef, SecretKeyDistr};
use primus_modulus::{BarrettModulus, NativeModulus};
use primus_reduce::RingContext;
use rand::{SeedableRng, rngs::StdRng};
use std::hint::black_box;

fn bench_domain<M: RingContext<u32>>(c: &mut Criterion, name: &str, modulus: M, dimension: usize) {
    let mut rng = StdRng::seed_from_u64(0x1_ee01);
    let params = LweParameters::new(dimension, 4, modulus, SecretKeyDistr::UniformBinary, 0.7);
    let sk = LweSecretKey::generate(&params, &mut rng);
    let signed: Vec<i32> = (0..dimension).map(|i| (i % 3) as i32 - 1).collect();
    let encoded_key: Vec<u32> = signed.iter().map(|&s| modulus.encode_signed(s)).collect();
    let plaintext = params
        .plaintext_codec()
        .encode_value(1u32, primus_encoding::PlaintextEmbedding::Unsigned);
    let cipher = LweSecretKeyRef::Signed(&signed).encrypt_encoded(
        plaintext,
        modulus,
        params.cipher_modulus_uniform_distr(),
        params.noise_distribution(),
        &mut rng,
    );
    let mut group = c.benchmark_group(format!("lwe/u32/{name}/n{dimension}"));
    group.bench_function("encrypt_allocating", |b| {
        b.iter(|| sk.encrypt(black_box(1u32), black_box(&params), &mut rng))
    });
    // Compare the same phase operation, ciphertext and logical key; encoding
    // the reusable key happens outside the timed closure.
    group.bench_function("encoded_phase", |b| {
        b.iter(|| {
            black_box(
                LweSecretKeyRef::Encoded(black_box(&encoded_key))
                    .decrypt_phase(black_box(&cipher), black_box(modulus)),
            )
        })
    });
    group.bench_function("signed_phase", |b| {
        b.iter(|| {
            black_box(
                LweSecretKeyRef::Signed(black_box(&signed))
                    .decrypt_phase(black_box(&cipher), black_box(modulus)),
            )
        })
    });
    let mut output = LweCiphertext::zero(dimension);
    group.bench_function("encrypt_to", |b| {
        b.iter(|| {
            sk.encrypt_to(
                black_box(1u32),
                black_box(&mut output),
                black_box(&params),
                &mut rng,
            )
        });
    });
    group.bench_function("signed_encrypt_encoded_to", |b| {
        b.iter(|| {
            let params = black_box(&params);
            LweSecretKeyRef::Signed(black_box(&signed)).encrypt_encoded_to(
                black_box(plaintext),
                black_box(&mut output),
                params.cipher_modulus(),
                params.cipher_modulus_uniform_distr(),
                params.noise_distribution(),
                &mut rng,
            )
        });
    });
    // Full-capacity packed encryption measures the shared mask/rotated-dot kernel.
    let messages: Vec<u32> = (0..dimension).map(|i| (i % 4) as u32).collect();
    group.bench_function("packed_messages_allocating", |b| {
        b.iter(|| sk.encrypt_multi_messages(black_box(&messages), black_box(&params), &mut rng))
    });
    group.finish();
}
fn secret_key(c: &mut Criterion) {
    // Keep the aligned baseline and a non-power-of-two mask with a SIMD tail.
    for dimension in [512, 805] {
        bench_domain(c, "native", NativeModulus::new(), dimension);
        bench_domain(c, "explicit", BarrettModulus::new(132_120_577), dimension);
    }
}
criterion_group!(benches, secret_key);
criterion_main!(benches);
