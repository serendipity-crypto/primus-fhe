// cargo bench -p primus_lwe --bench public_key
// These are performance fixtures, not evaluated security parameters.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use primus_lattice::lwe::LweIterMut;
use primus_lwe::{LweParameters, LwePublicKey, LweSecretKey, SecretKeyDistr};
use primus_modulus::{BarrettModulus, NativeModulus};
use primus_reduce::RingContext;
use rand::{SeedableRng, rngs::StdRng};
use std::hint::black_box;

// One dimension tracks key-generation cost; encryption below varies key sizes
// without repeating generation measurements at every batch count.
fn bench_generation<M: RingContext<u32>>(c: &mut Criterion, name: &str, modulus: M) {
    let dimension = 1024;
    let mut rng = StdRng::seed_from_u64(0x1_ee15);
    let params = LweParameters::new(dimension, 4, modulus, SecretKeyDistr::UniformBinary, 3.2);
    let secret = LweSecretKey::generate(&params, &mut rng);
    let mut group = c.benchmark_group(format!("lwe_public_key/u32/{name}/n{dimension}"));
    // Includes the owned key allocation and all n zero encryptions.
    group.bench_function("generate", |b| {
        b.iter(|| LwePublicKey::generate(black_box(&secret), black_box(&params), &mut rng))
    });
    group.finish();
}

fn bench_batch_domain<M: RingContext<u32>>(c: &mut Criterion, name: &str, modulus: M) {
    // Cover both Gaussian backends; CDT also tracks matrix-size scaling and
    // a non-power-of-two dimension without duplicating those cases for Ziggurat.
    for (sigma, dimension, count) in [
        (3.2, 512, 1),
        (3.2, 512, 64),
        (3.2, 805, 64),
        (3.2, 1024, 64),
        (30.0, 512, 1),
        (30.0, 512, 64),
    ] {
        let mut rng = StdRng::seed_from_u64(0x1_ee25);
        let params =
            LweParameters::new(dimension, 4, modulus, SecretKeyDistr::UniformBinary, sigma);
        let secret = LweSecretKey::generate(&params, &mut rng);
        let public = LwePublicKey::generate(&secret, &params, &mut rng);
        let messages = vec![1u32; count];
        let mut output = vec![0u32; (dimension + 1) * count];
        let mut group = c.benchmark_group(format!(
            "lwe_public_key/u32/{name}/sigma{sigma}/n{dimension}/count{count}"
        ));
        group.throughput(Throughput::Elements(count as u64));
        // One iteration encrypts exactly count independent messages; both cases
        // reuse identical output/key storage and include fresh random sampling.
        group.bench_function("single_loop", |b| {
            b.iter(|| {
                for (&message, mut ciphertext) in black_box(&messages)
                    .iter()
                    .zip(LweIterMut::new(black_box(&mut output), dimension + 1))
                {
                    black_box(&public).encrypt_to(
                        message,
                        &mut ciphertext,
                        black_box(&params),
                        &mut rng,
                    );
                }
            })
        });
        // Count one records single-message latency. Larger counts compare
        // row reuse against the same number of independent encryptions.
        if count > 1 {
            group.bench_function("batch_to", |b| {
                b.iter(|| {
                    black_box(&public).encrypt_batch_to(
                        black_box(&messages),
                        black_box(&mut output),
                        black_box(&params),
                        &mut rng,
                    );
                })
            });
        }
        group.finish();
    }
}

fn public_key(c: &mut Criterion) {
    bench_generation(c, "native", NativeModulus::new());
    bench_generation(c, "explicit", BarrettModulus::new(132_120_577));
    bench_batch_domain(c, "native", NativeModulus::new());
    bench_batch_domain(c, "explicit", BarrettModulus::new(132_120_577));
}

criterion_group!(benches, public_key);
criterion_main!(benches);
