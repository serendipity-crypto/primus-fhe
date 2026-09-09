// cargo bench -p primus_lwe --bench batch
// Performance fixtures, not evaluated security parameters.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use primus_lattice::lwe::LweIterMut;
use primus_lwe::{LweParameters, LwePublicKey, LweSecretKey, SecretKeyDistr};
use primus_modulus::{BarrettModulus, NativeModulus};
use primus_reduce::RingContext;
use rand::{SeedableRng, rngs::StdRng};
use std::hint::black_box;

fn bench_domain<M: RingContext<u32>>(c: &mut Criterion, name: &str, modulus: M) {
    for (dimension, count) in [(512, 1), (512, 8), (512, 64), (1024, 64)] {
        let mut rng = StdRng::seed_from_u64(0x1_ee25);
        let params = LweParameters::new(dimension, 4, modulus, SecretKeyDistr::UniformBinary, 3.2);
        let secret = LweSecretKey::generate(&params, &mut rng);
        let public = LwePublicKey::generate(&secret, &params, &mut rng);
        let messages = vec![1u32; count];
        let mut output = vec![0u32; (dimension + 1) * count];
        let mut group =
            c.benchmark_group(format!("lwe_batch/u32/{name}/n{dimension}/count{count}"));
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
        group.finish();
    }
}

fn batch(c: &mut Criterion) {
    bench_domain(c, "native", NativeModulus::new());
    bench_domain(c, "explicit", BarrettModulus::new(132_120_577));
}
criterion_group!(benches, batch);
criterion_main!(benches);
