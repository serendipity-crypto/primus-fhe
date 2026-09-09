// cargo bench -p primus_lwe --bench public_key
// These are performance fixtures, not evaluated security parameters.

use criterion::{Criterion, criterion_group, criterion_main};
use primus_lwe::{LweCiphertext, LweParameters, LwePublicKey, LweSecretKey, SecretKeyDistr};
use primus_modulus::{BarrettModulus, NativeModulus};
use primus_reduce::RingContext;
use rand::{SeedableRng, rngs::StdRng};
use std::hint::black_box;

fn bench_domain<M: RingContext<u32>>(c: &mut Criterion, name: &str, modulus: M) {
    for dimension in [512, 1024] {
        let mut rng = StdRng::seed_from_u64(0x1_ee15);
        let params = LweParameters::new(dimension, 4, modulus, SecretKeyDistr::UniformBinary, 3.2);
        let secret = LweSecretKey::generate(&params, &mut rng);
        let key = LwePublicKey::generate(&secret, &params, &mut rng);
        let mut output = LweCiphertext::zero(dimension);
        let mut group = c.benchmark_group(format!("lwe_public_key/u32/{name}/n{dimension}"));
        // Includes the owned key allocation and all n zero encryptions.
        group.bench_function("generate", |b| {
            b.iter(|| LwePublicKey::generate(black_box(&secret), black_box(&params), &mut rng))
        });
        // Reuses the complete public key and output, including fresh sampling
        // each iteration; matrix generation and allocation are outside timing.
        group.bench_function("encrypt_to", |b| {
            b.iter(|| {
                key.encrypt_to(
                    black_box(1u32),
                    black_box(&mut output),
                    black_box(&params),
                    &mut rng,
                )
            })
        });
        group.finish();
    }
}

fn public_key(c: &mut Criterion) {
    bench_domain(c, "native", NativeModulus::new());
    bench_domain(c, "explicit", BarrettModulus::new(132_120_577));
}

criterion_group!(benches, public_key);
criterion_main!(benches);
