use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use primus_glwe::{
    GlweParameters, GlweSecretKey, NttGlweCiphertext, NttGlweSecretKey, SecretKeyDistr,
};
use primus_modulus::BarrettModulus;
use primus_ntt::{NttTable, UintNttTable};
use primus_poly::Polynomial;

const PLAIN_MODULUS: u64 = 12_289;
const CIPHER_MODULUS: u64 = 1_125_899_906_826_241;

fn bench_glwe_decrypt_arbitrary_modulus(c: &mut Criterion) {
    type Value = u64;
    let mut group = c.benchmark_group("glwe_decrypt/arbitrary_modulus");
    let mut rng = rand::rng();

    for log_n in [12u32, 13] {
        let poly_length = 1usize << log_n;
        let cipher_modulus = BarrettModulus::new(CIPHER_MODULUS);
        let table = UintNttTable::new(log_n, cipher_modulus).unwrap();
        let params = GlweParameters::new(
            1,
            poly_length,
            PLAIN_MODULUS,
            cipher_modulus,
            SecretKeyDistr::UniformBinary,
            3.2,
        );

        let secret_key = GlweSecretKey::generate(&params, &mut rng);
        let secret_key = NttGlweSecretKey::from_coeff_secret_key(&secret_key, &table);
        let message = Polynomial::new(
            (0..poly_length)
                .map(|index| index as Value % PLAIN_MODULUS)
                .collect::<Vec<_>>(),
        );
        let mut ciphertext: NttGlweCiphertext<Vec<Value>> =
            NttGlweCiphertext::zero(poly_length * 2);
        secret_key.encrypt_to(&message, &mut ciphertext, &params, &table, &mut rng);

        let decrypted = secret_key.decrypt(&ciphertext, &params, &table);
        assert_eq!(decrypted.as_ref(), message.as_ref());

        group.bench_with_input(
            BenchmarkId::new("decrypt", poly_length),
            &ciphertext,
            |b, ct| {
                b.iter(|| {
                    let decrypted = secret_key.decrypt(black_box(ct), black_box(&params), &table);
                    black_box(decrypted);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_glwe_decrypt_arbitrary_modulus);
criterion_main!(benches);
