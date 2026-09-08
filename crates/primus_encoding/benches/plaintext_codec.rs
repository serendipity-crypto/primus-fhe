use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use primus_encoding::{PlaintextEmbedding, RoundedCodec, ScaledCodec};
const BATCH_LEN: usize = 4096;
const PLAIN_MODULUS: u64 = 12_289;
const CIPHER_MODULUS: u64 = 1_125_899_906_826_241;
const PLAIN_MODULUS_U32: u32 = 12_289;
const CIPHER_MODULUS_U32: u32 = 536_813_569;

fn ciphertext_values(q: u64) -> Vec<u64> {
    (0..BATCH_LEN)
        .map(|index| {
            ((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xd1b5_4a32_d192_ed03) % q
        })
        .collect()
}

fn message_values(t: u64) -> Vec<u64> {
    (0..BATCH_LEN).map(|index| index as u64 % t).collect()
}

fn ciphertext_values_u32(q: u32) -> Vec<u32> {
    (0..BATCH_LEN)
        .map(|index| ((index as u32).wrapping_mul(0x9e37_79b9) ^ 0xd1b5_4a32) % q)
        .collect()
}

fn message_values_u32(t: u32) -> Vec<u32> {
    (0..BATCH_LEN).map(|index| index as u32 % t).collect()
}

fn bench_plaintext_codec_u64(c: &mut Criterion) {
    let t = PLAIN_MODULUS;
    let q = CIPHER_MODULUS;
    let ciphertexts = ciphertext_values(q);
    let messages = message_values(t);
    let rounded_codec = RoundedCodec::new(t, Some(q));
    let native_rounded_codec = RoundedCodec::new(t, None);
    let mut encoded = vec![0u64; BATCH_LEN];
    let accumulator = ciphertext_values(q);
    let fixed_codec = ScaledCodec::new(t, Some(q));
    let centered_messages: Vec<_> = (0..BATCH_LEN).map(|i| (i as u64 * 17) % t).collect();

    let mut group = c.benchmark_group("plaintext_codec/u64");

    // ——— decode ———

    group.bench_with_input(
        BenchmarkId::new("decode_slice/rounded_explicit", BATCH_LEN),
        &ciphertexts,
        |b, ciphertexts| {
            b.iter_batched_ref(
                || black_box(ciphertexts).clone(),
                |values| {
                    rounded_codec.decode_slice_assign(values);
                    black_box(values);
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("decode_slice/rounded_native", BATCH_LEN),
        &ciphertexts,
        |b, ciphertexts| {
            b.iter_batched_ref(
                || black_box(ciphertexts).clone(),
                |values| {
                    native_rounded_codec.decode_slice_assign(values);
                    black_box(values);
                },
                BatchSize::SmallInput,
            );
        },
    );

    // ——— encode ———

    group.bench_function(
        BenchmarkId::new("encode_slice/rounded_explicit", BATCH_LEN),
        |b| {
            b.iter(|| {
                rounded_codec.encode_slice_to(
                    black_box(&messages),
                    black_box(&mut encoded),
                    PlaintextEmbedding::Unsigned,
                );
                black_box(&encoded);
            });
        },
    );

    group.bench_function(
        BenchmarkId::new("encode_slice/rounded_native", BATCH_LEN),
        |b| {
            b.iter(|| {
                native_rounded_codec.encode_slice_to(
                    black_box(&messages),
                    black_box(&mut encoded),
                    PlaintextEmbedding::Unsigned,
                );
                black_box(&encoded);
            });
        },
    );

    // ——— delta-factor ———

    group.bench_function(
        BenchmarkId::new("add_encode_scaled_slice/centered", BATCH_LEN),
        |b| {
            b.iter_batched_ref(
                || black_box(&accumulator).clone(),
                |acc| {
                    fixed_codec.add_encode_slice_assign(
                        acc,
                        black_box(&centered_messages),
                        PlaintextEmbedding::Centered,
                    );
                    black_box(acc);
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

fn bench_plaintext_codec_u32(c: &mut Criterion) {
    let t = PLAIN_MODULUS_U32;
    let q = CIPHER_MODULUS_U32;
    let ciphertexts = ciphertext_values_u32(q);
    let messages = message_values_u32(t);
    let rounded_codec = RoundedCodec::new(t, Some(q));
    let mut encoded = vec![0u32; BATCH_LEN];

    let mut group = c.benchmark_group("plaintext_codec/u32");

    group.bench_with_input(
        BenchmarkId::new("decode_slice/rounded_explicit", BATCH_LEN),
        &ciphertexts,
        |b, ciphertexts| {
            b.iter_batched_ref(
                || black_box(ciphertexts).clone(),
                |values| {
                    rounded_codec.decode_slice_assign(values);
                    black_box(values);
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("encode_slice/rounded_explicit", BATCH_LEN),
        |b| {
            b.iter(|| {
                rounded_codec.encode_slice_to(
                    black_box(&messages),
                    black_box(&mut encoded),
                    PlaintextEmbedding::Unsigned,
                );
                black_box(&encoded);
            });
        },
    );

    group.finish();
}

fn bench_power_of_two(c: &mut Criterion) {
    let messages: Vec<u64> = (0..BATCH_LEN).map(|i| i as u64 % 256).collect();
    let mut output = vec![0; BATCH_LEN];
    let mut group = c.benchmark_group("plaintext_codec/u64/power_of_two");
    for (name, q) in [("native", None), ("explicit", Some(1u64 << 63))] {
        let codec = RoundedCodec::new(256, q);
        group.bench_function(BenchmarkId::new(name, BATCH_LEN), |b| {
            b.iter(|| {
                codec.encode_slice_to(
                    black_box(&messages),
                    black_box(&mut output),
                    PlaintextEmbedding::Unsigned,
                );
            })
        });
    }
    group.finish();
}

// Scalar and short batches expose dispatch overhead; long batches exercise kernels.
fn bench_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("plaintext_codec/u64/dispatch");
    for (name, t, q) in [
        ("shift", 256, Some(1u64 << 63)),
        ("divide", 9, Some(45)),
        ("wide", 12289, Some(u64::MAX - 58)),
    ] {
        let codec = RoundedCodec::new(t, q);
        let scaled = ScaledCodec::new(t, q);
        group.bench_function(format!("{name}/decode_value"), |b| {
            b.iter(|| black_box(&codec).decode_value::<u64>(black_box(1)))
        });
        group.bench_function(format!("{name}/encode_value"), |b| {
            b.iter(|| {
                black_box(&codec).encode_value(black_box(t - 1), PlaintextEmbedding::Centered)
            })
        });
        group.bench_function(format!("{name}/add_encode_value"), |b| {
            b.iter_batched_ref(
                || q.unwrap_or(u64::MAX) - 1,
                |acc| {
                    black_box(&codec).add_encode_value_assign(
                        acc,
                        black_box(t - 1),
                        PlaintextEmbedding::Centered,
                    )
                },
                BatchSize::SmallInput,
            )
        });
        for len in [16, BATCH_LEN] {
            let messages: Vec<_> = (0..len).map(|i| i as u64 % t).collect();
            let mut output = vec![0; len];
            group.bench_function(BenchmarkId::new(format!("{name}/encode"), len), |b| {
                b.iter(|| {
                    black_box(&codec).encode_slice_to(
                        black_box(&messages),
                        black_box(&mut output),
                        PlaintextEmbedding::Centered,
                    )
                })
            });
            group.bench_function(BenchmarkId::new(format!("{name}/decode"), len), |b| {
                b.iter(|| {
                    black_box(&codec).decode_slice_to(black_box(&messages), black_box(&mut output))
                })
            });
            if name == "shift" {
                group.bench_function(BenchmarkId::new("shift/scaled_encode", len), |b| {
                    b.iter(|| {
                        black_box(&scaled).encode_slice_to(
                            black_box(&messages),
                            black_box(&mut output),
                            PlaintextEmbedding::Centered,
                        )
                    })
                });
            }
        }
    }
    group.finish();
}

fn bench_rns(c: &mut Criterion) {
    #[cfg(not(feature = "rns"))]
    let _ = c;
    #[cfg(feature = "rns")]
    {
        use primus_encoding::BfvRnsCodec;
        use primus_modulus::BarrettModulus;
        use primus_poly::{CrtPolynomial, Polynomial};
        use primus_rns::RNSBase;
        let base = RNSBase::new(&[1125899906826241u64, 1125899906629633].map(BarrettModulus::new))
            .unwrap();
        let codec = BfvRnsCodec::new(
            BarrettModulus::new(12289),
            base,
            BarrettModulus::new(2305843009213554689),
        );
        let input = Polynomial::new(
            (0..BATCH_LEN)
                .map(|i| (i as u64 * 17) % 12289)
                .collect::<Vec<_>>(),
        );
        let mut encoded = CrtPolynomial::<Vec<u64>>::zero(BATCH_LEN * 2);
        let mut output = Polynomial::<Vec<u64>>::zero(BATCH_LEN);
        let mut scratch = vec![0; codec.decode_scratch_len(BATCH_LEN)];
        let mut group = c.benchmark_group("plaintext_codec/bfv_rns/two_moduli");
        group.bench_function(BenchmarkId::new("encode_centered", BATCH_LEN), |b| {
            b.iter(|| {
                codec.encode_coeffs_to(
                    black_box(&input),
                    black_box(&mut encoded),
                    PlaintextEmbedding::Centered,
                );
            })
        });
        codec.encode_coeffs_to(&input, &mut encoded, PlaintextEmbedding::Centered);
        group.bench_function(BenchmarkId::new("decode", BATCH_LEN), |b| {
            b.iter_batched_ref(
                || encoded.clone(),
                |phase| {
                    codec.decode_coeffs_to(
                        black_box(phase),
                        black_box(&mut output),
                        black_box(&mut scratch),
                    )
                },
                BatchSize::SmallInput,
            )
        });
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_plaintext_codec_u64,
    bench_plaintext_codec_u32,
    bench_power_of_two,
    bench_rns,
    bench_dispatch
);
criterion_main!(benches);
