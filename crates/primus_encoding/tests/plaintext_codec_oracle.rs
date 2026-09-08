//! Independent mathematical oracle tests for the first TFHE plaintext-codec profiles.
//!
//! The reference functions deliberately use only `u128` arithmetic and do not
//! call production codec helpers.

use core::fmt::Debug;

use primus_encoding::{PlaintextEmbedding, RoundedCodec, ScaledCodec};
use primus_integer::FheUint;

const PLAINTEXT_MODULI: [u128; 7] = [2, 4, 7, 8, 16, 251, 256];

#[inline]
fn round_half_up(numerator: u128, denominator: u128) -> u128 {
    (numerator + denominator / 2) / denominator
}

#[inline]
fn centered_lift(message: u128, t: u128) -> (u128, bool) {
    let centered_half = t.div_ceil(2);
    if message < centered_half {
        (message, false)
    } else {
        (t - message, true)
    }
}

#[inline]
fn negate_mod(value: u128, q: u128) -> u128 {
    if value == 0 { 0 } else { q - value }
}

fn encode_exact_oracle(message: u128, t: u128, q: u128, embedding: PlaintextEmbedding) -> u128 {
    let (magnitude, is_negative) = match embedding {
        PlaintextEmbedding::Unsigned => (message, false),
        PlaintextEmbedding::Centered => centered_lift(message, t),
    };
    let encoded = round_half_up(magnitude * q, t) % q;
    if is_negative {
        negate_mod(encoded, q)
    } else {
        encoded
    }
}

fn encode_delta_oracle(message: u128, t: u128, q: u128, embedding: PlaintextEmbedding) -> u128 {
    let (magnitude, is_negative) = match embedding {
        PlaintextEmbedding::Unsigned => (message, false),
        PlaintextEmbedding::Centered => centered_lift(message, t),
    };
    let delta = round_half_up(q, t);
    let encoded = (magnitude * delta) % q;
    if is_negative {
        negate_mod(encoded, q)
    } else {
        encoded
    }
}

#[inline]
fn decode_oracle(encoded: u128, t: u128, q: u128) -> u128 {
    round_half_up(encoded * t, q) % t
}

fn to_value<T>(value: u128) -> T
where
    T: TryFrom<u128>,
{
    T::try_from(value).ok().unwrap()
}

fn assert_codec_matches_oracle<T>(explicit_q: Option<T>, q: u128)
where
    T: FheUint + Into<u128> + TryFrom<u128>,
    <T as TryFrom<u128>>::Error: Debug,
{
    for t in PLAINTEXT_MODULI {
        let codec = RoundedCodec::new(to_value(t), explicit_q);
        let scaled = ScaledCodec::new(to_value(t), explicit_q);

        for message in 0..t {
            for embedding in [PlaintextEmbedding::Unsigned, PlaintextEmbedding::Centered] {
                let encoded = codec.encode_value::<T>(to_value(message), embedding).into();
                assert_eq!(
                    encoded,
                    encode_exact_oracle(message, t, q, embedding),
                    "exact encoding mismatch: t={t}, message={message}, embedding={embedding:?}"
                );

                let delta_encoded = scaled
                    .encode_value::<T>(to_value(message), embedding)
                    .into();
                assert_eq!(
                    delta_encoded,
                    encode_delta_oracle(message, t, q, embedding),
                    "delta encoding mismatch: t={t}, message={message}, embedding={embedding:?}"
                );

                assert_eq!(codec.decode_value::<T>(to_value(encoded)).into(), message);
                assert_eq!(
                    scaled.decode_value::<T>(to_value(delta_encoded)).into(),
                    message
                );
            }
        }

        for embedding in [PlaintextEmbedding::Unsigned, PlaintextEmbedding::Centered] {
            let messages: Vec<T> = (0..t).map(to_value).collect();
            let expected: Vec<T> = (0..t)
                .map(|m| to_value(encode_delta_oracle(m, t, q, embedding)))
                .collect();
            let mut encoded = vec![T::ZERO; messages.len()];
            scaled.encode_slice_to(&messages, &mut encoded, embedding);
            assert_eq!(encoded, expected);
            let mut inplace = messages.clone();
            scaled.encode_slice_assign(&mut inplace, embedding);
            assert_eq!(inplace, expected);
            let mut acc = vec![to_value(q - 1); messages.len()];
            scaled.add_encode_slice_assign(&mut acc, &messages, embedding);
            for (&actual, &encoded) in acc.iter().zip(&expected) {
                assert_eq!(actual.into(), (q - 1 + encoded.into()) % q);
            }
        }

        // Exercise values immediately around ideal decoding boundaries. These
        // checks are oracle comparisons rather than roundtrip checks.
        let half_step = q / (2 * t);
        for message in 0..t {
            let center = encode_exact_oracle(message, t, q, PlaintextEmbedding::Unsigned);
            for distance in [
                half_step.saturating_sub(1),
                half_step,
                half_step.saturating_add(1),
            ] {
                for candidate in [(center + distance) % q, (center + q - distance % q) % q] {
                    let decoded: u128 = codec.decode_value::<T>(to_value(candidate)).into();
                    assert_eq!(
                        scaled.decode_value::<T>(to_value(candidate)).into(),
                        decode_oracle(candidate, t, q)
                    );
                    assert_eq!(
                        decoded,
                        decode_oracle(candidate, t, q),
                        "decode mismatch: t={t}, encoded={candidate}"
                    );
                }
            }
        }
    }
}

#[test]
fn native_u32_matches_u128_oracle() {
    assert_codec_matches_oracle::<u32>(None, 1u128 << u32::BITS);
}

#[test]
fn native_u64_matches_u128_oracle() {
    assert_codec_matches_oracle::<u64>(None, 1u128 << u64::BITS);
}

#[test]
fn explicit_u32_matches_u128_oracle() {
    for q in [1_073_692_673u32, u32::MAX - 4, u32::MAX] {
        assert_codec_matches_oracle(Some(q), q.into());
    }
}

#[test]
fn explicit_u64_matches_u128_oracle() {
    for q in [
        1_152_921_504_606_830_593u64,
        u64::MAX / 251,
        u64::MAX - 58,
        u64::MAX,
    ] {
        assert_codec_matches_oracle(Some(q), q.into());
    }
}

#[test]
fn small_moduli_and_nonzero_accumulators() {
    for (t, q) in [
        (12u64, 17u64),
        (4, 8),
        (7, 128),
        (7, 131),
        (2, 4),
        (9, 45),
        (9, 18),
        (16, 131),
    ] {
        let codec = RoundedCodec::new(t, Some(q));
        let delta = (q + t / 2) / t;
        let scaled =
            (2 * (t * delta).abs_diff(q) * (t - 1) < q).then(|| ScaledCodec::new(t, Some(q)));

        for embedding in [PlaintextEmbedding::Unsigned, PlaintextEmbedding::Centered] {
            let messages: Vec<_> = (0..t).collect();
            let mut acc = vec![q - 1; t as usize];
            codec.add_encode_slice_assign(&mut acc, &messages, embedding);
            for (m, &value) in messages.iter().zip(&acc) {
                let expected =
                    encode_exact_oracle(u128::from(*m), u128::from(t), u128::from(q), embedding)
                        as u64;
                assert_eq!(value, (q - 1 + expected) % q);
                assert_eq!(codec.decode_value::<u64>(expected), *m);
            }
            if let Some(scaled) = scaled {
                let expected: Vec<u64> = messages
                    .iter()
                    .map(|&m| encode_delta_oracle(m.into(), t.into(), q.into(), embedding) as u64)
                    .collect();
                let mut encoded = vec![0; messages.len()];
                scaled.encode_slice_to(&messages, &mut encoded, embedding);
                assert_eq!(encoded, expected);
                scaled.decode_slice_assign(&mut encoded);
                assert_eq!(encoded, messages);
                if q % t == 0 {
                    let mut rounded = vec![0; messages.len()];
                    codec.encode_slice_to(&messages, &mut rounded, embedding);
                    assert_eq!(rounded, expected);
                }
            }
            for phase in 0..q {
                if let Some(scaled) = scaled {
                    assert_eq!(
                        scaled.decode_value::<u64>(phase),
                        decode_oracle(phase.into(), t.into(), q.into()) as u64
                    );
                }
                assert_eq!(
                    codec.decode_value::<u64>(phase),
                    decode_oracle(phase.into(), t.into(), q.into()) as u64
                );
            }
        }
    }
    let codec = RoundedCodec::new(1u16 << 15, None);
    assert_eq!(codec.decode_value::<u16>(u16::MAX), 0);
    assert!(std::panic::catch_unwind(|| ScaledCodec::new(12u64, Some(17))).is_err());
}

#[test]
fn rejects_out_of_domain_messages_before_batch_writes() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let rounded = RoundedCodec::new(7u64, Some(131));
    let scaled = ScaledCodec::new(7u64, Some(131));
    for embedding in [PlaintextEmbedding::Unsigned, PlaintextEmbedding::Centered] {
        for m in [7, 8, u64::MAX] {
            assert!(catch_unwind(|| rounded.encode_value(m, embedding)).is_err());
            assert!(catch_unwind(|| scaled.encode_value(m, embedding)).is_err());
            let mut output = [1, 2];
            assert!(
                catch_unwind(AssertUnwindSafe(|| rounded.encode_slice_to(
                    &[0, m],
                    &mut output,
                    embedding
                )))
                .is_err()
            );
            assert_eq!(output, [1, 2]);
            assert!(
                catch_unwind(AssertUnwindSafe(|| scaled.add_encode_slice_assign(
                    &mut output,
                    &[0, m],
                    embedding
                )))
                .is_err()
            );
            assert_eq!(output, [1, 2]);
        }
    }
}
