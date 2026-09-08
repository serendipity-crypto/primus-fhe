use primus_encoding::{PlaintextEmbedding, RoundedCodec, ScaledCodec};
use primus_integer::FheUint;

fn message_values<T: FheUint>(t: T) -> Vec<T> {
    let len: usize = t.try_into().unwrap();
    (0..len).map(|value| T::try_from(value).unwrap()).collect()
}

fn assert_codec_roundtrip<T: FheUint>(codec: RoundedCodec<T>, t: T, q: Option<T>) {
    let messages = message_values(t);
    let scaled = ScaledCodec::new(t, q);

    for embedding in [PlaintextEmbedding::Unsigned, PlaintextEmbedding::Centered] {
        let mut encoded = vec![T::ZERO; messages.len()];
        codec.encode_slice_to(&messages, &mut encoded, embedding);

        for (&message, &encoded_value) in messages.iter().zip(&encoded) {
            assert_eq!(codec.encode_value(message, embedding), encoded_value);
            assert_eq!(codec.decode_value::<T>(encoded_value), message);
        }

        let mut decoded = vec![T::ZERO; messages.len()];
        codec.decode_slice_to(&encoded, &mut decoded);
        assert_eq!(decoded, messages);

        let mut inplace = messages.clone();
        codec.encode_slice_assign(&mut inplace, embedding);
        assert_eq!(inplace, encoded);
        codec.decode_slice_assign(&mut inplace);
        assert_eq!(inplace, messages);

        let mut accumulated = vec![T::ZERO; messages.len()];
        codec.add_encode_slice_assign(&mut accumulated, &messages, embedding);
        assert_eq!(accumulated, encoded);

        let mut delta_encoded = vec![T::ZERO; messages.len()];
        scaled.add_encode_slice_assign(&mut delta_encoded, &messages, embedding);

        for (&message, &encoded_value) in messages.iter().zip(&delta_encoded) {
            assert_eq!(scaled.encode_value(message, embedding), encoded_value);
            assert_eq!(scaled.decode_value::<T>(encoded_value), message);
        }

        let mut delta_decoded = vec![T::ZERO; messages.len()];
        scaled.decode_slice_to(&delta_encoded, &mut delta_decoded);
        assert_eq!(delta_decoded, messages);
    }
}

#[test]
fn scaled_narrow_roundtrip_near_product_limit() {
    let t = 12_289u64;
    let q = u64::MAX / t;
    assert!(q.checked_mul(t).is_some());
    assert!(q.checked_add(1).unwrap().checked_mul(t).is_none());

    let narrow = RoundedCodec::new(t, Some(q));
    assert_codec_roundtrip(narrow, t, Some(q));
}

fn check_profiles<T: FheUint + TryFrom<u64>>(scaled_t: u64, narrow_q: u64, wide_q: u64) {
    let value = |x| T::try_from(x).ok().unwrap();
    let profiles = [
        (value(256), None),
        (value(scaled_t), None),
        (value(256), Some(T::ONE << (T::BITS - 1))),
        (value(scaled_t), Some(value(narrow_q))),
        (value(scaled_t), Some(value(wide_q))),
    ];
    for (t, q) in profiles {
        assert_codec_roundtrip(RoundedCodec::new(t, q), t, q);
    }
}

#[test]
fn u16_slice_profiles() {
    check_profiles::<u16>(7, 4093, 65521);
}

#[test]
fn u32_slice_profiles() {
    check_profiles::<u32>(251, 16_777_213, 4_294_967_291);
}

#[test]
fn u64_slice_profiles() {
    check_profiles::<u64>(251, u64::MAX / 251, u64::MAX - 58);
}
