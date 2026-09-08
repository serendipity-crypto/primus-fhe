//! Coefficient encoding survives the Fourier representation conversion.
use primus_encoding::{PlaintextEmbedding, RoundedCodec};
use primus_fft::{FftEngine, FftTable, RustFftTable};

/// Run a polynomial through FFT forward-then-inverse and return the result.
fn fft_roundtrip_u32(values: &[u32], fft: &RustFftTable) -> Vec<u32> {
    let mut engine = FftEngine::new(fft);
    let n = fft.poly_length();
    assert_eq!(values.len(), n);

    let mut fourier = vec![primus_fft::Complex64::default(); fft.fourier_length()];
    engine.forward_as_torus(values, &mut fourier);
    let mut recovered = vec![0u32; n];
    engine.backward_as_torus(&fourier, &mut recovered);
    recovered
}

#[test]
fn fft_independence_same_codec_different_fft() {
    // 3-bit messages (q = 8), native torus u32
    let t = 8u32;
    let codec = RoundedCodec::new(t, None);

    for log_n in [2u32, 3, 4] {
        let fft = RustFftTable::new(log_n).unwrap();
        let n = fft.poly_length();

        // Create a polynomial of messages: [0, 1, 2, ..., t-1, 0, 1, ...]
        let messages: Vec<u32> = (0..n).map(|i| (i as u32) % t).collect();

        // Encode
        let mut encoded = vec![0u32; n];
        codec.encode_slice_to(&messages, &mut encoded, PlaintextEmbedding::Unsigned);

        // FFT roundtrip — this does NOT know about message modulus
        let recovered = fft_roundtrip_u32(&encoded, &fft);

        // Decode — should recover the original messages
        let mut decoded = vec![0u32; n];
        codec.decode_slice_to(&recovered, &mut decoded);
        assert_eq!(
            decoded, messages,
            "FFT independence failed at log_n={}",
            log_n
        );
    }
}

#[test]
fn fft_independence_different_moduli_same_fft() {
    let fft = RustFftTable::new(3).unwrap();
    let n = fft.poly_length();

    // Power-of-two message modulus
    {
        let t = 4u32;
        let codec = RoundedCodec::new(t, None);
        let messages: Vec<u32> = (0..n).map(|i| (i as u32) % t).collect();

        let mut encoded = vec![0u32; n];
        codec.encode_slice_to(&messages, &mut encoded, PlaintextEmbedding::Unsigned);
        let recovered = fft_roundtrip_u32(&encoded, &fft);

        let mut decoded = vec![0u32; n];
        codec.decode_slice_to(&recovered, &mut decoded);
        assert_eq!(decoded, messages, "pow2 moduli with same FFT failed");
    }

    // Non-power-of-two message modulus
    {
        let t = 7u32;
        let codec = RoundedCodec::new(t, None);
        let messages: Vec<u32> = (0..n).map(|i| (i as u32) % t).collect();

        let mut encoded = vec![0u32; n];
        codec.encode_slice_to(&messages, &mut encoded, PlaintextEmbedding::Unsigned);
        let recovered = fft_roundtrip_u32(&encoded, &fft);

        let mut decoded = vec![0u32; n];
        codec.decode_slice_to(&recovered, &mut decoded);
        assert_eq!(decoded, messages, "non-pow2 moduli with same FFT failed");
    }
}
