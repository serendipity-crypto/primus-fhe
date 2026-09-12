use primus_lwe::{LweParameters, SecretKeyDistr};
use primus_modulus::{BarrettModulus, NativeModulus};

#[test]
fn secret_support_is_checked_at_parameter_construction() {
    for (sigma, valid) in [(1.0, true), (1.1, false)] {
        // With q = 13, truncated magnitudes 12 and 13 straddle the bound.
        let result = std::panic::catch_unwind(|| {
            LweParameters::new(
                16,
                2u32,
                BarrettModulus::new(13),
                SecretKeyDistr::gaussian(sigma),
                0.7,
            )
        });
        assert_eq!(result.is_ok(), valid);
    }
}

#[test]
fn parameters_validate_ciphertext_length() {
    for dimension in [0, usize::MAX] {
        assert!(
            std::panic::catch_unwind(|| LweParameters::new(
                dimension,
                4u32,
                NativeModulus::new(),
                SecretKeyDistr::UniformBinary,
                0.7,
            ))
            .is_err()
        );
    }
    // Parameter construction does not allocate key or ciphertext storage.
    let params = LweParameters::new(
        usize::MAX - 1,
        4u32,
        NativeModulus::new(),
        SecretKeyDistr::UniformBinary,
        0.7,
    );
    assert_eq!(params.dimension() + 1, usize::MAX);
}
