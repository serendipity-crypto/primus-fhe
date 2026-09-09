use primus_lwe::{LweParameters, SecretKeyDistr};
use primus_modulus::NativeModulus;

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
