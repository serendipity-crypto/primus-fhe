use primus_distr::{SecretKeyDistr, SecretKeyDistrError};

#[test]
fn rejects_invalid_probabilities_and_weights() {
    let cases = [
        (
            SecretKeyDistr::Binary {
                one_probability: f64::NAN,
            },
            8,
            SecretKeyDistrError::InvalidProbability,
        ),
        (
            SecretKeyDistr::Ternary {
                negative_one_probability: 0.6,
                one_probability: 0.5,
            },
            8,
            SecretKeyDistrError::TernaryProbabilitySumExceedsOne,
        ),
        (
            SecretKeyDistr::FixedHammingWeightBinary { hamming_weight: 9 },
            8,
            SecretKeyDistrError::HammingWeightExceedsLength,
        ),
        (
            SecretKeyDistr::FixedHammingWeightTernary {
                negative_one_weight: 4,
                one_weight: 5,
            },
            8,
            SecretKeyDistrError::HammingWeightExceedsLength,
        ),
    ];

    for (distribution, length, expected) in cases {
        assert_eq!(distribution.validate_for_length(length), Err(expected));
    }
}
