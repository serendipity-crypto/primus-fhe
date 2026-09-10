use std::fmt::Debug;

use primus_distr::{
    DiscreteGaussian, SignedCDTSampler, SignedDiscreteGaussian, SignedDiscreteZiggurat,
};
use rand::{Rng, SeedableRng, distr::Distribution, rngs::StdRng};

fn check_batch<T: Clone + Default + Debug + PartialEq, D: Distribution<T>>(
    distr: &D,
    length: usize,
    sample_vec: impl FnOnce(&D, usize, &mut StdRng) -> Vec<T>,
    sample_to: impl FnOnce(&D, &mut [T], &mut StdRng),
) {
    let mut rng = StdRng::seed_from_u64(501);
    let expected: Vec<T> = distr.sample_iter(&mut rng).take(length).collect();
    let expected_next = rng.next_u64();

    let mut rng = StdRng::seed_from_u64(501);
    assert_eq!(sample_vec(distr, length, &mut rng), expected);
    assert_eq!(rng.next_u64(), expected_next);

    let mut output = vec![T::default(); length];
    let mut rng = StdRng::seed_from_u64(501);
    sample_to(distr, &mut output, &mut rng);
    assert_eq!(output, expected);
    assert_eq!(rng.next_u64(), expected_next);
}

#[test]
fn signed_support_bound_preserves_custom_backend_tail_cuts() {
    let cdt = SignedDiscreteGaussian::Cdt(SignedCDTSampler::<i32>::new(3.2, 8.0).unwrap());
    let ziggurat =
        SignedDiscreteGaussian::Ziggurat(SignedDiscreteZiggurat::<i32>::new(30.0, 8.0).unwrap());
    assert_eq!(cdt.maximum_magnitude(), 25);
    assert_eq!(ziggurat.maximum_magnitude(), 240);
}

#[test]
fn gaussian_batches_preserve_scalar_samples_and_rng_consumption() {
    for sigma in [3.2, 30.0] {
        let signed = SignedDiscreteGaussian::<i32>::new(sigma).unwrap();
        for length in [0, 1, 17, 257] {
            check_batch(
                &signed,
                length,
                SignedDiscreteGaussian::sample_vec,
                SignedDiscreteGaussian::sample_to,
            );
            for modulus_minus_one in [1_008u64, u64::MAX] {
                let encoded = DiscreteGaussian::new(sigma, modulus_minus_one).unwrap();
                check_batch(
                    &encoded,
                    length,
                    DiscreteGaussian::sample_vec,
                    DiscreteGaussian::sample_to,
                );
            }
        }
    }
}
