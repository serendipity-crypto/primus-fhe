use primus_integer::{AsFrom, AsInto, FheUint};
use primus_modulus::{BarrettModulus, CompactModulus, NativeModulus, PowOf2Modulus, UintModulus};
use primus_reduce::EncodeSigned;

#[test]
fn bounded_signed_encoding_matches_integer_remainders() {
    fn check<T: FheUint>(modulus: impl EncodeSigned<T>) {
        let q = modulus
            .explicit_value()
            .map_or(1i128 << T::BITS, |q| q.as_into());
        let min = -(1i128 << (T::BITS - 1));
        let max = -min - 1;
        let input: Vec<T::SignedInteger> = [
            min,
            min + 1,
            -(q - 1),
            -98,
            -97,
            -96,
            -1,
            0,
            1,
            96,
            97,
            98,
            q - 1,
            max,
        ]
        .into_iter()
        .filter(|&v| v >= min && v <= max && v.abs() < q)
        .map(T::SignedInteger::as_from)
        .cycle()
        .take(65)
        .collect();
        for length in [0, 1, input.len()] {
            let input = &input[..length];
            let expected: Vec<T> = input
                .iter()
                .map(|&v| {
                    let v: i128 = v.as_into();
                    T::as_from(v.rem_euclid(q))
                })
                .collect();
            let mut output = vec![T::MAX; input.len()];
            modulus.encode_signed_slice_to(input, &mut output);
            assert_eq!(output, expected);
            for (&value, &expected) in input.iter().zip(&expected) {
                assert_eq!(modulus.encode_signed(value), expected);
            }
        }
    }
    fn check_moduli<T: FheUint>() {
        check(NativeModulus::<T>::new());
        check(BarrettModulus::new(T::as_from(97u32)));
        check(CompactModulus::new(T::as_from(97u32)));
        check(UintModulus::new(T::as_from(97u32)));
        // A full-width explicit modulus includes signed MIN in the bounded domain.
        check(UintModulus::new(T::MAX));
        check(PowOf2Modulus::new(T::ONE << 7u32));
        check(PowOf2Modulus::new(T::ONE << (T::BITS - 1)));
    }
    check_moduli::<u16>();
    check_moduli::<u32>();
    check_moduli::<u64>();
    #[cfg(feature = "derive")]
    {
        #[derive(primus_modulus::Barrett)]
        #[modulus(ty = u64, value = 97)]
        struct DerivedModulus;
        check(DerivedModulus);
    }
}

#[test]
fn signed_slice_length_is_checked_before_writing() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let modulus = UintModulus::new(97u32);
    for input in [&[-1i32][..], &[-1, 0, 1][..]] {
        let mut output = [17; 2];
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                modulus.encode_signed_slice_to(input, &mut output);
            }))
            .is_err()
        );
        assert_eq!(output, [17; 2]);
    }
}
