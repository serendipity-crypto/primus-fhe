use primus_integer::FheUint;

/// Computes `round(lhs * rhs / divisor)` with ties upward using two limbs.
///
/// # Correctness
///
/// `divisor` must be nonzero and the rounded quotient must fit in `T`.
/// Equivalently, the high limb after adding `floor(divisor/2)` must be less
/// than `divisor`, as required by `div_wide`.
#[inline]
pub(super) fn mul_div_round<T>(lhs: T, rhs: T, divisor: T) -> T
where
    T: FheUint,
{
    let (lo, hi) = lhs.carrying_mul(rhs, divisor >> 1u32);
    T::div_wide(lo, hi, divisor)
}

/// Computes `round(lhs * rhs / divisor)` with ties upward using one limb.
/// Quotient/remainder rounding avoids adding a potentially overflowing bias
/// to the product.
///
/// # Correctness
///
/// `divisor` must be nonzero and `lhs * rhs` must fit in `T`.
#[inline]
pub(super) fn narrow_mul_div_round<T>(lhs: T, rhs: T, divisor: T) -> T
where
    T: FheUint,
{
    let product = lhs * rhs;
    let (mut quotient, rem) = product.div_rem(divisor);
    if rem >= centered_half(divisor) {
        quotient += T::ONE;
    }
    quotient
}

#[inline]
pub(super) fn try_from_decoded<M, T>(decoded: T) -> M
where
    M: TryFrom<T>,
{
    M::try_from(decoded)
        .map_err(|_| "out of range integral type conversion attempted")
        .unwrap()
}

#[inline]
pub(super) fn centered_half<T: FheUint>(t: T) -> T {
    (t >> 1u32) + (t & T::ONE)
}

#[inline]
pub(super) fn convert_message<M, T>(message: M) -> T
where
    T: FheUint,
    M: TryInto<T>,
{
    message
        .try_into()
        .map_err(|_| "out of range integral type conversion attempted")
        .unwrap()
}

/// Returns the magnitude and negative flag of the centered lift.
///
/// # Correctness
///
/// Requires `t >= 2`, `message < t`, and `half = ceil(t/2)`. A negative lift
/// then has strictly positive magnitude, including `1 -> -1` for `t = 2`.
#[inline]
pub(super) fn lift_centered_from_raw<T: FheUint>(message: T, t: T, half: T) -> (T, bool) {
    if message < half {
        (message, false)
    } else {
        (t - message, true)
    }
}

#[inline]
pub(super) fn checked_message<M: TryInto<T>, T: FheUint>(message: M, t: T) -> T {
    let message = convert_message(message);
    assert!(message < t, "message outside plaintext domain");
    message
}

/// Returns `(floor(q/t), q % t)` after validating the modulus pair.
/// `None` represents the mathematical modulus `q = 2^T::BITS`.
///
/// # Panics
///
/// Panics if `t < 2` or an explicit `q <= t`.
pub(super) fn modulus_div_rem<T: FheUint>(t: T, q: Option<T>) -> (T, T) {
    assert!(t >= T::TWO, "plaintext modulus must be at least 2");
    assert!(
        q.is_none_or(|q| q > t),
        "ciphertext modulus must exceed plaintext modulus"
    );
    match q {
        Some(q) => q.div_rem(t),
        None => {
            let quotient = T::div_wide(T::ZERO, T::ONE, t);
            // q - quotient*t is in [0,t); wrapping recovers it without
            // representing the native modulus q in one limb.
            (quotient, T::ZERO.wrapping_sub(quotient.wrapping_mul(t)))
        }
    }
}
