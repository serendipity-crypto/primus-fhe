//! Owned LWE keys and borrowed views over encoded or signed coefficients.

mod borrowed;
mod owned;
mod packed;

pub use borrowed::LweSecretKeyRef;
pub use owned::LweSecretKey;

use num_traits::Signed;
use primus_integer::{FheUint, SignedInteger};

/// Encodes a signed coefficient with magnitude strictly less than `modulus`.
#[inline]
pub(crate) fn encode_signed<T: FheUint>(coefficient: T::SignedInteger, modulus: T) -> T {
    debug_assert!(coefficient.unsigned_abs() < modulus);
    if coefficient.is_negative() {
        modulus.wrapping_add_signed(coefficient)
    } else {
        coefficient.cast_to_unsigned()
    }
}
