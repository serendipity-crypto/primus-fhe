//! Reusable public-key encryption workspace.

use primus_integer::FheUint;
use primus_lattice::{MAX_POLY_LENGTH, MIN_POLY_LENGTH};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Reusable ephemeral secret storage for NTT GLWE public-key encryption.
///
/// Each encryption overwrites the polynomial before using it. The workspace
/// may be reused across keys and moduli with the same polynomial length;
/// it owns no parameters or precomputation. Storage is securely erased on drop.
pub struct NttGlwePublicEncryptContext<T: FheUint> {
    pub(super) ephemeral: Vec<T>,
}

impl<T: FheUint> NttGlwePublicEncryptContext<T> {
    /// Allocates one ephemeral polynomial of length `poly_length`.
    ///
    /// # Panics
    ///
    /// Panics if the length is not a power of two in the supported GLWE range.
    #[must_use]
    pub fn new(poly_length: usize) -> Self {
        assert!(
            (MIN_POLY_LENGTH..=MAX_POLY_LENGTH).contains(&poly_length)
                && poly_length.is_power_of_two(),
            "invalid public encryption workspace polynomial length"
        );
        Self {
            ephemeral: vec![T::ZERO; poly_length],
        }
    }
}

impl<T: FheUint> Zeroize for NttGlwePublicEncryptContext<T> {
    fn zeroize(&mut self) {
        self.ephemeral.zeroize();
    }
}

impl<T: FheUint> ZeroizeOnDrop for NttGlwePublicEncryptContext<T> {}

impl<T: FheUint> Drop for NttGlwePublicEncryptContext<T> {
    fn drop(&mut self) {
        self.zeroize();
    }
}
