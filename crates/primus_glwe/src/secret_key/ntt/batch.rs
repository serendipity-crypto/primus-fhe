//! Batch encryption of constant polynomials into NTT GGSW ciphertexts.

use super::{NttGadgetEncryptContext, NttGlweSecretKey};
use crate::{GlevParameters, NttGgswCiphertext};
use primus_integer::FheUint;
use primus_ntt::NttTable;
use primus_reduce::FieldContext;

impl<T: FheUint> NttGlweSecretKey<T> {
    /// Encrypts a batch of constant ring polynomials into consecutive NTT GGSWs.
    ///
    /// Each `input[i]` represents the polynomial `input[i] + 0X + ... + 0X^(N-1)`.
    /// Applies the gadget basis without plaintext encoding, as in [`Self::encrypt_ggsw_to`].
    ///
    /// `output` uses `[input][row][level][component][coefficient]` storage and
    /// must contain exactly `input.len() * params.ggsw_len()` values. Reuses
    /// `context` without allocating and checks shared resources once per batch.
    /// Empty input with empty output is accepted after resource validation and
    /// consumes no randomness.
    ///
    /// # Correctness
    ///
    /// Each input must be a canonical residue modulo `params.cipher_modulus()`.
    /// This key must use the supplied table's NTT representation.
    ///
    /// # Panics
    ///
    /// Panics on incompatible key, table, workspace, output length, or length
    /// overflow. Compatibility checks precede all output writes and sampling.
    /// A panicking RNG or transform can leave partial output and modified workspace.
    pub fn encrypt_ggsw_constant_batch_to<M, Table, R>(
        &self,
        input: &[T],
        output: &mut [T],
        params: &GlevParameters<T, M>,
        ntt: &Table,
        rng: &mut R,
        context: &mut NttGadgetEncryptContext<T>,
    ) where
        M: FieldContext<T>,
        Table: NttTable<ValueT = T>,
        R: rand::Rng + rand::CryptoRng,
    {
        self.assert_gadget_compatible(params, ntt);
        context.assert_ggsw_compatible(params.size());
        let ggsw_len = params.ggsw_len();
        let output_len = input
            .len()
            .checked_mul(ggsw_len)
            .expect("NTT GGSW batch length overflow");
        assert_eq!(
            output.len(),
            output_len,
            "NTT GGSW batch output layout mismatch"
        );

        for (&constant, output) in input.iter().zip(output.chunks_exact_mut(ggsw_len)) {
            context.encoded.as_mut().fill(T::ZERO);
            context.encoded.as_mut()[0] = constant;
            self.encrypt_ggsw_encoded_kernel_to(
                &mut NttGgswCiphertext::new(output),
                params,
                ntt,
                rng,
                context,
            );
        }
    }
}
