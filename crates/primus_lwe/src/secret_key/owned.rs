use primus_data::{Data, DataMut};
use primus_integer::{FheUint, Size};
use primus_lattice::lwe::Lwe;
use primus_reduce::RingContext;
use rand::distr::Distribution;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{LweCiphertext, LweParameters, PlaintextEmbedding, SecretKeyDistr};

use super::LweSecretKeyRef;

/// Represents a secret key for the Learning with Errors (LWE) cryptographic scheme.
///
/// Secret coefficients are erased when the key (including each clone) is dropped.
/// Message operations use [`LweParameters`] for encoding and sampling; use
/// [`Self::as_view`] to borrow the coefficients for single-ciphertext raw operations.
/// Batch operations, including raw encoded encryption and phase decryption,
/// require this owned key, ensuring that reused single-ciphertext operations
/// always receive encoded coefficients.
/// Construction guarantees that the ciphertext length `dimension() + 1` fits
/// in `usize`.
///
/// # Correctness
///
/// Operations require the key and parameters to have the same dimension and
/// ciphertext modulus. Coefficients must be canonical residues under that
/// modulus, satisfying [`primus_reduce::ReduceDotProduct`]. The modulus is not
/// stored in the key; callers must preserve it when supplying parameters.
#[derive(Clone)]
pub struct LweSecretKey<T: FheUint> {
    data: Vec<T>,
    distr: SecretKeyDistr,
}

impl<T: FheUint> Zeroize for LweSecretKey<T> {
    #[inline]
    fn zeroize(&mut self) {
        self.data.zeroize();
    }
}

impl<T: FheUint> ZeroizeOnDrop for LweSecretKey<T> {}

impl<T: FheUint> Drop for LweSecretKey<T> {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl<T: FheUint> AsRef<[T]> for LweSecretKey<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<T: FheUint> Size for LweSecretKey<T> {
    #[inline]
    fn byte_count(&self) -> usize {
        self.data.byte_count()
    }
}

impl<T: FheUint> LweSecretKey<T> {
    /// Creates a new [`LweSecretKey<T>`].
    ///
    /// # Correctness
    ///
    /// `key` must contain canonical residues for the modulus used in subsequent
    /// operations and satisfy `distr`. Neither condition is checked here.
    ///
    /// # Panics
    ///
    /// Panics if `key.len() + 1` overflows `usize`.
    #[inline]
    pub fn new(key: Vec<T>, distr: SecretKeyDistr) -> Self {
        key.len().checked_add(1).expect("LWE length overflow");
        Self { data: key, distr }
    }

    /// Borrows the canonical secret coefficients for raw LWE operations.
    #[must_use]
    #[inline]
    pub fn as_view(&self) -> LweSecretKeyRef<'_, T> {
        LweSecretKeyRef::Encoded(self.as_ref())
    }

    /// Returns the dimension of this [`LweSecretKey<T>`].
    #[inline]
    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    /// Returns the distribution of this [`LweSecretKey<T>`].
    #[inline]
    pub fn distr(&self) -> SecretKeyDistr {
        self.distr
    }

    /// Generates a new [`LweSecretKey<T>`] with random values.
    #[inline]
    pub fn generate<R, M>(params: &LweParameters<T, M>, rng: &mut R) -> Self
    where
        R: rand::Rng + rand::CryptoRng,
        M: RingContext<T>,
    {
        let distr = params.secret_key_distr();
        let key = match distr {
            SecretKeyDistr::UniformBinary => {
                primus_distr::sample_uniform_binary_values(params.dimension(), rng)
            }
            SecretKeyDistr::Binary { one_probability } => {
                primus_distr::sample_binary_values_with_probability(
                    params.dimension(),
                    one_probability,
                    rng,
                )
            }
            SecretKeyDistr::SparseTernary => primus_distr::sample_sparse_ternary_values(
                params.cipher_modulus_minus_one(),
                params.dimension(),
                rng,
            ),
            SecretKeyDistr::UniformTernary => primus_distr::sample_uniform_ternary_values(
                params.cipher_modulus_minus_one(),
                params.dimension(),
                rng,
            ),
            SecretKeyDistr::Ternary {
                negative_one_probability,
                one_probability,
            } => primus_distr::sample_ternary_values_with_probabilities(
                params.cipher_modulus_minus_one(),
                params.dimension(),
                negative_one_probability,
                one_probability,
                rng,
            ),
            SecretKeyDistr::FixedHammingWeightBinary { hamming_weight } => {
                primus_distr::sample_fixed_hamming_weight_binary_values(
                    params.dimension(),
                    hamming_weight,
                    rng,
                )
            }
            SecretKeyDistr::FixedHammingWeightTernary {
                negative_one_weight,
                one_weight,
            } => primus_distr::sample_fixed_hamming_weight_ternary_values(
                params.cipher_modulus_minus_one(),
                params.dimension(),
                negative_one_weight,
                one_weight,
                rng,
            ),
            SecretKeyDistr::Gaussian(_) => params
                .secret_key_gaussian()
                .expect("validated Gaussian LWE secret-key distribution")
                .sample_iter(rng)
                .take(params.dimension())
                .collect(),
        };
        Self { data: key, distr }
    }

    /// Encrypts a canonical message in `[0,t)` with unsigned embedding.
    ///
    /// See [`Self::encrypt_with_embedding`] for correctness and panic conditions.
    #[inline]
    pub fn encrypt<R, M, Msg>(
        &self,
        message: Msg,
        params: &LweParameters<T, M>,
        rng: &mut R,
    ) -> LweCiphertext<T>
    where
        Msg: TryInto<T>,
        R: rand::Rng + rand::CryptoRng,
        M: RingContext<T>,
    {
        self.encrypt_with_embedding(message, params, rng, PlaintextEmbedding::Unsigned)
    }

    /// Encrypts a canonical message in `[0,t)` with centered embedding.
    ///
    /// The input remains an unsigned residue: `t - 1` represents `-1`.
    /// See [`Self::encrypt_with_embedding`] for correctness and panic conditions.
    #[inline]
    pub fn encrypt_centered<R, M, Msg>(
        &self,
        message: Msg,
        params: &LweParameters<T, M>,
        rng: &mut R,
    ) -> LweCiphertext<T>
    where
        Msg: TryInto<T>,
        R: rand::Rng + rand::CryptoRng,
        M: RingContext<T>,
    {
        self.encrypt_with_embedding(message, params, rng, PlaintextEmbedding::Centered)
    }

    /// Encodes the message with the selected embedding, then encrypts the encoded residue.
    ///
    /// # Correctness
    ///
    /// The key must satisfy the [type's parameter contract](Self#correctness).
    /// Both embeddings accept residues in `[0,t)`; see
    /// [`RoundedCodec::encode_value`](primus_encoding::RoundedCodec::encode_value).
    ///
    /// # Panics
    ///
    /// Panics before allocating or sampling if `message` cannot be represented
    /// by `T` or lies outside `[0,t)`.
    #[inline]
    pub fn encrypt_with_embedding<R, M, Msg>(
        &self,
        message: Msg,
        params: &LweParameters<T, M>,
        rng: &mut R,
        embedding: PlaintextEmbedding,
    ) -> LweCiphertext<T>
    where
        Msg: TryInto<T>,
        R: rand::Rng + rand::CryptoRng,
        M: RingContext<T>,
    {
        debug_assert_eq!(self.dimension(), params.dimension());

        let plaintext = params.plaintext_codec().encode_value(message, embedding);
        self.as_view().encrypt_encoded(
            plaintext,
            params.cipher_modulus(),
            params.cipher_modulus_uniform_distr(),
            params.noise_distribution(),
            rng,
        )
    }

    /// Encrypts a canonical message in `[0,t)` into existing storage.
    ///
    /// See [`Self::encrypt_with_embedding_to`] for contracts and panic conditions.
    #[inline]
    pub fn encrypt_to<R, M, Msg>(
        &self,
        message: Msg,
        output: &mut Lwe<impl DataMut<Elem = T>>,
        params: &LweParameters<T, M>,
        rng: &mut R,
    ) where
        Msg: TryInto<T>,
        R: rand::Rng + rand::CryptoRng,
        M: RingContext<T>,
    {
        self.encrypt_with_embedding_to(message, output, params, rng, PlaintextEmbedding::Unsigned);
    }

    /// Encrypts a canonical message in `[0,t)` with centered embedding into
    /// existing storage. `t - 1` represents `-1`.
    ///
    /// See [`Self::encrypt_with_embedding_to`] for contracts and panic conditions.
    #[inline]
    pub fn encrypt_centered_to<R, M, Msg>(
        &self,
        message: Msg,
        output: &mut Lwe<impl DataMut<Elem = T>>,
        params: &LweParameters<T, M>,
        rng: &mut R,
    ) where
        Msg: TryInto<T>,
        R: rand::Rng + rand::CryptoRng,
        M: RingContext<T>,
    {
        self.encrypt_with_embedding_to(message, output, params, rng, PlaintextEmbedding::Centered);
    }

    /// Encodes a canonical message and overwrites all coefficients of `output`.
    /// No allocation is performed.
    ///
    /// # Correctness
    ///
    /// The key must satisfy the [type's parameter contract](Self#correctness).
    ///
    /// # Panics
    ///
    /// Panics before writing if `message` cannot be represented by `T`, is outside
    /// `[0,t)`, or the output length is not `self.dimension() + 1`. A panicking
    /// RNG can leave partial output.
    #[inline]
    pub fn encrypt_with_embedding_to<R, M, Msg>(
        &self,
        message: Msg,
        output: &mut Lwe<impl DataMut<Elem = T>>,
        params: &LweParameters<T, M>,
        rng: &mut R,
        embedding: PlaintextEmbedding,
    ) where
        Msg: TryInto<T>,
        R: rand::Rng + rand::CryptoRng,
        M: RingContext<T>,
    {
        debug_assert_eq!(self.dimension(), params.dimension());
        let plaintext = params.plaintext_codec().encode_value(message, embedding);
        self.as_view().encrypt_encoded_to(
            plaintext,
            output,
            params.cipher_modulus(),
            params.cipher_modulus_uniform_distr(),
            params.noise_distribution(),
            rng,
        );
    }

    /// Decrypts the [`LweCiphertext<T>`] back to message.
    ///
    /// Returns a canonical residue in `[0,t)` for either encryption embedding.
    ///
    /// # Correctness
    ///
    /// The key must satisfy the [type's parameter contract](Self#correctness).
    /// The ciphertext must have dimension `params.dimension()` and canonical
    /// coefficients under the same modulus. Recovering the original message
    /// also requires its noise to remain within the codec's decoding margin.
    ///
    /// # Panics
    ///
    /// Panics if the ciphertext has no body, its mask length differs from the
    /// key length, or the decoded residue cannot be represented by `Msg`.
    #[inline]
    pub fn decrypt<M, Msg>(
        &self,
        cipher_text: &Lwe<impl Data<Elem = T>>,
        params: &LweParameters<T, M>,
    ) -> Msg
    where
        Msg: TryFrom<T>,
        M: RingContext<T>,
    {
        let modulus = params.cipher_modulus();

        debug_assert_eq!(self.dimension(), params.dimension());
        let plaintext = self.as_view().decrypt_phase(cipher_text, modulus);

        params.plaintext_codec().decode_value(plaintext)
    }

    /// Returns the decoded message and noise magnitude under unsigned embedding.
    ///
    /// See [`Self::decrypt_with_noise_and_embedding`] for the distance definition,
    /// correctness and panic conditions.
    #[inline]
    pub fn decrypt_with_noise<M, Msg>(
        &self,
        cipher_text: &Lwe<impl Data<Elem = T>>,
        params: &LweParameters<T, M>,
    ) -> (Msg, T)
    where
        Msg: TryFrom<T>,
        M: RingContext<T>,
    {
        self.decrypt_with_noise_and_embedding(cipher_text, params, PlaintextEmbedding::Unsigned)
    }

    /// Returns the decoded message and noise magnitude under centered embedding.
    ///
    /// The magnitude is unsigned, even for negative noise. See
    /// [`Self::decrypt_with_noise_and_embedding`] for the distance definition,
    /// correctness and panic conditions.
    #[inline]
    pub fn decrypt_centered_with_noise<M, Msg>(
        &self,
        cipher_text: &Lwe<impl Data<Elem = T>>,
        params: &LweParameters<T, M>,
    ) -> (Msg, T)
    where
        Msg: TryFrom<T>,
        M: RingContext<T>,
    {
        self.decrypt_with_noise_and_embedding(cipher_text, params, PlaintextEmbedding::Centered)
    }

    /// Decrypts the [`LweCiphertext<T>`] and computes noise under the selected embedding.
    ///
    /// Returns `(message, distance)`, where `message` is in `[0,t)` and `distance`
    /// is the nonnegative circular distance between the phase `b - <a,s>` and
    /// the re-encoding of that decoded message. It is not signed noise and does
    /// not detect a failure to recover the original message.
    ///
    /// The correctness and panic conditions of [`Self::decrypt`] apply.
    #[inline]
    pub fn decrypt_with_noise_and_embedding<M, Msg>(
        &self,
        cipher_text: &Lwe<impl Data<Elem = T>>,
        params: &LweParameters<T, M>,
        embedding: PlaintextEmbedding,
    ) -> (Msg, T)
    where
        Msg: TryFrom<T>,
        M: RingContext<T>,
    {
        let modulus = params.cipher_modulus();

        debug_assert_eq!(self.dimension(), params.dimension());
        let plaintext = self.as_view().decrypt_phase(cipher_text, modulus);

        let message: T = params.plaintext_codec().decode_value(plaintext);
        let fresh: T = params.plaintext_codec().encode_value(message, embedding);

        (
            Msg::try_from(message)
                .map_err(|_| "out of range integral type conversion attempted")
                .unwrap(),
            modulus
                .reduce_sub(plaintext, fresh)
                .min(modulus.reduce_sub(fresh, plaintext)),
        )
    }
}
