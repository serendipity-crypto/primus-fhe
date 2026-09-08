# primus_encoding

English | [简体中文](README.zh_CN.md)

Plaintext coefficient encoding and decoding for Primus FHE.

## APIs

| Codec | Encoding | Current use |
| --- | --- | --- |
| `RoundedCodec<T>` | `round(lift(m)*q/t)` | LWE and TFHE lookup tables |
| `ScaledCodec<T>` | `lift(m)*round(q/t) mod q` | Single-modulus GLWE/NTRU |
| `BfvRnsCodec<T,M>` | `lift(m)*floor(Q/t) mod Q` | RNS coefficient scaling (`rns` feature) |

Public types are available directly at the crate root; implementation modules
are private. The single-modulus codecs accept
`None` for the native modulus `2^T::BITS`. The two public codecs are independent;
they share private integer-scaling and decoding kernels. When `t` divides `q`,
both encode with the exact integer scale `q/t`. Otherwise `RoundedCodec` rounds
each scaled message, while `ScaledCodec` uses one rounded integer scale.

Integer scales use shifts when the scale is a power of two, and ordinary
one-word multiplication otherwise. The fixed-scale constructor's recovery bound
guarantees `(t-1)*delta < q`, so magnitude encoding needs no modular product.
Centered negation and accumulation still use the ciphertext modulus.

Decoding uses `round(c/delta) mod t` only when `t` divides `q`; a power-of-two
rounded scale alone does not imply this identity. Other parameters use the
native high-product or explicit narrow/wide ratio kernels. All batch arithmetic
dispatch occurs outside coefficient loops.

TFHE obtains its per-message codec from its own parameter layer. GLWE TFHE can
reuse its small-LWE codec because construction validates equal plaintext and
ciphertext moduli. NTRU lookup-table construction creates its output codec once
at the compilation boundary.

These are coefficient codecs. BFV/BGV integer slot packing, BGV's unscaled
plaintext lifting, and CKKS canonical embedding are not implemented.

## Encoding contracts

Messages must be canonical residues in `[0,t)`. Unsigned embedding lifts them
to `[0,t)`; centered embedding lifts them to `[-floor(t/2),ceil(t/2))`, including
`1 -> -1` when `t=2`. Rounded encoding rounds the magnitude with ties upward,
then applies its sign; decoding rounds the canonical phase times `t/q`, with
ties upward, modulo `t`. Accumulators and decoding inputs must be canonical
ciphertext residues in the codec's modulus or ordered RNS basis.

`RoundedCodec` requires `t >= 2` and `q > t`. `ScaledCodec` additionally checks
`abs(t*round(q/t)-q)*(t-1) < q/2`, a sufficient condition for noiseless recovery
under either embedding. For a chosen integer lift `m` and noise `e`, its recovery
condition is `abs((t*delta-q)*m + t*e) < q/2`. Encoding parameters and conventions
must agree between producers and consumers.

`BfvRnsCodec` uses the product `Q` of its ordered ciphertext moduli. Its
constructor checks conservative sufficient recovery bounds
`Q > 4*(Q % t)*(t-1)` and `gamma > 4*k`, where `k` is the number of moduli,
as well as the modulus and coprimality conditions documented in rustdoc.
For phase `delta*m+e`, a sufficient decode bound is
`abs(t*e-(Q % t)*m)/Q + k/gamma < 1/2`.

RNS encoding produces coefficient-domain `CrtPolynomial` data; callers perform
NTT conversions separately. `decode_coeffs_to` overwrites its coefficient-domain
input and needs exactly `decode_scratch_len(output.len())` scratch elements.
The codec is a BFV building block, not a complete BFV scheme.

Single-modulus slice methods use `_to` for separate output and `_assign` for
in-place updates. RNS uses `encode_coeffs_to`, `add_encode_coeffs_assign`, and
`decode_coeffs_to`; polynomial length is inferred from the plaintext slice.
Batch encoding validates message ranges and exact lengths before writing.

## Source layout

- `rounded.rs` and `scaled.rs`: the two single-modulus codecs and their APIs.
- `integer_scale.rs`, `decode.rs`, and `helpers.rs`: shared private kernels and
  boundary helpers.
- `bfv_rns/`: the BFV RNS codec; `mod.rs` owns parameters and construction,
  `encode.rs` implements encoding and accumulation, and `decode.rs` implements
  decoding and its workspace contract.

Tests cover API consistency, independent arithmetic oracles, and BFV RNS
contracts. `benches/plaintext_codec.rs` contains the codec benchmarks.

## Features

- Default: single-modulus codecs only.
- `rns`: enables `primus_data`, `primus_poly`, and `primus_rns` dependencies.
- `simd`: enables nightly SIMD arithmetic; does not enable `rns` by itself.

## Validation

```sh
cargo test -p primus_encoding
cargo test -p primus_encoding --features rns
cargo +nightly test -p primus_encoding --features rns,simd
cargo bench -p primus_encoding --bench plaintext_codec
```
