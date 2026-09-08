# primus_encoding

[English](README.md) | 简体中文

Primus FHE 的明文系数编码与解码。

## API

| 编码器 | 编码规则 | 当前用途 |
| --- | --- | --- |
| `RoundedCodec<T>` | `round(lift(m)*q/t)` | LWE 和 TFHE 查找表 |
| `ScaledCodec<T>` | `lift(m)*round(q/t) mod q` | 单模数 GLWE/NTRU |
| `BfvRnsCodec<T,M>` | `lift(m)*floor(Q/t) mod Q` | RNS 系数缩放（`rns` feature） |

所有类型均在 crate 根部重导出。单模数编码器用 `None` 表示原生模数
`2^T::BITS`。两个公开编码器互相独立，复用私有的整数缩放与解码内核。
`t` 整除 `q` 时，两者都使用精确整数尺度 `q/t`；否则 `RoundedCodec`
对每个缩放消息舍入，`ScaledCodec` 使用统一的舍入整数尺度。

整数尺度为二次幂时使用移位，否则使用普通单字乘法。固定尺度构造器的恢复条件
保证 `(t-1)*delta < q`，因此绝对值编码无需模乘。中心取负和累加仍需要
密文模数运算。

仅当 `t` 整除 `q` 时，解码才使用 `round(c/delta) mod t`；舍入后的尺度
是二次幂并不足以保证该等式。其他参数使用原生乘法高半部分或显式窄／宽乘积
比例舍入内核。批处理算术策略均在系数循环外选择。

TFHE 从自身参数层取得逐消息编码器。GLWE TFHE 在构造时验证了明文与密文模数
均相同，因此复用小 LWE 编码器；NTRU 查找表则在构建入口创建一次输出编码器。

这些类型负责系数编码。目前未实现 BFV/BGV 整数槽打包、BGV 的无缩放明文
提升，以及 CKKS 的典范嵌入。

## 编码契约

消息必须是 `[0,t)` 内的规范剩余。无符号嵌入提升到 `[0,t)`，中心嵌入提升到
`[-floor(t/2),ceil(t/2))`，包括 `t=2` 时的 `1 -> -1`。逐消息编码先对绝对值
舍入（中点向上），再应用符号；解码对规范相位乘以 `t/q` 后舍入（中点向上），
结果模 `t`。累加器与解码输入必须是对应密文模数或有序 RNS 基上的规范剩余。

`RoundedCodec` 要求 `t >= 2` 且 `q > t`。`ScaledCodec` 还检查
`abs(t*round(q/t)-q)*(t-1) < q/2`，这是两种嵌入无噪声恢复的充分条件。
对于选定的整数提升 `m` 和噪声 `e`，恢复条件为
`abs((t*delta-q)*m + t*e) < q/2`。生产方与消费方必须使用一致的编码参数和约定。

`BfvRnsCodec` 使用有序密文模数的乘积 `Q`。除 rustdoc 中记录的模数范围与
互素条件外，构造器检查保守的恢复充分条件：`Q > 4*(Q % t)*(t-1)` 和
`gamma > 4*k`，其中 `k` 为模数数量。对于相位 `delta*m+e`，解码的充分条件为
`abs(t*e-(Q % t)*m)/Q + k/gamma < 1/2`。

RNS 编码输出系数域 `CrtPolynomial`，调用方单独执行 NTT 转换。
`decode_coeffs_to` 会覆盖系数域输入，并要求工作区恰好包含
`decode_scratch_len(output.len())` 个元素。该编码器是 BFV 的组成部分，
并非完整 BFV 方案。

单模数切片方法使用 `_to` 表示独立输出，`_assign` 表示原地更新。RNS 使用
`encode_coeffs_to`、`add_encode_coeffs_assign` 和 `decode_coeffs_to`，从明文
切片推导多项式长度。批量编码在写入前检查消息范围和精确长度。

## Feature

- 默认：仅单模数编码器。
- `rns`：启用 `primus_data`、`primus_poly` 和 `primus_rns` 依赖。
- `simd`：启用 nightly SIMD 算术，不会单独启用 `rns`。

## 验证

```sh
cargo test -p primus_encoding
cargo test -p primus_encoding --features rns
cargo +nightly test -p primus_encoding --features rns,simd
cargo bench -p primus_encoding --bench plaintext_codec
```
