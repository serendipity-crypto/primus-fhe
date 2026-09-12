# primus_glwe

[English](README.md) | [简体中文](README.zh_CN.md)

单模数 GLWE 密钥和运算，分别提供 NTT 与原生环面 Fourier 表示。下文 `k` 为 GLWE 维数，`N` 为多项式长度。

## 密钥与表示

| 类型 | 存储与职责 |
| --- | --- |
| `GlweSecretKey<T>` | 有符号系数多项式；采样，以及密钥转换、生成的输入 |
| `NttGlweSecretKey<T>` | 模 `q` 的规范 NTT 剩余类；NTT 加解密 |
| `FourierGlweSecretKey` | 按整数尺度变换的 Fourier 私钥多项式；原生环面 Fourier 加解密 |
| `NttGlwePublicKey<S>` | 一个 NTT 零加密密文；公钥加密 |

随机生成密钥使用 `generate`。密钥不保存变换表，调用方必须保持生成时的模数与变换表示。公钥原始字节采用原生字节序，不包含参数元数据。

## 加密与解密

NTT 私钥、Fourier 私钥和 NTT 公钥共享以下普通加密接口：

| 方法 | 输入 | 输出存储 |
| --- | --- | --- |
| `encrypt` / `encrypt_to` | `[0, t)` 中的明文，使用无符号嵌入 | 分配 / 覆盖 |
| `encrypt_centered_to` | `[0, t)` 中的明文，使用居中嵌入 | 覆盖 |
| `encrypt_encoded_to` | 已编码的密文环系数 | 覆盖；不做明文缩放 |
| `encrypt_zeros` / `encrypt_zeros_to` | 零多项式 | 分配 / 覆盖 |

私钥提供 `decrypt`、`decrypt_to` 和 `phase_to`。Phase 提取返回带噪声的系数域值，不做解码；两种明文嵌入使用同一解码器。解密返回与明文相同的无符号整数类型。

```text
ntt_sk.encrypt_to(input, output, params, ntt_table, rng)
ntt_pk.encrypt_to(input, output, params, ntt_table, rng, context)
fourier_sk.encrypt_to(input, output, params, fft, rng, context)
ntt_sk.phase_to(input, output, modulus, ntt_table)
fourier_sk.phase_to(input, output, fft, context)
```

NTT 私钥的普通运算无需 context。Fourier 运算使用 `FourierGlweEncryptContext<T>` / `FourierGlweDecryptContext`；NTT 公钥加密使用 `NttGlwePublicEncryptContext<T>`。这些 context 按 `N` 构造，并在相同长度下复用。Context 保存工作区而非参数，析构时擦除私密中间值。

`_to` 路径复用输出和工作区。布局、变换不匹配会在写输出前失败。无效明文值可能在部分写入或消耗随机数后触发 panic。已编码的 NTT 输入必须是 `[0, q)` 中的规范剩余类，该范围由调用方保证。

## Gadget 与截断密文

`encrypt_glev_to` 和 `encrypt_ggsw_to` 对系数域环多项式应用 gadget 基，不做明文缩放。因此 GGSW 控制位使用常数多项式 `0` 或 `1`。两者使用由 `GadgetSize` 构造的 gadget context：GLev 要求多项式长度匹配；GGSW 还要求 level 数量匹配。

`encrypt_ggsw_constant_batch_to` 将规范模数剩余类组成的常数切片加密为连续的 NTT GGSW，仅做一次批量检查，不分配临时缓冲区。输出长度为 `input.len() * params.ggsw_len()`。

NTT 的 `encrypt_truncated_zeros`、`phase_truncated` 和 `decrypt_truncated` 操作系数域密文，其 mask 完整，body 最多包含 `N` 个系数。Phase 提取和解密只返回保留的系数，内部工作区仍保存完整多项式。

## 求值原语

求值密钥保存自己的布局和分解基。NTT 求值参数为 `input, output, modulus, ntt, context`；Fourier 求值省略 `modulus`。Context 是可复用的工作区。必须保持密钥的变换表示，仅长度和模数匹配不能证明表示兼容。

两种自同构密钥都提供系数域 `apply_to`。`NttGlweAutomorphismKey::apply_ntt_to` 和 `FourierGlweAutomorphismKey::apply_fourier_to` 复用同一密钥处理变换域输入、输出。Fourier 求值要求使用生成密钥时的同一个 FFT table 实例；直接处理 Fourier 输入、输出的舍入结果可能与系数域往返不同。

`NttGlweTraceKey<T>` 和 `FourierGlweTraceKey<T>` 在以下系数域操作间共享自同构密钥。`M` 表示输入明文；所有输出 phase 系数都可能存在求值误差。

| Trace key 方法 | 目标明文 |
| --- | --- |
| `apply_to` / `apply_reverse_to` | 常数 `N*M[0]` / `M[0]` |
| `apply_partial_to(input, r, ...)` | `d * sum_j M[j*d] X^(j*d)`，其中 `d=N/r` |
| `apply_reverse_partial_to(input, r, ...)` | `sum_j M[j*d] X^(j*d)` |
| `project_coefficient_to` / `project_coefficients_to` | 常数 `M[index]` / 按选择顺序排列的常数 |
| `expand_coefficients_to` | 按系数顺序排列的 `N` 个常数 GLWE |
| `expand_partial_coefficients_to(input, count, ...)` | 明文高位全零时，展开前 `count` 项为常数 |
| `pack_lwe_to` / `pack_lwes_to` | LWE 消息对应的常数 / `p` 个 LWE 的 `sum_i m[i] X^(i*N/p)` |

Partial trace 的 `retained_coefficient_count`（`r`）是 `1..=N` 内的 2 的幂。它在一个 GLWE 中保留等间隔位置：`N=8, r=2` 保留索引 0 和 4。`r=N` 复制输入，`r=1` 为 full trace。反向 trace 每级先缩放，再自同构和相加：NTT 乘 `2^-1 mod q`，Fourier 对无符号系数取 `floor(x/2)`。NTT 域运算不直接继承 torus RevHomTrace 的噪声界。

投影支持任意索引、重复索引和空选择，每个索引执行一次反向 trace，写入 `indices.len() * size.glwe_len()` 个值。部分展开在 `count` 个输出 GLWE 块中构建共享树，先按 `count` 归一化一次，再执行 `count-1` 次自同构。`count` 必须是 `1..=N` 内的 2 的幂；`count=1` 复制输入，`count=N` 为完整展开。NTT 归一化使用域上的逆元，Fourier 使用无符号向下除法。两条路径具有不同的误差行为。

部分展开产生常数的前提是明文 `count..N` 项全零。这个未检查前提针对明文，不针对密文 mask 或 body。否则第 `i` 个输出的目标为 `sum_j M[i+j*count] X^(j*count)`。所有输出保持环次数 `N`，使用普通 trace context。

Packing 使用 [RevHomTrace 算法](https://github.com/Stirling75/RevHomTrace/blob/main/src/glwe_conv_rev.rs)。每个 LWE 的维数必须为 `k*N`，私钥等于 GLWE 私钥的系数展平，模数和编码相同。批量输入为 `p` 个完整 LWE 组成的平坦切片，`p` 是 `1..=N` 内的 2 的幂；仅 `p=N` 时槽位相邻。为固定数量构造 `NttGlwePackingContext::new(size, p)` 或 `FourierGlwePackingContext::new(size, p)`。单条 LWE packing 使用 trace context。求值复用输出和工作区，在写入前检查形状、索引及后端兼容性。

## 源码与测试

[私钥](src/secret_key)、[公钥](src/public_key)、[key switching](src/key_switch)、[自同构](src/automorphism)、[trace/packing](src/trace) 和 [scheme switching](src/scheme_switch.rs) 中维护公开契约与实现细节。

测试按操作分组：普通密钥工作流、gadget phase 与 external product、CMUX、key switching、自同构、scheme switching，以及 trace/展开/packing。`tests/common` 保存自同构和 trace 测试共用的小规模朴素 phase oracle。边界拒绝和容量擦除使用独立测试文件。Fourier 自同构和 trace/packing 测试覆盖 RustFFT 和 tfhe-fft。

```sh
cargo test -p primus_glwe
cargo clippy -p primus_glwe --all-targets -- -D warnings
cargo +nightly test -p primus_glwe --features simd
```

## 基准

```sh
cargo bench -p primus_glwe --bench encryption
cargo bench -p primus_glwe --bench primitives
# 执行全部 case，不收集计时样本：
cargo bench -p primus_glwe -- --test
```

两份基准均使用 `(k, N) = (1, 1024)` 和 `(2, 4096)`。每次迭代执行一次操作，复用输出和工作区；密钥、变换表构造与分配在计时外。参数和固定 seed 记录在基准源码中。这些工作负载用于跟踪回归，不用于比较相同安全级别，也不作为安全参数建议。

| 基准 | 测量内容 |
| --- | --- |
| [encryption](benches/encryption.rs) | 私钥/公钥加密、私钥解密、GLev/GGSW 生成；包含采样、编解码及必要变换 |
| [primitives](benches/primitives.rs) | 普通/反向 trace；8 和 `N/8` 项的投影与部分展开；完整展开；1、8、`N` 条 LWE packing；两种 FFT 后端的直接 Fourier 自同构 |

普通和反向 trace 分别按各自 API 的明文尺度测量。投影与部分展开使用同一份明文高位为零的密文，throughput 按输出消息数量计算。编解码变体在 `primus_encoding` 中单独测量。
