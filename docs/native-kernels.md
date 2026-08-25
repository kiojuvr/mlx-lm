# Native GLM Kernels

The native GLM DSA and routed-MoE kernels are optional. Without them, the
server continues to run through the Python/MLX sparse-prefill and projection
fallbacks.

The vendored sources live under
`mlx_lm/custom_kernels/glm_moe_dsa`. They are derived from oMLX's Apache-2.0
GLM custom kernels; see the license and README in that directory.

## Build requirements

Install full Xcode, not only Command Line Tools. First confirm that the active
developer directory points to Xcode:

```sh
xcode-select -p
xcodebuild -version
```

If `xcode-select -p` reports `/Library/Developer/CommandLineTools`, select the
full Xcode installation and clear the `xcrun` cache:

```sh
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
xcrun --kill-cache
```

Xcode 26 may also require the separate Metal Toolchain component. Check its
status before downloading it:

```sh
xcodebuild -showComponent MetalToolchain
```

If the status is not `installed`, download it:

```sh
xcodebuild -downloadComponent MetalToolchain
```

Clear the tool lookup cache and verify both the selected path and compiler:

```sh
xcrun --kill-cache
xcrun --find metal
xcrun metal --version
```

The build requires MLX 0.32.2, CMake 3.27 or newer, nanobind 2.15.0, and
wheel/setuptools in the isolated build environment. The `pyproject.toml`
build-system section pins these dependencies.

Build the editable checkout:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
MLX_LM_WITH_CUSTOM_KERNEL=1 \
uv pip install --python .venv/bin/python --no-deps -e .
```

If `xcrun metal --version` still invokes the Xcode stub, prefix the verification
and build commands with the toolchain identifier reported by
`xcodebuild -showComponent MetalToolchain`, for example
`TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883`. The identifier is
installation-specific; do not copy the example without checking the local
component report.

## Verification

Verify that the extension is visible:

```sh
.venv/bin/python - <<'PY'
from mlx_lm.models import glm_moe_dsa
print(glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status())
PY
```

Expected fields include:

```text
available=True
source='mlx_lm.custom_kernels.glm_moe_dsa'
```

Run the model-free arithmetic smoke test:

```sh
.venv/bin/python benchmarks/glm52_prefill_benchmark.py \
  --mode native-smoke \
  --json-output glm52-native-smoke.json
```

Expected fields include `native_smoke_passed=True`,
`native_indexer_smoke_passed=True`, and
`native_smoke_source='mlx_lm.custom_kernels.glm_moe_dsa'`. The same run checks
the q8 V-up projection used for quantized GLM DSA `unembed_out` weights.

For timing and tile sweeps, use the commands in
[Prefill and decode benchmarks](glm52-prefill-benchmark.md). Keeping those
measurements in one document avoids duplicating changing benchmark results
here.

## DSA Indexer precision contract

MLA KV quantization does not alter Indexer precision.
`indexer.weights_proj` remains an FP32 `Linear`, even when a mixed or custom
quantization predicate requests quantization globally. The model's hard
exclusion cannot be overridden by that predicate.

The native Indexer uses the versioned FP32 symbols only. Q and K remain FP16 or
BF16; head weights, post-scale/ReLU accumulation, score output, and top-k
selection remain FP32. If those symbols are unavailable, inference falls back
to the exact FP32 block/dense implementation instead of the older 16-bit score
ABI.

Native FP32 score storage is query-chunked so a score tensor stays within
`MLX_LM_GLM_DSA_NATIVE_INDEXER_MAX_SCORE_BYTES`, whose default is 256 MiB.
Every chunk retains its absolute causal offset. If one 64-query Metal tile
would exceed the limit, the model uses the memory-bounded exact block Indexer.

The native Indexer prefill route is enabled by default for supported GLM-5.2 M3
chunks at context 4096 and above. It remains compatible with `--kv-bits 8`
because it uses the separate floating-point DSA Indexer cache. The
single-query decode kernel is an opt-in measurement route:

```sh
MLX_LM_GLM_DSA_NATIVE_DECODE_INDEXER=1
```

It remains off for serving because the tested 8K exact-hit decode run was
slower than the Python/MLX path.

## Masks and fallbacks

Direct GLM attention and MTP masks support boolean hard masks and floating
additive masks with rank 2 or 4. Only internally proven, unpadded full-causal
masks can enter the causal-only native Indexer and native sparse kernels.
Custom masks, batched padding, windows, and other cache-generated masks retain
their values through the exact fallback and selected-KV paths. A fully masked
query row has defined zero attention output.

Unsupported shapes, dtypes, layouts, masks, quantization modes, and sharding
configurations fall back to the existing MLX implementation. Native routes are
optimizations; they do not widen the model's accepted numerical contract.

## Sparse DSA/MLA prefill

The GLM-5.2 sparse prefill path avoids the old full
`(heads, query_length, context_length)` intermediate. It:

1. updates the MLA latent KV cache using the existing cache classes;
2. processes query tokens in microbatches;
3. gathers only selected top-k latent KV and RoPE K rows;
4. dequantizes only selected int8 latent KV in the Python path;
5. absorbs the non-RoPE query into MLA latent space; and
6. performs exact sparse attention before projecting the output.

The Python selected-KV path is enabled by default but waits until the effective
context reaches one token below
`MLX_LM_GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT` (default 131072). Generation
prefill reserves the final prompt token for logits, which explains the
one-token offset.

Disable sparse and native routing only for controlled short-context
comparisons:

```sh
MLX_LM_GLM_DSA_FAST_PREFILL=0 python ...
```

The Python query microbatch and Indexer key block can be tuned with:

```sh
MLX_LM_GLM_DSA_FAST_PREFILL_QUERY_CHUNK=32
MLX_LM_GLM_DSA_FAST_PREFILL_KEY_BLOCK=8192
```

The native sparse MLA route has its own default handoff at context 6144. The
int8 cache route is opt-in:

```sh
MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV=1
MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_MAX_CONTEXT=262144
```

This route keeps cache storage int8 but temporarily dequantizes the full latent
KV tensor for the native kernel. Keep the maximum context bounded until the
target system and prompt distribution have been profiled.

## Routed-MoE prefill

Supported GLM-5.2 prefill calls use two native routed-MoE optimizations by
default:

* a weighted reduction that consumes expert-sorted down-projection output,
  combines inverse scatter with the FP32 router-weighted sum, and avoids an
  unsorted top-k expert tensor;
* a paired Metal dispatch for the affine gate and up projections using a shared
  block plan.

Both paths require at least 64 routed rows and leave single-token decode
unchanged.

The paired gate/up route requires affine mode, group size 64, matching 2-bit or
3-bit quantization, `uint32` packed weights, scale/bias dtype matching the
activation, no added linear bias, and no expert sharding. The weighted-sum
route requires the GLM top-8, hidden-size-6144 shape and FP32 router scores.
Calls failing any predicate use the existing MLX `gather_qmm` or reduction
path.

For controlled A/B measurements:

```sh
MLX_LM_GLM_MOE_PREFILL_GATE_UP=0
MLX_LM_GLM_MOE_PREFILL_WEIGHTED_SUM=0
```

An isolated M3 Ultra synthetic run at model dimensions 6144→2048 measured the
paired gate/up projection at about 16.0x the two-dispatch baseline for 512
tokens and 18.3x for 2048 tokens. This is a projection microbenchmark, not an
end-to-end prefill speedup.

## Experimental projection routes

The q8 and q4 V-up routes and q4 q_a/q_b projection routes are experimental
measurement paths. They are not part of the recommended server profile:

```sh
MLX_LM_GLM_DSA_NATIVE_Q8_VUP=1
MLX_LM_GLM_DSA_NATIVE_Q4_VUP=1
MLX_LM_GLM_DSA_NATIVE_Q4_QA=1
MLX_LM_GLM_DSA_NATIVE_Q4_QB=1
MLX_LM_GLM_DSA_Q_A_DENSE_CACHE=1
```

The dense q_a cache trades approximately 1.5–2 GB of resident memory for a
warmed projection path. Current tile aliases, predicates, per-tile
measurements, profile-isolation guidance, and microbenchmark commands live in
[Prefill and decode benchmarks](glm52-prefill-benchmark.md).
