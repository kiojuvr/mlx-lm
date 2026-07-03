## GLM-5.2 local fork notes

This repository is a GLM-5.2-focused local fork of `mlx-lm`.

It is intended for Apple Silicon users running very large GLM-5.2 MLX models locally, especially Mac Studio M3 Ultra 512GB-class machines. The goal is practical long-context GLM-5.2 operation, not general-purpose upstream compatibility.

This branch is not intended as an upstream `mlx-lm` PR. Several changes intentionally specialize the runtime for GLM-5.2 trusted single-model local use.

### Main changes

- GLM-5.2 / `glm_moe_dsa` support, including DSA shared-indexer cache handling.
- Responses API text content compatibility.
- Automatic local prompt checkpoint save/load inspired by `ds4.c`.
- GLM-5.2-specific MLA latent int8 KV cache support via `--kv-bits 8`.
- Server-side support for `--kv-bits`, `--kv-group-size`, and `--quantized-kv-start`.
- Local GLM-5.2 runtime cache layout designed for one-command invalidation.
- Vendored GLM MoE DSA native custom kernels for optional native DSA indexer
  score/top-k, sparse MLA prefill, and experimental q projection probes. These
  are built from this repository and no longer require a runtime oMLX checkout.

### Native custom-kernel build

The GLM DSA native routes are optional. Without them, the server still runs with
the Python/MLX sparse prefill and projection fallbacks. To build the native
kernels self-contained from this repository, install Apple's full Xcode, not
only Command Line Tools. Xcode 26 may also require the separate Metal Toolchain
component:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
xcodebuild -downloadComponent MetalToolchain
```

The build requires MLX 0.31.2, CMake 3.27+, nanobind 2.12.0, and wheel/setuptools
inside the isolated build environment. The `pyproject.toml` build-system section
pins those build dependencies. With `uv`, rebuild the editable install with:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
MLX_LM_WITH_CUSTOM_KERNEL=1 \
uv pip install --python /Users/kioju/.venvs/mlx-glm52/bin/python --no-deps -e .
```

Verify that the vendored native extension is visible:

```sh
python - <<'PY'
from mlx_lm.models import glm_moe_dsa
print(glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status())
PY
```

Expected fields include `available=True` and
`source='mlx_lm.custom_kernels.glm_moe_dsa'`. The vendored sources live under
`mlx_lm/custom_kernels/glm_moe_dsa` and are derived from oMLX's Apache-2.0 GLM
custom kernels; see the license file in that directory.

For a quick arithmetic smoke test that does not load the full model:

```sh
python benchmarks/glm52_prefill_benchmark.py \
  --mode native-smoke \
  --json-output glm52-native-smoke.json
```

Expected fields include `native_smoke_passed=True`,
`native_indexer_smoke_passed=True`, and
`native_smoke_source='mlx_lm.custom_kernels.glm_moe_dsa'`. The same smoke run
also checks the native q8 V-up projection used for quantized GLM DSA
`unembed_out` weights; expect `native_q8_vup_smoke_passed=True`.

To time native q8 V-up against the MLX `quantized_matmul` fallback without
loading the model:

```sh
python benchmarks/glm52_prefill_benchmark.py \
  --mode native-smoke \
  --native-smoke-benchmark-runs 20 \
  --native-q8-vup-benchmark-q-len 256 \
  --json-output glm52-native-smoke-bench.json
```

### Recommended target model

This branch has been tested with:

[avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw](https://huggingface.co/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw)

Example local path when downloaded through LM Studio:

    ~/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw

### Generate example

```
MLX_METAL_FAST_SYNCH=1 python -m mlx_lm generate \
  --model "$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw" \
  --prompt "Hello. Briefly introduce yourself." \
  --max-tokens 32 \
  --kv-bits 8 \
  --kv-group-size 64 \
  --quantized-kv-start 4096 \
  --temp 0.4 \
  --top-p 0.95
```

### Recommended GLM-5.2 serving settings

For long-context latency on Apple silicon, especially when 200K+ token prompts
are common rather than exceptional, the recommended starting point is:

```
MLX_LM_PROMPT_CHECKPOINT_DEBUG=1 \
MLX_METAL_FAST_SYNCH=1 \
MLX_LM_GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT=131072 \
MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV=1 \
MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_MAX_CONTEXT=65536 \
python -m mlx_lm server \
  --model "$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw" \
  --host 0.0.0.0 \
  --port 8000 \
  --kv-bits 8 \
  --kv-group-size 64 \
  --quantized-kv-start 4096 \
  --prefill-step-size 2048 \
  --prefill-max-qk-tokens 67108864 \
  --glm-dsa-adaptive-prefill-step-size 0 \
  --checkpoint-cache-dir /Volumes/USB-SSD-2/mlx-lm-glm52-local/prompt-checkpoints \
  --checkpoint-save-exact disabled \
  --request-max-tokens-floor 384000 \
  --prompt-concurrency 1 \
  --decode-concurrency 1 \
  --disable-batching \
  --loop-guard-ngram-size 64 \
  --loop-guard-repeats 3 \
  --loop-guard-min-tokens 256
```

Do not pass `--model-name` for this OpenCode setup unless you have explicitly verified that you need request-facing model-name aliasing. The normal single-model local server workflow loads the model from `--model` and serves OpenCode requests through `/v1/chat/completions`.

OpenCode may still send a conservative `max_tokens` value such as 32000 even
when `limit.output` is set higher. `--request-max-tokens-floor 384000` raises
that request cap on the server side so long coding-agent turns are not cut off
early with `finish_reason=length`.

The recommended server command intentionally leaves `--temp` and `--top-p` unset so request-side clients can control sampling. In local use, lower-temperature request settings helped reduce repetitive reasoning loops and “thought-loop” style failure modes while still preserving enough diversity for useful responses.

`--loop-guard-*` is a server-side fuse for exact repeated token loops during long decode, including repeated reasoning/thought spans. The default guard watches for repeated 8/16/32/64-token windows after 256 generated tokens; set `--loop-guard-ngram-size 0` to disable it. If the model still enters near-duplicate but non-exact loops, lower request sampling first (`temperature`, `top_p`) and add a small request-side `repetition_penalty` such as `1.05` to `1.10` when your client supports it.

`--kv-bits 8` is not a prefill-compute speedup by itself. Its value is that GLM MLA int8 KV cache reduces long-context KV memory and keeps 200K+ prompts inside the intended memory envelope. The native DSA indexer score/top-k route remains compatible with this setting because it uses the DSA indexer cache, not the GLM MLA KV cache.

Prompt checkpointing remains the dominant TTFT optimization for repeated coding-agent prefixes. For latency-focused 200K+ serving, `--disable-batching` keeps requests on the single-request path that writes and reuses disk prompt checkpoints, including frontier checkpoints. Disable final exact checkpoints for this long-running server profile: 190K-token exact checkpoints are around 11GB each on the tested setup and can spend tens of seconds writing only to be pruned immediately. The measured cold-prefill sweep favored `--prefill-step-size 2048`, `--prefill-max-qk-tokens 67108864`, and adaptive GLM DSA prefill disabled (`--glm-dsa-adaptive-prefill-step-size 0`). The QK cap shrinks only the chunks whose query-by-context product would get too large. Keep `MLX_LM_GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT` at its default/effective 131072 handoff for the Python selected-KV sparse path; lowering that handoff increased runtime and memory in the tested 128K runs. The vendored native DSA indexer route is enabled by default through `MLX_LM_GLM_DSA_NATIVE_INDEXER` and can replace the Python/MLX indexer score plus top-k path for supported GLM-5.2 M3 chunks at context 4096 and above. The vendored native sparse MLA route has its own lower handoff, `MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_MIN_CONTEXT` (default 8192). `MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV=1` lets it consume int8 GLM MLA KV cache by temporarily dequantizing the full latent KV cache for the native kernel; keep `MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_MAX_CONTEXT` bounded until your target context length is profiled. Do not force `MLX_LM_GLM_DSA_FAST_PREFILL_KEY_BLOCK=2048` unless you are profiling it; the default key block is 8192. If Metal recovery or memory pressure appears on your real prompt distribution, retry with `--prefill-step-size 1024` first, then 512.

For shorter mixed workloads where throughput matters more than per-request TTFT
and disk frontier checkpoints are less important, continuous batching can still
be useful:

```
MLX_METAL_FAST_SYNCH=1 \
python -m mlx_lm server \
  --model "$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw" \
  --host 0.0.0.0 \
  --port 8000 \
  --kv-bits 8 \
  --kv-group-size 64 \
  --quantized-kv-start 4096 \
  --prefill-step-size 2048 \
  --prompt-concurrency 2 \
  --decode-concurrency 2
```

Avoid `--prompt-concurrency 4` as a default: it barely improved TPS at 4096
tokens, but roughly doubled TTFT and added 6 to 9GB peak memory. Also avoid
decode concurrency higher than prompt concurrency for fresh long-prefix
workloads unless you are comfortable with mixed-cache rejection churn; one
q8/p2/c4 run hit 90 rejections and p95 wait around 195s.

### Prompt checkpoints and LCP reuse

This fork automatically stores trusted local GLM-5.2 prompt checkpoints and reuses the longest exact token prefix it can validate. The lookup is deterministic longest-common-prefix reuse, not fuzzy matching:

- exact full-prompt hits replay the last prompt token and skip almost all prefill;
- prefix/frontier hits restore cached KV/DSA state for the shared prefix and prefill only the changed suffix;
- delta hits restore a base prefix/frontier checkpoint plus a suffix-only cache file, avoiding multi-GB exact checkpoint writes for 190K+ token conversations;
- disabled checkpointing reports `checkpoint_resolution=disabled`, `disk_cached_tokens=0`, and a full fresh prefill;
- mismatched tokens, incompatible GLM DSA metadata, incompatible GLM MLA KV settings, malformed checkpoints, or missing files fall back to a normal miss.

Benchmark-backed measurements on this local fork showed exact 4096/8192-token checkpoint hits dropping TTFT from roughly 23s/52s to roughly 0.2s. Controlled LCP runs are the clean way to measure partial reuse: for example, an 8192-token cached prefix plus a 2048-token suffix should report `expected_reused_prefix_tokens=8192`, `disk_cached_tokens=8192`, `fresh_prompt_tokens=2048`, and `checkpoint_expected_match=true`.

Use `--no-prompt-checkpoint` only for cold or disabled-baseline measurements. Use `--checkpoint-save-exact disabled` or `--no-save-exact-checkpoint` when you want to store only configured prefix/frontier checkpoints without also creating a final exact full-prompt checkpoint.

### Controlled LCP benchmark examples

Set `MODEL` once:

```sh
MODEL="$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw"
```

Confirm `echo "$MODEL"` prints the model directory before using examples that
pass `--model "$MODEL"`.

Disabled baseline with the same controlled-LCP row schema:

```sh
LCP_DISABLED_DIR="$(mktemp -d)"
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --mode controlled-lcp \
  --lcp-prefix-tokens 8192 --lcp-suffix-tokens 2048 \
  --max-tokens 1 --prefill-step-size 2048 \
  --fast-prefill disabled \
  --checkpoint-cache-dir "$LCP_DISABLED_DIR" \
  --no-prompt-checkpoint \
  --json-output glm52-lcp-disabled-8192-2048.json
```

6144-token prefix reuse plus a 2048-token suffix:

```sh
LCP_6144_DIR="$(mktemp -d)"
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --mode controlled-lcp \
  --lcp-prefix-tokens 6144 --lcp-suffix-tokens 2048 \
  --max-tokens 1 --prefill-step-size 2048 \
  --fast-prefill disabled \
  --checkpoint-cache-dir "$LCP_6144_DIR" \
  --checkpoint-save-exact disabled \
  --json-output glm52-lcp-6144-2048.json
```

8192-token prefix reuse plus a 2048-token suffix:

```sh
LCP_8192_DIR="$(mktemp -d)"
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --mode controlled-lcp \
  --lcp-prefix-tokens 8192 --lcp-suffix-tokens 2048 \
  --max-tokens 1 --prefill-step-size 2048 \
  --fast-prefill disabled \
  --checkpoint-cache-dir "$LCP_8192_DIR" \
  --checkpoint-save-exact disabled \
  --json-output glm52-lcp-8192-2048.json
```

Same 8192+2048 reuse path with GLM MLA int8 KV cache:

```sh
LCP_INT8_DIR="$(mktemp -d)"
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --mode controlled-lcp \
  --lcp-prefix-tokens 8192 --lcp-suffix-tokens 2048 \
  --max-tokens 1 --prefill-step-size 2048 \
  --fast-prefill disabled \
  --kv-bits 8 --kv-group-size 64 --quantized-kv-start 4096 \
  --checkpoint-cache-dir "$LCP_INT8_DIR" \
  --checkpoint-save-exact disabled \
  --json-output glm52-lcp-int8-8192-2048.json
```

The controlled benchmark raises on checkpoint-enabled prefix/exact rows when actual `disk_cached_tokens` differs from the expected prefix length. Disabled baseline rows are valid comparison rows and should report zero expected and actual disk reuse.

On the tested Mac Studio M3 Ultra 512GB setup, a controlled 10240-token run with an 8192-token cached prefix and a 2048-token fresh suffix reduced TTFT from roughly 71.6s full fresh prefill to roughly 17.8s prefix reuse.

### Fast sparse DSA prefill caveat

GLM DSA sparse prefill is enabled by default because the old dense fallback materialized full `(heads, query_length, context_length)` prefill tensors and could OOM well below the advertised long-context envelope. To avoid the Python selected-KV sparse path becoming pathologically slow too early, it waits until the effective context reaches one token below `MLX_LM_GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT` (default 131072), because generation prefill leaves the final prompt token for logits. For sparse chunks, MLA attention stays in latent space and avoids selected K/V projection; long chunks whose causal prefix already covers the full top-k set also skip the redundant selected-mask gather. Use `--fast-prefill disabled` only for short-context comparison runs. `--fast-prefill-query-chunk` controls selected-query microbatches, and `MLX_LM_GLM_DSA_FAST_PREFILL_KEY_BLOCK` controls the DSA indexer key block size. If the vendored native extension is built, `MLX_LM_GLM_DSA_NATIVE_INDEXER` can route supported GLM-5.2 M3 indexer score/top-k chunks from context 4096 onward. This route remains usable with `--kv-bits 8`. `MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL` can route supported GLM MLA chunks from its own `MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_MIN_CONTEXT` threshold (default 8192), without waiting for the Python sparse handoff. The int8 GLM MLA KV path is opt-in through `MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV=1`; it keeps the cache stored as int8 but temporarily dequantizes the full latent KV tensor for native sparse MLA. Use `MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_MAX_CONTEXT` as a memory guard.

Benchmark rows include `glm_dsa_native_sparse_prefill_route_state`,
`glm_dsa_native_sparse_prefill_primary_fallback`, and
`glm_dsa_native_sparse_prefill_config_blocker`. With
`--native-sparse-quantized-kv enabled`, 32K cold prefill on the tested setup
dropped from about 395s to about 231s, replacing the long Python/MLX sparse
attention tail with native sparse MLA while leaving q projection as the next
large bottleneck. A 16K profile sweep favored the 8192-token native sparse
handoff: it measured about 111s versus about 117s for the previous 11264-token
default, while avoiding the 8K slowdown seen when forcing the handoff to 4096.

The vendored native `glm_dsa_q8_vup_flat` kernel can be enabled separately for
quantized GLM DSA `unembed_out` projection when the fixed M3 GLM shape matches
64 heads, latent dim 512, value dim 256, affine int8 weights, and group size 64.
It is opt-in via `MLX_LM_GLM_DSA_NATIVE_Q8_VUP=1` or `--native-q8-vup enabled`
because the model-free microbench can be slower than MLX `quantized_matmul` on
some lengths. Benchmark rows report `glm_dsa_native_q8_vup_hits` and
`glm_dsa_native_q8_vup_fallback_reasons`.

The vendored native q4 q projection probes are also opt-in:
`MLX_LM_GLM_DSA_NATIVE_Q4_QA=1` / `--native-q4-qa enabled` and
`MLX_LM_GLM_DSA_NATIVE_Q4_QB=1` / `--native-q4-qb enabled`. They are useful for
isolated profiling but are not part of the recommended server command yet. On
the tested 8K cold prefill, q projection split into about 29.7s q_a projection,
0.18s q_a RMSNorm, and 2.1s q_b projection; the q4 native probes did not reduce
end-to-end TTFT.

There is also an opt-in q_a dense-cache probe:
`MLX_LM_GLM_DSA_Q_A_DENSE_CACHE=1` / `--q-a-dense-cache enabled`. It
dequantizes each q4 `q_a_proj` weight to a dense fp16/bf16 matrix on first use
and reuses it for later prefill calls. This trades roughly 1.5-2GB of extra
resident memory for a warmed q_a projection path, so it is a measurement knob
rather than a recommended server setting. On the tested 8K repeat run, the
warmed path was effectively unchanged versus dense-cache disabled.

**Bottleneck hypothesis**
The bottleneck is still long-context prefill itself: later 32k chunks climbed to around 40s per 2048-token chunk. DSA/top-k and long-context attention/dequantization are the likely next places to profile, but checkpoint reuse is the practical answer for repeated coding-agent prefixes right now.

### OpenCode configuration example

OpenCode can use the local `mlx-lm` server through the OpenAI-compatible provider.

Example `opencode.json` provider entry:

```
"mlx-lm": {
  "name": "mlx-lm (local)",
  "npm": "@ai-sdk/openai-compatible",
  "options": {
    "baseURL": "http://mac-studio:8000/v1",
    "apiKey": "mlx-lm",
    "timeout": 7200000,
    "chunkTimeout": 7200000
  },
  "models": {
    "default_model": {
      "name": "GLM-5.2-Alis-MLX-Dynamic-3.5bpw",
      "limit": {
        "context": 1048576,
        "output": 384000
      }
    }
  }
}
```

In this configuration, OpenCode sends requests using the model key `default_model`. The `name` field is a display label for the local GLM-5.2 model.

Do not add numeric sampling parameters such as `temperature` or `top_p` under `models.default_model`. OpenCode expects `provider.<name>.models.<model>.temperature` to be a boolean or omitted, so numeric values there make the configuration invalid.

### Local runtime cache

This fork uses a shared GLM-5.2 local runtime cache root:

    ~/.cache/mlx-lm/glm52-local/

Prompt checkpoints are stored under:

    ~/.cache/mlx-lm/glm52-local/prompt-checkpoints/

A `kv/` directory is also reserved under the same root.

For benchmark runs, prefer an isolated checkpoint directory so measurements do not touch normal serving cache state:

    --checkpoint-cache-dir "$(mktemp -d)"

The server and benchmark `--checkpoint-cache-dir` option maps to the `MLX_LM_PROMPT_CHECKPOINT_CACHE_DIR` environment override. It redirects the prompt checkpoint files and manifest only; no model math, cache tensor layout, or checkpoint validation rules change. For example, to keep prompt checkpoints on the USB SSD:

    --checkpoint-cache-dir /Volumes/USB-SSD-2/mlx-lm-glm52-local/prompt-checkpoints

Use an empty checkpoint directory after changing model weights, quantization, tokenizer, adapters, GLM implementation details, or KV quantization settings. Reusing stale prompt checkpoints from a different runtime is the main way a path move can look like a generation-quality regression.

When changing model weights, quantization, tokenizer, adapters, GLM implementation details, or KV quantization settings, invalidate the local runtime cache by moving or deleting the shared root:

    mv ~/.cache/mlx-lm/glm52-local \
       ~/.cache/mlx-lm/glm52-local.bak.$(date +%Y%m%d_%H%M%S)

To clear only prompt checkpoints while keeping the reserved cache root:

    rm -rf ~/.cache/mlx-lm/glm52-local/prompt-checkpoints

#### Prompt checkpoint retention limits

The prompt checkpoint manifest/pruning layer uses these default limits:

    MLX_LM_PROMPT_CHECKPOINT_MAX_FILES=256
    MLX_LM_PROMPT_CHECKPOINT_MAX_BYTES=256GiB
    MLX_LM_PROMPT_CHECKPOINT_MAX_FRONTIERS_PER_RUN=16

They bound checkpoint file count, total checkpoint storage, and frontier checkpoint saves per generation run.
The byte limit accepts plain bytes or binary/decimal suffixes such as `512GiB`,
`1TiB`, or `2TB`. The 256GiB default is sized to hold a 1M-token GLM-5.2
checkpoint chain with practical headroom for base, frontier, and delta files
before pruning starts.

### Important limitations

- This is a GLM-5.2-specialized fork, not a generic `mlx-lm` runtime.
- GLM MLA KV quantization currently supports only `--kv-bits 8`.
- DSA indexer cache remains floating-point and is intentionally not quantized.
- With GLM MLA `--kv-bits 8`, the server can use the continuous `BatchGenerator` path. It will not merge quantized and unquantized GLM MLA caches in the same active batch, so fresh short requests may wait for an incompatible quantized batch instead of being quantized earlier than `--quantized-kv-start`. Compatible queued requests can still bypass that waiting request and join the active batch.
- Other model families with `--kv-bits` still use the single-request `stream_generate` path unless they grow batch-compatible quantized cache support.
- Prompt checkpointing is trusted single-model local cache reuse. It validates prefix, cache structure, GLM DSA metadata, and GLM MLA KV settings, but it does not prove full model weight, tokenizer, adapter, or artifact identity.
- Long-context decode currently dequantizes the full MLA latent cache on read. Sparse or block-wise dequantization is future work.
- If generation behaves unexpectedly after changing model/runtime settings, clear the GLM-5.2 local runtime cache first.

### Prefill benchmark

See `docs/glm52-prefill-benchmark.md` for the GLM-5.2 prefill benchmark command, measured fields, and current batching notes. The benchmark script lives at `benchmarks/glm52_prefill_benchmark.py`. Use `--mode single` with `--max-tokens 1` for TTFT / checkpoint measurements, and `--mode queued --max-tokens 8` or similar to expose admission wait time under serving load.

### Smoke test result

A short smoke test with `avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw` on Mac Studio M3 Ultra 512GB completed successfully with:

    --kv-bits 8 --kv-group-size 64 --quantized-kv-start 4096

Observed short-prompt result:

    Prompt: 19 tokens, ~6.9 tokens/sec
    Generation: 32 tokens, ~21 tokens/sec
    Peak memory: ~329 GB

Short prompts do not meaningfully demonstrate long-context KV memory savings or prompt checkpoint benefit. Use longer prompts to evaluate those paths.

## MLX LM 

MLX LM is a Python package for generating text and fine-tuning large language
models on Apple silicon with MLX.

Some key features include:

* Integration with the Hugging Face Hub to easily use thousands of LLMs with a
  single command. 
* Support for quantizing and uploading models to the Hugging Face Hub.
* [Low-rank and full model
  fine-tuning](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
  with support for quantized models.
* Distributed inference and fine-tuning with `mx.distributed`

The easiest way to get started is to install the `mlx-lm` package:

**With `pip`**:

```sh
pip install mlx-lm
```

**With `conda`**:

```sh
conda install -c conda-forge mlx-lm
```

### Quick Start

To generate text with an LLM use:

```bash
mlx_lm.generate --prompt "How tall is Mt Everest?"
```

To chat with an LLM use:

```bash
mlx_lm.chat
```

This will give you a chat REPL that you can use to interact with the LLM. The
chat context is preserved during the lifetime of the REPL.

Commands in `mlx-lm` typically take command line options which let you specify
the model, sampling parameters, and more. Use `-h` to see a list of available
options for a command, e.g.:

```bash
mlx_lm.generate -h
```

The default model for generation and chat is
`mlx-community/Llama-3.2-3B-Instruct-4bit`.  You can specify any MLX-compatible
model with the `--model` flag. Thousands are available in the
[MLX Community](https://huggingface.co/mlx-community) Hugging Face
organization.

### Python API

You can use `mlx-lm` as a module:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)
```

To see a description of all the arguments you can do:

```
>>> help(generate)
```

Check out the [generation
example](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/examples/generate_response.py)
to see how to use the API in more detail. Check out the [batch generation
example](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/examples/batch_generate_response.py)
to see how to efficiently generate continuations for a batch of prompts.

The `mlx-lm` package also comes with functionality to quantize and optionally
upload models to the Hugging Face Hub.

You can convert models using the Python API:

```python
from mlx_lm import convert

repo = "mistralai/Mistral-7B-Instruct-v0.3"
upload_repo = "mlx-community/My-Mistral-7B-Instruct-v0.3-4bit"

convert(repo, quantize=True, upload_repo=upload_repo)
```

This will generate a 4-bit quantized Mistral 7B and upload it to the repo
`mlx-community/My-Mistral-7B-Instruct-v0.3-4bit`. It will also save the
converted model in the path `mlx_model` by default.

To see a description of all the arguments you can do:

```
>>> help(convert)
```

#### Streaming

For streaming generation, use the `stream_generate` function. This yields
a generation response object.

For example,

```python
from mlx_lm import load, stream_generate

repo = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
model, tokenizer = load(repo)

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
)

for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
    print(response.text, end="", flush=True)
print()
```

#### Sampling

The `generate` and `stream_generate` functions accept `sampler` and
`logits_processors` keyword arguments. A sampler is any callable which accepts
a possibly batched logits array and returns an array of sampled tokens.  The
`logits_processors` must be a list of callables which take the token history
and current logits as input and return the processed logits. The logits
processors are applied in order.

Some standard sampling functions and logits processors are provided in
`mlx_lm.sample_utils`.

### Command Line

You can also use `mlx-lm` from the command line with:

```
mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.3 --prompt "hello"
```

This will download a Mistral 7B model from the Hugging Face Hub and generate
text using the given prompt.

For a full list of options run:

```
mlx_lm.generate --help
```

To quantize a model from the command line run:

```
mlx_lm.convert --model mistralai/Mistral-7B-Instruct-v0.3 -q
```

For more options run:

```
mlx_lm.convert --help
```

You can upload new models to Hugging Face by specifying `--upload-repo` to
`convert`. For example, to upload a quantized Mistral-7B model to the
[MLX Hugging Face community](https://huggingface.co/mlx-community) you can do:

```
mlx_lm.convert \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    -q \
    --upload-repo mlx-community/my-4bit-mistral
```

Models can also be converted and quantized directly in the
[mlx-my-repo](https://huggingface.co/spaces/mlx-community/mlx-my-repo) Hugging
Face Space.

### Long Prompts and Generations 

`mlx-lm` has some tools to scale efficiently to long prompts and generations:

- A rotating fixed-size key-value cache.
- Prompt caching

To use the rotating key-value cache pass the argument `--max-kv-size n` where
`n` can be any integer. Smaller values like `512` will use very little RAM but
result in worse quality. Larger values like `4096` or higher will use more RAM
but have better quality.

Caching prompts can substantially speedup reusing the same long context with
different queries. To cache a prompt use `mlx_lm.cache_prompt`. For example:

```bash
cat prompt.txt | mlx_lm.cache_prompt \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --prompt - \
  --prompt-cache-file mistral_prompt.safetensors
``` 

Then use the cached prompt with `mlx_lm.generate`:

```
mlx_lm.generate \
    --prompt-cache-file mistral_prompt.safetensors \
    --prompt "\nSummarize the above text."
```

The cached prompt is treated as a prefix to the supplied prompt. Also notice
when using a cached prompt, the model to use is read from the cache and need
not be supplied explicitly.

Prompt caching can also be used in the Python API in order to avoid
recomputing the prompt. This is useful in multi-turn dialogues or across
requests that use the same context. See the
[example](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/chat.py)
for more usage details.

### Supported Models

`mlx-lm` supports thousands of LLMs available on the Hugging Face Hub. If the
model you want to run is not supported, file an
[issue](https://github.com/ml-explore/mlx-lm/issues/new) or better yet, submit
a pull request. Many supported models are available in various quantization
formats in the [MLX Community](https://huggingface.co/mlx-community) Hugging
Face organization.

For some models the tokenizer may require you to enable the `trust_remote_code`
option. You can do this by passing `--trust-remote-code` in the command line.
If you don't specify the flag explicitly, you will be prompted to trust remote
code in the terminal when running the model. 

Tokenizer options can also be set in the Python API. For example:

```python
model, tokenizer = load(
    "qwen/Qwen-7B",
    tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True},
)
```

### Server

`mlx-lm` includes an OpenAI-compatible HTTP server:

```bash
mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --port 8080
```

The server supports the following endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat Completions API |
| `POST /v1/completions` | Text Completions API |
| `POST /v1/responses` | **Responses API** (new) |

#### Responses API

The `/v1/responses` endpoint implements the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses), enabling compatibility with clients that require this format (e.g., [Codex CLI](https://github.com/openai/codex)).

Features:
- Translates Responses API requests to the internal Chat Completions pipeline
- Supports both streaming (SSE) and non-streaming responses
- Handles native tool calls via the model's tool parser
- Fallback parsing for models that emit tool calls as plain text JSON
- Consolidates system/developer messages at the start of the conversation
- Filters unsupported tool types (web_search, image_generation, namespace)

**Example usage with Codex CLI:**

```toml
# ~/.codex/config.toml
model_provider = "mlx-local"
model = "mlx-community/Qwen3.6-35B-A3B-4bit"

[model_providers.mlx-local]
name = "MLX Local"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
```

**Server options for optimal performance:**

```bash
mlx_lm.server \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 8080 \
  --max-tokens 8192 \
  --prompt-cache-size 10 \
  --prefill-step-size 4096
```

| Option | Default | Description |
|--------|---------|-------------|
| `--max-tokens` | 512 | Maximum tokens to generate per response |
| `--prompt-cache-size` | 1 | Number of KV caches to keep (increase for multiple sessions) |
| `--prefill-step-size` | 2048 | Tokens processed per prefill step (increase for faster prompt processing) |

### Large Models

> [!NOTE]
    This requires macOS 15.0 or higher to work.

Models which are large relative to the total RAM available on the machine can
be slow. `mlx-lm` will attempt to make them faster by wiring the memory
occupied by the model and cache. This requires macOS 15 or higher to
work.

If you see the following warning message:

> [WARNING] Generating with a model that requires ...

then the model will likely be slow on the given machine. If the model fits in
RAM then it can often be sped up by increasing the system wired memory limit.
To increase the limit, set the following `sysctl`:

```bash
sudo sysctl iogpu.wired_limit_mb=N
```

The value `N` should be larger than the size of the model in megabytes but
smaller than the memory size of the machine.
