# GLM-5.2 Prefill Benchmark And Batching Notes

This fork now includes a lightweight prefill benchmark:

```sh
uv run \
  --python /opt/homebrew/bin/python3.12 \
  --with 'mlx>=0.31.2' --with numpy --with 'transformers>=5.7.0' \
  --with sentencepiece --with protobuf --with pyyaml --with jinja2 \
  --with huggingface_hub \
  python benchmarks/glm52_prefill_benchmark.py \
    --model "$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw" \
    --lengths 128,8192,32768 \
    --repeat-runs 2 \
    --repeat-prefix-tokens 8192 \
    --max-tokens 1 \
    --prefill-step-size 2048 \
    --kv-bits 8 \
    --kv-group-size 64 \
    --quantized-kv-start 4096 \
    --json-output glm52-prefill.json
```

The benchmark reports:

- prompt token count
- tokenization seconds
- time to first token
- TTFT p50/p95 for queued runs
- prompt tokens/sec
- request wait p50/p95 before admission for queued runs
- active batch cache kind and active batch size
- GLM MLA fp/int8 admission counters
- peak MLX memory
- prompt progress callback count
- prompt checkpoint resolution, lookup time, and cache accounting from debug logs
- controlled LCP fields: requested total tokens, stored prefix tokens, expected
  reused prefix tokens, actual `disk_cached_tokens`, `fresh_prompt_tokens`, and
  `fresh_prefill_tokens`
- policy-sweep fields: `policy_name`, `policy_store_prefix_tokens`,
  `policy_reused_ratio`, and `policy_fresh_prefill_ratio`
- GLM DSA sparse prefill fast-path hits and fallback reasons
- optional GLM DSA stage timings for q projection, KV cache update,
  DSA top-k, latent KV dequantization, latent K/V projection, sparse gather,
  attention, and total prefill

Use `--no-prompt-checkpoint` for cold prefill measurements. Use
`--checkpoint-cache-dir "$(mktemp -d)"` when measuring checkpoint behavior so a
run does not touch the normal `~/.cache/mlx-lm/glm52-local` checkpoint cache.
The server exposes the same option; it maps to the existing
`MLX_LM_PROMPT_CHECKPOINT_CACHE_DIR` environment override and changes only the
prompt checkpoint file location.
Use `--checkpoint-save-exact disabled` or `--no-save-exact-checkpoint` when you
want configured prefix/frontier checkpoints without also saving the final exact
full-prompt checkpoint.

## GLM DSA Sparse Prefill Fast Path

This branch adds an exact, GLM-5.2-specific sparse prefill path for DSA/MLA
layers. The DSA indexer computes `topk_indices` in query/key blocks, and the
attention path uses those indices to avoid building the old multi-token sparse
attention mask and full `(heads, query_length, context_length)` prefill tensors.

The implementation order is intentionally conservative:

- update the MLA latent KV cache using the existing cache classes;
- process query tokens in microbatches;
- gather only the selected top-k latent KV and RoPE K per query microbatch;
- dequantize only selected int8 latent KV when the GLM MLA cache is quantized;
- absorb the non-RoPE query into the MLA latent space, avoiding selected K/V
  projection;
- compute exact sparse attention over selected latent KV and project only the
  output back to value-head space.

The fast path is on by default, but it waits until the effective context reaches
`MLX_LM_GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT` (default 131072) before using exact
sparse attention. This keeps short and early prefill chunks on the faster
dense fallback while retaining the memory-bounded sparse path for longer
contexts. Disable it only for short-context comparison runs with:

```sh
MLX_LM_GLM_DSA_FAST_PREFILL=0 python ...
```

The benchmark exposes the same switch as `--fast-prefill disabled`. The query
microbatch size defaults to 16 and can be tuned with:

```sh
MLX_LM_GLM_DSA_FAST_PREFILL_QUERY_CHUNK=32 python ...
```

The DSA indexer key block defaults to `max(index_topk, 8192)` and can be tuned
with:

```sh
MLX_LM_GLM_DSA_FAST_PREFILL_KEY_BLOCK=8192 python ...
```

The sparse handoff point can be tuned with:

```sh
MLX_LM_GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT=131072 python ...
```

The path falls back to the previous implementation when any safeguard is not
satisfied. Current fallback reasons include:

- `disabled`: `MLX_LM_GLM_DSA_FAST_PREFILL` is false or `--fast-prefill disabled`
  is used;
- `no_topk_indices`: the indexer did not return sparse indices, usually because
  the current effective context is not larger than `index_topk`;
- `decode`: `L == 1`, preserving the existing decode gather path;
- `batch_size_not_one`: batched prompt prefill remains on the old path for now;
- `below_sparse_min_context`: the effective context is still below
  `MLX_LM_GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT`;
- `causal_prefix_shorter_than_topk`: early chunks where the first query token
  has fewer valid causal keys than `index_topk`;
- `unsupported_cache:*`: cache type is not a GLM MLA KV cache supported by the
  sparse path;
- `non_scalar_offset`: cache offsets that cannot be resolved to one value;
- `topk_shape`, `topk_heads`, `topk_rank`, `topk_exceeds_context`, and
  `unsupported_kv_heads`: shape/layout guards.

These reasons are available from
`mlx_lm.models.glm_moe_dsa.get_glm_dsa_prefill_profile()` and are also emitted
by `benchmarks/glm52_prefill_benchmark.py`. Set
`MLX_LM_GLM_DSA_FAST_PREFILL_DEBUG=1` to log decisions from the model code.
When the path is actually used with large `topk` values, the model logs a
one-time warning because 4k GLM-5.2 profiling showed this exact gather path is
currently slower than fallback at `index_topk=2048`.

Use `--prefill-profile` to force synchronized stage timings. This adds overhead
but reports:

- `glm_dsa_q_projection_seconds`
- `glm_dsa_kv_cache_update_seconds`
- `glm_dsa_dsa_indexer_topk_seconds`
- `glm_dsa_latent_kv_dequantization_seconds`
- `glm_dsa_latent_kv_projection_seconds`
- `glm_dsa_sparse_gather_seconds`
- `glm_dsa_attention_seconds`
- `glm_dsa_total_prefill_seconds`

## Longest-Prefix Checkpoint Reuse

Practical TTFT improvement now comes from prompt-prefix reuse rather than exact
sparse gather prefill. The checkpoint path is a safe longest-common-prefix
lookup over trusted local GLM-5.2 prompt checkpoint files:

- checkpoint filenames are derived from `sha256(json_tokens)` and token length;
- the manifest records checkpoint files, lengths, kind (`exact`, `prefix`,
  `frontier`, or `unknown`), size, hit count, and metadata identity hashes;
- lookup scans manifest entries by descending prefix length, hashes the request
  tokens up to each candidate length, and returns only exact token-prefix hash
  matches;
- `generate_step` tries longest candidates first, then falls back to shorter
  candidates if loading or validation rejects a longer candidate;
- `load_prompt_checkpoint` validates namespace, model metadata, tokenizer
  metadata, cache signature, GLM DSA metadata, GLM MLA KV quantization state, and
  requested `kv_bits`/`kv_group_size`/`quantized_kv_start` settings before reuse;
- exact full-prompt hits replay the last token from the checkpointed prompt so
  generated logits match normal prefill semantics;
- partial prefix hits restore KV/DSA state from the cached prefix and prefill
  only the suffix;
- token mismatches, incompatible model/cache signatures, malformed checkpoints,
  missing files, or unsupported cache settings fall back to normal prefill.

There is no fuzzy matching: a checkpoint is reused only when the candidate token
sequence is an identical prefix of the request and all cache/model settings
validate.

Run separate commands to keep the main effects distinct:

```sh
MODEL="$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw"

# Unquantized single request, checkpoint disabled:
# isolates normal prefill with no checkpoint reuse and no KV quantization.
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --lengths 4096,8192 --max-tokens 1 \
  --fast-prefill disabled --prefill-profile \
  --no-prompt-checkpoint

# Fast sparse prefill enabled, fp cache, checkpoint disabled:
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --lengths 4096,8192 --max-tokens 1 \
  --fast-prefill enabled --fast-prefill-query-chunk 16 \
  --prefill-profile --no-prompt-checkpoint

# Fast sparse prefill disabled, int8 GLM MLA cache, checkpoint disabled:
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --lengths 4096,8192 --max-tokens 1 \
  --fast-prefill disabled --prefill-profile \
  --kv-bits 8 --kv-group-size 64 --quantized-kv-start 4096 \
  --no-prompt-checkpoint

# Fast sparse prefill enabled, int8 GLM MLA cache, checkpoint disabled:
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --lengths 4096,8192 --max-tokens 1 \
  --fast-prefill enabled --fast-prefill-query-chunk 16 \
  --prefill-profile \
  --kv-bits 8 --kv-group-size 64 --quantized-kv-start 4096

# Full miss, fp cache, isolated checkpoint directory.
MISS_DIR="$(mktemp -d)"
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --lengths 8192 --repeat-prefix-tokens 0 \
  --repeat-runs 1 --max-tokens 1 --prefill-step-size 2048 \
  --fast-prefill disabled \
  --checkpoint-cache-dir "$MISS_DIR" \
  --json-output glm52-full-miss.json

# Exact hit, fp cache. The first repeat is a miss that saves the exact
# checkpoint; the second repeat should report checkpoint_resolution=exact.
EXACT_DIR="$(mktemp -d)"
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --lengths 8192 --repeat-runs 2 \
  --repeat-prefix-tokens 0 --max-tokens 1 \
  --prefill-step-size 2048 --fast-prefill disabled \
  --checkpoint-cache-dir "$EXACT_DIR" \
  --json-output glm52-exact-hit.json

# Controlled LCP disabled baseline. This emits comparable controlled-lcp rows
# with expected_reused_prefix_tokens=0, disk_cached_tokens=0, and
# checkpoint_resolution=disabled.
LCP_DISABLED_DIR="$(mktemp -d)"
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --mode controlled-lcp \
  --lcp-prefix-tokens 8192 --lcp-suffix-tokens 2048 \
  --max-tokens 1 --prefill-step-size 2048 \
  --fast-prefill disabled \
  --checkpoint-cache-dir "$LCP_DISABLED_DIR" \
  --no-prompt-checkpoint \
  --json-output glm52-lcp-disabled-8192-2048.json

# Controlled 6144-token prefix reuse plus a 2048-token fresh suffix. This mode
# stores only the configured prefix checkpoint, then fails if disk_cached_tokens
# does not equal expected_reused_prefix_tokens.
LCP_6144_DIR="$(mktemp -d)"
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --mode controlled-lcp \
  --lcp-prefix-tokens 6144 --lcp-suffix-tokens 2048 \
  --max-tokens 1 --prefill-step-size 2048 \
  --fast-prefill disabled \
  --checkpoint-cache-dir "$LCP_6144_DIR" \
  --checkpoint-save-exact disabled \
  --json-output glm52-lcp-6144-2048.json

# Controlled 8192-token prefix reuse plus a 2048-token fresh suffix.
LCP_8192_DIR="$(mktemp -d)"
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --mode controlled-lcp \
  --lcp-prefix-tokens 8192 --lcp-suffix-tokens 2048 \
  --max-tokens 1 --prefill-step-size 2048 \
  --fast-prefill disabled \
  --checkpoint-cache-dir "$LCP_8192_DIR" \
  --checkpoint-save-exact disabled \
  --json-output glm52-lcp-8192-2048.json

# Controlled prefix reuse with GLM MLA int8 KV cache.
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

# Policy sweep for ds4-style checkpoint tuning. Each candidate gets an isolated
# checkpoint directory under POLICY_DIR. The default candidates compare disabled
# baseline, the derived ds4 boundary, and the full shared prefix when distinct.
POLICY_DIR="$(mktemp -d)"
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --mode policy-sweep \
  --lcp-prefix-tokens 8192 --lcp-suffix-tokens 2048 \
  --max-tokens 1 --prefill-step-size 2048 \
  --fast-prefill disabled \
  --checkpoint-cache-dir "$POLICY_DIR" \
  --json-output glm52-policy-sweep-8192-2048.json

# Explicit policy candidates use name:length. Length 0 is the disabled baseline.
POLICY_CUSTOM_DIR="$(mktemp -d)"
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --mode policy-sweep \
  --lcp-prefix-tokens 8192 --lcp-suffix-tokens 2048 \
  --policy-candidates disabled:0,prefix-6144:6144,prefix-8192:8192 \
  --max-tokens 1 --prefill-step-size 2048 \
  --fast-prefill disabled \
  --checkpoint-cache-dir "$POLICY_CUSTOM_DIR" \
  --json-output glm52-policy-custom-8192-2048.json

# Queued serving workload with repeated coding-agent prefixes:
# exercises admission waiting and the conservative fp/int8 guard. Use
# --max-tokens > 1 so active quantized decode batches stay alive long enough for
# queued fresh fp requests to encounter the guard.
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --mode queued --batch-size 2 \
  --prefill-batch-size 2 --completion-batch-size 4 \
  --queued-requests 8 --queued-prefix-tokens 4096 \
  --queued-suffix-tokens 128,512,2048 \
  --max-tokens 8 --prefill-step-size 2048 \
  --fast-prefill enabled --prefill-profile \
  --kv-bits 8 --kv-group-size 64 --quantized-kv-start 4096 \
  --no-prompt-checkpoint \
  --json-output glm52-queued-int8.json
```

Policy sweep JSON contains store/hit pairs for every candidate. Prefer the
smallest candidate that keeps `checkpoint_expected_match` true and leaves
`fresh_prefill_tokens` within the latency budget. For coding-agent continuation
workloads, the ds4-style default generally means:

- keep prefix checkpoints aligned to stable boundaries instead of exact prompt
  tails;
- keep continued checkpoints at the rounded interval boundary;
- avoid storing many nearby variants that differ only in the last unstable
  tokens;
- use `checkpoint_lookup_seconds` and `checkpoint_manifest_entries` to check
  whether the manifest is becoming the bottleneck.

Controlled LCP JSON contains both the store run and measured hit run. The hit
row should have `checkpoint_expected_match: true`, and the actual disk hit
length should equal the expected prefix:

```json
{
  "case": "controlled-lcp-hit",
  "requested_total_tokens": 8192,
  "stored_prefix_tokens": 6144,
  "expected_reused_prefix_tokens": 6144,
  "disk_cached_tokens": 6144,
  "fresh_prompt_tokens": 2048,
  "fresh_prefill_tokens": 2047,
  "checkpoint_resolution": "prefix",
  "checkpoint_expected_match": true
}
```

For the 8192+2048 int8 case, the expected accounting is:

```json
{
  "case": "controlled-lcp-hit",
  "requested_total_tokens": 10240,
  "stored_prefix_tokens": 8192,
  "expected_reused_prefix_tokens": 8192,
  "disk_cached_tokens": 8192,
  "fresh_prompt_tokens": 2048,
  "fresh_prefill_tokens": 2047,
  "checkpoint_resolution": "prefix",
  "checkpoint_expected_match": true,
  "checkpoint_save_exact": "disabled"
}
```

## Current Implementation Baseline

Before this change, the server deliberately rejected continuous batching whenever `--kv-bits` was set. That kept GLM MLA int8 KV correct, but forced long-context GLM-5.2 requests through the single-request `stream_generate` path.

This pass adds batch-compatible GLM MLA int8 cache support:

- `GlmMlaKVCache.merge(...)` now produces `BatchGlmMlaKVCache`.
- Batched GLM MLA caches can convert to `BatchQuantizedGlmMlaKVCache`.
- `BatchQuantizedGlmMlaKVCache` supports `merge`, `filter`, `extend`, `extract`, `trim`, `prepare`, and `finalize`.
- `BatchGenerator` now carries `kv_bits`, `kv_group_size`, and `quantized_kv_start` through prompt and decode batches.
- The server allows continuous batching for GLM MLA `--kv-bits 8`; non-GLM `--kv-bits` requests still use the single-request path.

The batch path intentionally does not merge quantized and unquantized GLM MLA caches in one active batch. This preserves `quantized_kv_start` semantics for new short requests instead of silently quantizing them early. Such requests wait until an active incompatible batch drains. The scheduler scans beyond a waiting incompatible request and may admit later compatible requests into the active batch; responses remain keyed per request, so this does not change output text semantics or each request's token order.

The benchmark-visible admission fields are:

- `glm_mla_quantized_batch_admitted`: requests admitted into or alongside a quantized GLM MLA batch.
- `glm_mla_quantized_batch_rejected_mixed_cache`: admission attempts skipped because fp and int8 GLM MLA caches would mix.
- `glm_mla_waited_for_compatible_batch`: unique requests that had to wait for a compatible active batch.
- `active_batch_cache_kind`: `none`, `fp`, `quantized`, or `mixed` for live prompt/decode caches.
- `active_batch_size`: live prompt plus decode requests at the measurement point.
- `queued_request_count`: requests still waiting for admission.

## Recommended GLM-5.2 Settings

For latency-focused long-context coding-agent workloads on this fork, especially
when 200K+ token prompts are common:

```sh
MLX_LM_PROMPT_CHECKPOINT_DEBUG=1 \
MLX_METAL_FAST_SYNCH=1 \
MLX_LM_GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT=131072 \
python -m mlx_lm server \
  --model "$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw" \
  --host 0.0.0.0 \
  --port 8000 \
  --kv-bits 8 \
  --kv-group-size 64 \
  --quantized-kv-start 4096 \
  --prefill-step-size 1024 \
  --prefill-max-qk-tokens 67108864 \
  --checkpoint-cache-dir /Volumes/USB-SSD-2/mlx-lm-glm52-local/prompt-checkpoints \
  --checkpoint-min-tokens 512 \
  --checkpoint-cold-max-tokens 30000 \
  --checkpoint-boundary-trim-tokens 32 \
  --checkpoint-boundary-align-tokens 2048 \
  --checkpoint-continued-interval-tokens 10000 \
  --checkpoint-shutdown-save-limit 4 \
  --checkpoint-max-age-seconds 0 \
  --prompt-concurrency 1 \
  --decode-concurrency 1 \
  --disable-batching \
  --loop-guard-ngram-size 64 \
  --loop-guard-repeats 3 \
  --loop-guard-min-tokens 256
```

This keeps requests on the single-request checkpoint path, which is the lowest
TTFT path for repeated or partially reused long prompts. Use
`--prefill-step-size 2048` only after checking peak memory and Metal stability
on your real prompt distribution. `--prefill-max-qk-tokens` keeps dense fallback
chunks below the configured query-by-context budget and can be set to `0` to
disable context-aware step shrinking.

The checkpoint defaults above are the current ds4-style policy: save stable
boundaries rather than unstable tails, round continued checkpoints to a roughly
10K-token interval, and preserve a few live RAM frontiers on shutdown. Set
`--checkpoint-max-age-seconds` only after measuring real cache hit windows; the
default keeps age eviction off and lets file/byte budgets control pruning.

The loop guard is intentionally a decode-time fuse, not a sampling replacement:
it stops exact repeated token n-grams after the configured minimum generated
token count. If a client can pass request parameters, combine it with conservative
sampling and a small `repetition_penalty` for prompts that still produce
near-duplicate reasoning loops.

For shorter mixed workloads where throughput matters more than per-request TTFT,
continuous batching remains available:

```sh
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

## Validation In This Pass

Unit validation covered:

- exact GLM DSA fast prefill output closeness against the fallback path
- exact GLM DSA fast prefill output closeness with `QuantizedGlmMlaKVCache`
- sparse prefill waiting until the configured minimum effective context
- sparse prefill falling back while the causal prefix is shorter than top-k
- `L == 1` decode staying on the existing decode path
- unsupported batched prompt prefill falling back instead of crashing
- prompt checkpoint token-prefix mismatch staying a full miss
- server CLI defaults for context-aware prefill chunk shrinking and decode loop
  guard options
- incompatible GLM DSA checkpoint metadata being rejected before reuse
- isolated prompt-checkpoint cache directories via `--checkpoint-cache-dir`
- storing prefix checkpoints while skipping final exact checkpoint saves
- packed quantized GLM MLA cache merge/filter/extend/extract behavior
- conversion from batched floating GLM MLA cache to batched int8 cache
- server batchability routing for GLM MLA `--kv-bits 8`
- tiny GLM DSA `BatchGenerator(..., kv_bits=8)` smoke generation

Local 4k GLM-5.2 measurements on
`avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw`, fp cache,
`--prefill-step-size 2048`, `--max-tokens 1`, and
`--no-prompt-checkpoint`:

| Fast path | Query chunk | Profile | TTFT seconds | Prompt tok/s | Peak GB | Fast hits | Total prefill s | Sparse gather s | Attention s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| disabled | n/a | yes | 25.7184 | 159.6563 | 331.6076 | 0 | 25.3178 | 0.0000 | 4.4312 |
| enabled | 16 | yes | 109.0962 | 37.5676 | 333.1855 | 78 | 108.6823 | 40.8803 | 46.4829 |
| enabled | 16 | no | 98.2606 | 41.7132 | 367.1439 | 78 | n/a | n/a | n/a |
| enabled | 32 | no | 98.4812 | 41.6202 | 404.9708 | 78 | n/a | n/a | n/a |

`--fast-prefill-query-chunk 128` ran out of Metal memory on the second prefill
chunk. The old dense fallback can still be useful for short-context comparison
runs, but long-context serving should keep the memory-bounded sparse path
enabled. The profile shows the new dominant costs are selected K/V gather and
per-query sparse attention, not latent KV projection or dequantization.

Local short GLM-5.2 benchmark smoke with
`avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw`, `--lengths 128`,
`--max-tokens 1`, `--kv-bits 8`, `--kv-group-size 64`, and
`--quantized-kv-start 4096`:

| Run | Prompt tokens | Checkpoint | TTFT seconds | Prompt tok/s | Peak memory GB |
| --- | ---: | --- | ---: | ---: | ---: |
| cold synthetic-128 | 128 | miss | 4.4570 | 29.1685 | 328.9001 |
| repeat synthetic-128 | 128 | exact | 3.1597 | 41.3816 | 328.5258 |
| batched synthetic-128 x2 | 256 total | batch-n/a | 5.0365 | 53.1160 | 329.1723 |

Short queued GLM-5.2 smoke, using the same local model with
`--mode queued`, `--batch-size 2`, `--prefill-batch-size 2`,
`--completion-batch-size 4`, `--queued-requests 4`,
`--queued-prefix-tokens 64`, `--queued-suffix-tokens 16,32`,
`--max-tokens 8`, `--prefill-step-size 64`, `--kv-bits 8`, and
`--quantized-kv-start 0`:

| Prompt tokens | TTFT p50 | TTFT p95 | Prompt tok/s | Peak GB | Wait p50 | Wait p95 | Mixed-cache rejections | Unique waited |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 528 | 7.4045 | 9.1247 | 63.5999 | 328.9791 | 3.1213 | 6.2274 | 14 | 2 |

This smoke forces early quantization so the guard is visible at short prompt
lengths. Full 8k/32k and 4096-token queued-prefix benchmark numbers should
still be regenerated on the target machine because they depend heavily on memory
pressure, prompt checkpoint state, and concurrent server load.

## Remaining Bottlenecks

Based on the 4k profile above, the next useful work should prioritize:

- a fused or native MLX sparse-attention kernel that accepts query-wise top-k
  indices without materializing `(B, H, L_micro, topk, D)` K/V tensors;
- reducing DSA indexer `q @ k` work, since the current exact path still computes
  full indexer scores before top-k;
- selective latent KV dequantization only after the sparse kernel problem is
  solved, because the measured fp-cache bottleneck is gather/attention rather
  than dequantization;
- capturing server chat-template/tokenization cache hit rates for repeated
  coding-agent request prefixes;
- keeping the conservative scheduler policy for mixed fp/int8 GLM MLA batches.
