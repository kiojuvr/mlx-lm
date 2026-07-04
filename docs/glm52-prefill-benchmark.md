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
    --prefill-step-size 8192 \
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
- configured `prefill_max_qk_tokens` and GLM DSA adaptive prefill knobs, plus
  observed maximum adaptive/effective prefill chunk sizes from debug logs
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

The Python selected-KV sparse path is on by default, but it waits until the
effective context reaches one token below
`MLX_LM_GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT` (default 131072) before using exact
sparse attention. Generation prefill leaves the final prompt token for logits,
so a 131072-token prompt can expose at most 131071 tokens to the attention call.
This keeps short and early prefill chunks on the faster dense fallback while
retaining the memory-bounded sparse path for longer contexts. Disable all GLM
DSA sparse/native prefill routing only for short-context comparison runs with:

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

When the vendored native sparse MLA symbol is available, the model can try a
deeper native route before the Python sparse handoff. It is enabled by default
but only activates for the fixed M3 GLM shape currently supported by the
vendored kernel:
64 heads, latent dim 512, RoPE dim 64, top-k 2048, unquantized `GlmMlaKVCache`,
and an effective context at or above
`MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_MIN_CONTEXT` (default 6144). Quantized
GLM MLA KV cache remains on the existing selected-KV sparse path unless the
explicit quantized-KV native sparse option below is enabled.

The benchmark exposes `--native-sparse-prefill enabled|disabled|default` and
`--native-sparse-prefill-min-context`. It also reports native availability,
symbol source, import error, hit count, and fallback reasons so a run can
distinguish "native extension missing" from "shape or cache guard rejected".
On the tested M3 Ultra setup, the default handoff was moved from 11264 to 6144.
The first sweep moved it to 8192, improving the 16K profile from about 117s to
about 111s while forcing 4096 made the 8K profile slower. The later chunk-route
sweep favored 6144 over 8192: 8K prefill-stop improved from about 55.85s to
53.72s, and 16K prefill-stop improved from about 110.28s to 109.90s. Delaying
the handoff to 10240 regressed the 16K run to about 112.15s.
With the 6144-token handoff and the QK cap left at 67108864, raising the base
prefill step from 2048 to 4096 reduced 16K prefill-stop from about 109.90s to
105.44s and 32K prefill-stop from about 225.18s to 219.63s. Raising the base
step again to 8192 reduced 16K to 104.11s, 32K to 207.10s, 64K to 436.05s, and
128K to 963.13s. The 8192-token first chunk crosses the native sparse handoff
immediately, so the measured runs stayed on the native sparse MLA route with no
early dense chunk; the QK cap shrank later chunks automatically. On the 128K
run, q projection dropped from about 550.84s at 4096 to about 434.19s at 8192,
native sparse attention dropped from about 310.29s to 297.42s, and native
indexer top-k dropped from about 77.43s to 75.21s. Peak memory stayed about
336.41GB at 128K and rose by about 0.53GB at 16K through 64K compared with the
4096-step runs.

The 8192-step setting also held up when the native sparse quantized-KV guard was
extended beyond 128K. On the tested Mac Studio M3 Ultra 512GB setup, 160K cold
prefill measured about 1302.12s with 219 native chunks and 0 dense chunks,
196608 tokens measured about 1646.78s with 311 native chunks and 0 dense
chunks, and 204800 tokens measured about 1739.78s with 336 native chunks and 0
dense chunks. The 204800 run peaked at about 340.64GB and reported about
730.91s in q projection, 491.57s in native sparse attention, 196.14s in native
indexer top-k, and 27.83s in native sparse KV dequantization.

For latency experiments with `--kv-bits 8`, the native sparse MLA route can be
enabled over int8 GLM MLA KV cache with:

```sh
--native-sparse-quantized-kv enabled \
--native-sparse-quantized-kv-max-context 262144
```

This keeps the persistent cache quantized but temporarily dequantizes the full
latent KV tensor for the native sparse MLA kernel. The route is disabled by
default because the transient full-cache dequantization trades memory for
latency; keep the max-context guard bounded until the target prompt length is
profiled.

Benchmark rows report:

- `glm_dsa_native_sparse_prefill_quantized_kv`
- `glm_dsa_native_sparse_prefill_quantized_kv_env`
- `glm_dsa_native_sparse_prefill_quantized_kv_max_context`

The same vendored extension can also accelerate the DSA indexer score/top-k
stage before sparse MLA attention is selected. This route is enabled by default
when `MLX_LM_GLM_DSA_NATIVE_INDEXER` is unset, and can be controlled with:

```sh
MLX_LM_GLM_DSA_NATIVE_INDEXER=0 python ...
```

or in the benchmark with `--native-indexer enabled|disabled|default`. The
current native indexer route is intentionally limited to the fixed GLM-5.2 M3
shape: 32 DSA indexer heads, head dim 128, top-k 2048, batch size 1, fp16/bf16
inputs, and effective context at or above 4096. It remains compatible with GLM
MLA int8 KV cache because it reads the separate DSA indexer cache rather than
the quantized MLA latent KV cache.

Benchmark rows report:

- `glm_dsa_native_indexer`
- `glm_dsa_native_indexer_available`
- `glm_dsa_native_indexer_source`
- `glm_dsa_native_indexer_scores_available`
- `glm_dsa_native_indexer_topk_available`
- `glm_dsa_native_indexer_hits`
- `glm_dsa_native_indexer_fallback_reasons`

Before a long model run, use the native smoke mode to check the self-contained
extension and compare the tiny native sparse MLA output against a dense MLX
reference. This mode does not load the GLM-5.2 model:

```sh
python benchmarks/glm52_prefill_benchmark.py \
  --mode native-smoke \
  --json-output glm52-native-smoke.json
```

The expected result is `native_smoke_passed=True` with
`native_smoke_source='mlx_lm.custom_kernels.glm_moe_dsa'`. The same run also
checks native DSA indexer score/top-k; expect
`native_indexer_smoke_passed=True`. It also checks the native q8 V-up projection
for quantized GLM DSA `unembed_out` weights; expect
`native_q8_vup_smoke_passed=True`.

Add timing runs to compare native q8 V-up with the MLX `quantized_matmul`
reference without loading the GLM-5.2 model:

```sh
python benchmarks/glm52_prefill_benchmark.py \
  --mode native-smoke \
  --native-smoke-benchmark-runs 20 \
  --native-q8-vup-benchmark-q-len 256 \
  --json-output glm52-native-smoke-bench.json
```

The timing fields are:

- `native_q8_vup_native_seconds_mean`
- `native_q8_vup_reference_seconds_mean`
- `native_q8_vup_speedup_mean`

For full benchmark runs, the native route diagnostics are:

- `glm_dsa_native_sparse_prefill_route_state`
- `glm_dsa_native_sparse_prefill_primary_fallback`
- `glm_dsa_native_sparse_prefill_config_blocker`
- `glm_dsa_native_sparse_prefill_attempt_min_context`

With the usual long-context memory-saving configuration
`--kv-bits 8 --quantized-kv-start 4096`, expect
`glm_dsa_native_sparse_prefill_config_blocker=quantized_kv_at_native_threshold`.
That means the extension is loaded, but the current native sparse MLA route is
not used for those chunks because the GLM MLA KV cache becomes int8 before the
native threshold.

The V-up routes are independent from sparse MLA. When `unembed_out` is a
quantized affine `QuantizedMultiLinear` with the fixed GLM-5.2 M3 shape
64 heads, latent dim 512, value dim 256, group size 64, the model can use
vendored native kernels for the latent-to-value projection. The q8 route uses
`glm_dsa_q8_vup_flat`; the q4 route uses `glm_dsa_q4_vup_flat`. They are opt-in
because the model-free microbench can be slower than MLX `quantized_matmul` on
some lengths, and q4 did not move end-to-end TTFT in the tested 16K native
sparse MLA profile despite hitting the native route.
Benchmark rows report:

- `glm_dsa_native_q8_vup`
- `glm_dsa_native_q8_vup_available`
- `glm_dsa_native_q8_vup_source`
- `glm_dsa_native_q8_vup_hits`
- `glm_dsa_native_q8_vup_fallback_reasons`
- `glm_dsa_native_q4_vup`
- `glm_dsa_native_q4_vup_available`
- `glm_dsa_native_q4_vup_source`
- `glm_dsa_native_q4_vup_hits`
- `glm_dsa_native_q4_vup_fallback_reasons`

Use `--native-q8-vup enabled|disabled|default` and
`--native-q4-vup enabled|disabled|default` to force or disable these routes for
comparison runs. With the default environment, both remain disabled.

The q4 q projection probes are independent from sparse MLA and V-up routes. They
route the fixed GLM-5.2 M3 affine q4 `q_a_proj` and `q_b_proj` calls through
vendored native kernels:

```sh
--native-q4-qa enabled \
--native-q4-qa-tile bk64 \
--native-q4-qb enabled
```

These routes are disabled by default and are currently profiling probes rather
than recommended serving settings. On the tested 8K cold prefill,
`--prefill-profile` showed q projection dominated by q_a projection: about
29.7s q_a projection, 0.18s q_a RMSNorm, and 2.1s q_b projection. Earlier q4
native probes did not improve end-to-end TTFT in that run.

`--native-q4-qa-tile` selects the q_a native q4 tile; default/unset currently
maps to `bk64`. Available tiles are `bk32`, `bk64`, `bm16`, `bn16`, `bn64`,
`bm64`, `bm16bn64`, and `bm64bn64`. The same-build 8K comparison measured
`bk64` at about 52.99s TTFT and 26.41s q_a projection versus about
53.26s TTFT and 26.58s q_a projection with the q_a native q4 route disabled.
`bn64` was best in a short 2K sweep but regressed at 8K to about 53.41s TTFT and
26.71s q_a projection.

Use the standalone q_a tile microbench when the full model is already resident
or when only the projection kernel is under investigation:

```sh
python benchmarks/glm52_q4_qa_tile_microbench.py \
  --q-len 8192 \
  --runs 5 \
  --warmup-runs 1 \
  --json-output /path/to/glm52-q4qa-tile-microbench-8192.json
```

This synthetic benchmark compares the fixed GLM-5.2 M3 q_a shape against
`mx.quantized_matmul` without loading model weights. On the tested 8K synthetic
run, `mx.quantized_matmul` measured about 0.00905s best / 0.00932s mean;
native q4 q_a measured about 0.00860s / 0.00920s for `bk32`, 0.00880s /
0.00901s for `bk64`, and slower means for the newly exposed `bm16`, `bn16`,
`bm16bn64`, and `bm64bn64` variants. This points to only a small tile-level
margin, so q_a-side gains likely need a larger change than tile selection.

The native q4 q_b probe has the same style of selector:
`MLX_LM_GLM_DSA_NATIVE_Q4_QB_TILE` or `--native-q4-qb-tile`. Unset/default maps
to `bm64`; available tiles are `bk32`, `bk64`, `bm16`, `bn16`, `bn64`, `bm64`,
`bm16bn64`, `bm64bn64`, and `bk64bn64`. Use the standalone q_b tile microbench
for low-memory sweeps:

```sh
python benchmarks/glm52_q4_qb_tile_microbench.py \
  --q-len 8192 \
  --runs 5 \
  --warmup-runs 1 \
  --json-output /path/to/glm52-q4qb-tile-microbench-8192.json
```

On the tested synthetic 8192 run, `bm64` measured about 0.02203s best /
0.02236s mean versus about 0.02245s / 0.02285s for the previous `bk32`
template. The margin is small, but `bm64` is the better long-query default for
the opt-in native q4 q_b route.

For memory-for-latency comparison runs, `--q-a-dense-cache enabled` can
dequantize the fixed GLM-5.2 M3 q4 `q_a_proj` weights into dense fp16/bf16
matrices on first use and reuse them for later prefill calls in the same model
process. This route is disabled by default because it adds roughly 1.5-2GB of
resident memory across the full model and is only useful if warmed q_a
projection time improves enough to justify that footprint. On the tested 8K
repeat run, warmed dense-cache prefill was 51.46s versus 51.51s with the route
disabled, while peak memory increased by about 1.96GB.

Benchmark rows report:

- `glm_dsa_q_a_dense_cache`
- `glm_dsa_q_a_dense_cache_env`
- `glm_dsa_q_a_dense_cache_hits`
- `glm_dsa_q_a_dense_cache_builds`
- `glm_dsa_q_a_dense_cache_fallback_reasons`
- `glm_dsa_native_q4_qa`
- `glm_dsa_native_q4_qa_env`
- `glm_dsa_native_q4_qa_available`
- `glm_dsa_native_q4_qa_source`
- `glm_dsa_native_q4_qa_tile_env`
- `glm_dsa_native_q4_qa_tile`
- `glm_dsa_native_q4_qa_hits`
- `glm_dsa_native_q4_qa_fallback_reasons`
- `glm_dsa_native_q4_qb`
- `glm_dsa_native_q4_qb_available`
- `glm_dsa_native_q4_qb_source`
- `glm_dsa_native_q4_qb_tile_env`
- `glm_dsa_native_q4_qb_tile`
- `glm_dsa_native_q4_qb_hits`
- `glm_dsa_native_q4_qb_fallback_reasons`

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
- native indexer fallback reasons include `missing_symbol`,
  `below_native_indexer_min_context`, `unsupported_index_heads:*`,
  `unsupported_index_head_dim:*`, `unsupported_topk:*`, `unsupported_dtype:*`,
  `mixed_dtype`, `mixed_weight_dtype`, `scores_unavailable`, and
  `topk_unavailable`.
- native sparse MLA over int8 KV can additionally report
  `batched_quantized_kv_cache`, `quantized_kv_context_exceeds_limit`,
  `unsupported_kv_bits:*`, and `unsupported_kv_group_size:*`.
- q_a dense-cache fallback reasons include `disabled`,
  `unquantized_q_a_proj`, `unsupported_bits:*`, `unsupported_group_size:*`,
  `unsupported_mode:*`, `missing_biases`, `unsupported_input_dim:*`,
  `unsupported_q_lora_rank:*`, `unsupported_weight_dtype:*`,
  `unsupported_dtype:*`, `mixed_dtype`, shape guard failures, and
  `runtime_error:*`.

These reasons are available from
`mlx_lm.models.glm_moe_dsa.get_glm_dsa_prefill_profile()` and are also emitted
by `benchmarks/glm52_prefill_benchmark.py`. Set
`MLX_LM_GLM_DSA_FAST_PREFILL_DEBUG=1` to log decisions from the model code.
When the path is actually used with large `topk` values, the model logs a
one-time warning because 4k GLM-5.2 profiling showed this exact gather path is
currently slower than fallback at `index_topk=2048`.

Use `--prefill-profile` to force synchronized stage timings. This adds overhead.
Because MLX evaluates lazily, a stage can otherwise inherit unfinished work from
earlier expressions when its output is synchronized. Add
`--prefill-profile-isolate enabled` for q_projection investigation; it evaluates
the selected q_projection inputs before timing q_a, q_a RMSNorm, and q_b so the
sub-stage attribution is less likely to charge upstream work to q_a projection.
On the tested 2K synthetic prefill, normal profiling reported about 9.18s in
q_a projection, but isolated profiling reported about 0.23s q_a projection,
0.03s q_a RMSNorm, 0.51s q_b projection, and 0.78s q_projection total. Treat
non-isolated q_projection sub-stage timings as coarse synchronization markers,
not literal kernel time.
After extending the same isolated-input handling to indexer and attention
stages, an 8K native sparse profile measured about 53.27s TTFT, 2.75s
q_projection, 0.49s DSA indexer top-k, 0.28s native indexer scores, 0.04s
native indexer top-k, 1.02s latent projection, and 16.87s native sparse MLA
attention. That makes native sparse MLA attention the next meaningful
attention-side optimization target; the indexer path is no longer the dominant
cost at this length.

For sparse MLA tile experiments, use
`MLX_LM_GLM_DSA_SPARSE_MLA_TILE` or `--native-sparse-mla-tile`. Unset/default
maps to `bk256_dc32_wm8`; available aliases are `bk128`, `bk256`,
`bk128_dc64`, `wm4`, `bk128_wm4`, and `bk128_dc64_wm4`. The standalone
model-free microbench checks a small dense reference and then times larger
synthetic sparse MLA shapes:

```sh
python benchmarks/glm52_sparse_mla_tile_microbench.py \
  --q-len 512 \
  --k-len 8192 \
  --topk 2048 \
  --runs 5 \
  --json-output /path/to/glm52-sparse-mla-tile-microbench-512.json
```

Treat this as an experimental measurement knob until full-prompt profiles show
a stable win on the target prompt length. Initial synthetic sweeps at
512x8192/topk2048 and 2048x16384/topk2048 still favored the default `bk256`
tile.
Benchmark rows report:

- `glm_dsa_prefill_profile`
- `glm_dsa_prefill_profile_isolate`
- `glm_dsa_prefill_profile_isolate_env`
- `glm_dsa_q_projection_seconds`
- `glm_dsa_q_a_projection_seconds`
- `glm_dsa_q_a_dense_cache_dequantization_seconds`
- `glm_dsa_q_a_dense_projection_seconds`
- `glm_dsa_native_q4_qa_projection_seconds`
- `glm_dsa_q_a_layernorm_seconds`
- `glm_dsa_q_b_projection_seconds`
- `glm_dsa_native_q4_qb_projection_seconds`
- `glm_dsa_kv_cache_update_seconds`
- `glm_dsa_dsa_indexer_topk_seconds`
- `glm_dsa_native_indexer_scores_seconds`
- `glm_dsa_native_indexer_topk_seconds`
- `glm_dsa_latent_kv_dequantization_seconds`
- `glm_dsa_latent_kv_projection_seconds`
- `glm_dsa_native_sparse_kv_dequantization_seconds`
- `glm_dsa_native_q8_vup_seconds`
- `glm_dsa_native_q4_vup_seconds`
- `glm_dsa_sparse_gather_seconds`
- `glm_dsa_attention_seconds`
- `glm_dsa_native_sparse_attention_seconds`
- `glm_dsa_total_prefill_seconds`
- `glm_dsa_sparse_mla_tile_env`
- `glm_dsa_sparse_mla_tile`

The prompt-checkpoint debug stream now also carries per-prefill-chunk GLM DSA
route deltas. Benchmark JSON rows include `checkpoint_prefill_chunk_summaries`
with each chunk's token range, wall time, sparse route (`native_sparse`,
`fast_sparse`, or `dense`), native indexer hits, sparse MLA hits, fallback
reason deltas, and selected stage-time deltas when `--prefill-profile` is
enabled. Table output summarizes this as:

- `checkpoint_prefill_chunk_seconds_total`
- `checkpoint_max_prefill_chunk_seconds`
- `checkpoint_slowest_prefill_chunk_start_tokens`
- `checkpoint_slowest_prefill_chunk_tokens`
- `checkpoint_slowest_prefill_chunk_route`
- `checkpoint_native_sparse_prefill_chunks`
- `checkpoint_fast_sparse_prefill_chunks`
- `checkpoint_dense_prefill_chunks`
- `checkpoint_native_indexer_chunks`

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

Run separate commands to keep the main effects distinct.

The examples below use `--model "$MODEL"`; confirm `echo "$MODEL"` prints the
model directory before running them.

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
  --prefill-max-qk-tokens 67108864 \
  --glm-dsa-adaptive-prefill-step-size 8192 \
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

# Cold prefill step sweep for GLM DSA tuning. Checkpoints are disabled by
# default so each candidate measures prefill chunking instead of reuse. The
# benchmark writes glm52-prefill-step-sweep.json.partial after each completed
# candidate, so interrupted long-context runs still keep completed rows.
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --mode prefill-sweep \
  --lengths 8192,32768 \
  --max-tokens 1 \
  --prefill-step-candidates 1024,2048,4096,8192 \
  --prefill-max-qk-token-candidates 67108864 \
  --glm-dsa-adaptive-prefill-step-candidates 0,8192 \
  --fast-prefill enabled \
  --json-output glm52-prefill-step-sweep.json

# Probe only the short/medium finalists at 131072 tokens. Replace the candidate
# lists with the winners from the previous run. The min-context candidates tune
# where the sparse path starts; use --prefill-stop-after-tokens first so the
# sweep writes partial rows without waiting for every full 131072-token run.
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --mode prefill-sweep \
  --lengths 131072 \
  --max-tokens 1 \
  --prefill-step-candidates 4096 \
  --prefill-max-qk-token-candidates 67108864 \
  --glm-dsa-adaptive-prefill-step-candidates 0 \
  --fast-prefill-min-context-candidates 98304,114688,131072 \
  --prefill-stop-after-tokens 98304 \
  --fast-prefill enabled \
  --json-output glm52-prefill-step-sweep-131k-finalists.json

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
MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV=1 \
MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_MAX_CONTEXT=262144 \
python -m mlx_lm server \
  --model "$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw" \
  --host 0.0.0.0 \
  --port 8000 \
  --kv-bits 8 \
  --kv-group-size 64 \
  --quantized-kv-start 4096 \
  --prefill-step-size 8192 \
  --prefill-max-qk-tokens 67108864 \
  --glm-dsa-adaptive-prefill-step-size 0 \
  --checkpoint-cache-dir /Volumes/USB-SSD-2/mlx-lm-glm52-local/prompt-checkpoints \
  --checkpoint-min-tokens 512 \
  --checkpoint-cold-max-tokens 30000 \
  --checkpoint-boundary-trim-tokens 32 \
  --checkpoint-boundary-align-tokens 2048 \
  --checkpoint-continued-interval-tokens 10000 \
  --checkpoint-save-exact disabled \
  --checkpoint-shutdown-save-limit 0 \
  --checkpoint-max-age-seconds 0 \
  --prompt-concurrency 1 \
  --decode-concurrency 1 \
  --disable-batching \
  --prefill-progress-interval-tokens 2048 \
  --decode-progress-interval-tokens 512 \
  --loop-guard-ngram-size 64 \
  --loop-guard-repeats 3 \
  --loop-guard-min-tokens 256
```

This keeps requests on the single-request checkpoint path, which is the lowest
TTFT path for repeated or partially reused long prompts. Use
`--prefill-step-size 4096` if 8192 shows Metal recovery or memory pressure on
your real prompt distribution, then 2048 and 1024 if needed.
`--prefill-max-qk-tokens` keeps prefill chunks below the configured
query-by-context budget and can be set to `0` to disable context-aware step
shrinking. A 131072 native quantized-KV guard was profiled through a 128K
prefill-stop run on the tested setup, and the recommended 262144 guard has been
profiled through a 204800-token prefill-stop run with native chunks 336/336 and
dense chunks 0. Raise it beyond 262144 only after profiling the target context
length.

For GLM DSA step-size tuning, compare 4096 and 8192 first, with 2048 as the
memory-pressure fallback. Treat `--glm-dsa-adaptive-prefill-step-size 8192` as a
benchmark-only knob. The adaptive step is GLM-only and still passes through the
`--prefill-max-qk-tokens` cap, so it mainly helps earlier/mid-context prefill
where the QK budget allows a larger chunk.

The checkpoint defaults above are the current ds4-style policy: save stable
boundaries rather than unstable tails, round continued checkpoints to a roughly
10K-token interval, and skip shutdown-time RAM checkpoint flushes by default. For
long-running coding-agent sessions, `--checkpoint-save-exact disabled` avoids
writing large exact full-prompt checkpoints that are often immediately superseded
by RAM/server cache or pruned by the byte budget. The progress intervals keep
long prefill/decode phases visible without requiring checkpoint debug logging. Set
`--checkpoint-max-age-seconds` only after measuring real cache hit windows; the
default keeps age eviction off and lets file/byte budgets control pruning.
If you explicitly enable shutdown saves with `--checkpoint-shutdown-save-limit`,
keep `--checkpoint-shutdown-max-tokens` bounded so Ctrl+C does not spend minutes
serializing a 200K-class prompt cache during process exit.

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
