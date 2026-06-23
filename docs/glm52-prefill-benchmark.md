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
- prompt checkpoint resolution and lookup time from debug logs

Use `--no-prompt-checkpoint` for cold prefill measurements. Keep `--repeat-runs 2` or higher to expose exact and repeated-prefix checkpoint reuse.

Run separate commands to keep the main effects distinct:

```sh
# Unquantized single request, checkpoint disabled:
# isolates normal prefill with no checkpoint reuse and no KV quantization.
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --lengths 128,8192 --max-tokens 1 \
  --no-prompt-checkpoint

# Unquantized continuous batching, checkpoint disabled:
# isolates batching without quantized KV.
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --mode batch --batch-size 2 \
  --lengths 128,8192 --max-tokens 1 \
  --no-prompt-checkpoint

# GLM MLA int8 single-request fallback/emulation, checkpoint disabled:
# isolates quantized KV memory behavior without continuous batching.
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --lengths 128,8192 --max-tokens 1 \
  --kv-bits 8 --kv-group-size 64 --quantized-kv-start 4096 \
  --no-prompt-checkpoint

# GLM MLA int8 BatchGenerator path:
# isolates continuous batching plus quantized KV. Batch mode does not exercise
# disk prompt checkpoints; use single mode for checkpoint miss/exact-hit timings.
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --mode batch --batch-size 2 \
  --lengths 128,8192 --max-tokens 1 \
  --kv-bits 8 --kv-group-size 64 --quantized-kv-start 4096

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
  --kv-bits 8 --kv-group-size 64 --quantized-kv-start 4096 \
  --no-prompt-checkpoint \
  --json-output glm52-queued-int8.json

# Repeat a single-mode run without --no-prompt-checkpoint to measure exact and
# prefix checkpoint reuse separately from batching.
python benchmarks/glm52_prefill_benchmark.py \
  --model "$MODEL" --lengths 8192 --repeat-runs 2 \
  --repeat-prefix-tokens 8192 --max-tokens 1 \
  --kv-bits 8 --kv-group-size 64 --quantized-kv-start 4096
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

For long coding-agent workloads on this fork:

```sh
MLX_METAL_FAST_SYNCH=1 python -m mlx_lm server \
  --model "$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw" \
  --host 0.0.0.0 \
  --port 8000 \
  --kv-bits 8 \
  --kv-group-size 64 \
  --quantized-kv-start 4096 \
  --prefill-step-size 2048
```

Use larger `--prefill-step-size` values only after checking peak memory and TTFT on your workload. The default `2048` is conservative for stability.

## Validation In This Pass

Unit validation covered:

- packed quantized GLM MLA cache merge/filter/extend/extract behavior
- conversion from batched floating GLM MLA cache to batched int8 cache
- server batchability routing for GLM MLA `--kv-bits 8`
- tiny GLM DSA `BatchGenerator(..., kv_bits=8)` smoke generation

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

The likely next wins are:

- DSA prefill profiling around top-k indexer work, sparse mask construction, and `take_along_axis`.
- Reducing per-chunk synchronization in prefill once memory growth is characterized.
- Capturing server chat-template/tokenization cache hit rates for repeated coding-agent request prefixes.
- Avoiding full latent-cache dequantization during long-context decode, likely with a block-wise or sparse dequantized attention path.
- Scheduler policy is conservative by design: it bypasses incompatible queued requests only for compatible work, and it still waits rather than converting fp GLM MLA cache state into int8 mid-flight.
