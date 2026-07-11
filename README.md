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

### Installation

The local stable setup uses a `uv`-managed CPython 3.13.14 virtual environment.
This venv does not need the `pip` module installed; use `uv pip` from outside
the environment to install or refresh packages:

```sh
uv python install 3.13.14
uv venv --python 3.13.14 /Users/kioju/.venvs/mlx-glm52
source /Users/kioju/.venvs/mlx-glm52/bin/activate
```

Install the known-good dependency set:

```sh
uv pip install --python /Users/kioju/.venvs/mlx-glm52/bin/python \
  mlx==0.31.2 \
  mlx-lm==0.31.3 \
  transformers==5.12.1 \
  safetensors==0.8.0 \
  numpy==2.4.6 \
  tokenizers==0.22.2 \
  sentencepiece==0.2.1 \
  protobuf==7.35.1 \
  huggingface-hub==1.20.1
```

Install this checkout as the active editable `mlx-lm` package. Add
`MLX_LM_WITH_CUSTOM_KERNEL=1` when the vendored GLM custom kernels should be
built into the editable install:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
MLX_LM_WITH_CUSTOM_KERNEL=1 \
uv pip install --python /Users/kioju/.venvs/mlx-glm52/bin/python --no-deps -e .
```

If the checkout does not already have a local `.venv` link, create one for
shorter commands:

```sh
ln -sfn /Users/kioju/.venvs/mlx-glm52 .venv
```

### Benchmark snapshot

Current local measurements use a single-device M3 Ultra 512GB Mac Studio, the
3.5bpw dynamic-quantized GLM-5.2 MLX checkpoint above, MLX 0.31.2, int8 GLM MLA
KV cache, native sparse MLA over quantized KV, and the serving command below.

The updated checkpoint includes native MTP-layer weights. This fork can load
those weights and exposes an opt-in GLM DSA MTP speculative decode path through
`--mtp-speculative`, but the recommended long-context OpenCode server command
below intentionally keeps it disabled. The decode numbers below are therefore
single-stream baseline decode numbers, not MTP-accelerated decode.

Recent OpenCode task log, July 9, 2026. This run primarily measured
server-cache-covered suffix prefill; disk checkpoint candidates were found, but
the already-live server cache covered the useful prefixes.

Prefill / first-token measurements. Prefill TPS is fresh prefill tokens divided
by summed prefill chunk time; first-token time includes lookup, scheduling, and
the final decode step overhead around those chunks.

| Workload | Prompt tokens | Reused tokens | Fresh prompt tokens | Prefill chunks | Prefill TPS | First token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold small first request | 3,458 | 0 | 3,458 | 22.308s | 155.0 | 22.859s |
| Mostly cached tool turn | 12,918 | 12,654 | 264 | 2.735s | 96.2 | 2.945s |
| Larger suffix turn | 43,143 | 32,699 | 10,444 | 71.657s | 145.7 | 71.960s |
| Final long decode turn | 44,417 | 44,086 | 331 | 3.279s | 100.6 | 3.549s |

Decode measurements:

| Workload | Prompt tokens | Generated tokens | Decode seconds | Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Final long completed turn | 44,417 | 5,041 | 320.230 | 15.742 |
| Short tool-call turns | 12,918 to 43,945 | 141 to 474 | 8.849 to 29.032 | about 15.9 to 16.3 |
| 8K synthetic decode-context exact-hit baseline | 8,192 | 64 | 3.897 | 16.420 |
| 8K native decode indexer probe, opt-in | 8,192 | 64 | 4.088 | 15.657 |

The final long turn logged steady decode progress at roughly 15.7 tok/s:

```text
generation progress: prompt_tokens=44417 generated_tokens=4096 decode_seconds=260.206 decode_tps=15.741
generation complete: prompt_tokens=44417 generated_tokens=5041 decode_seconds=320.230 decode_tps=15.742
```

### Recommended GLM-5.2 serving settings

For long-context latency on Apple silicon, especially when 200K+ token prompts
are common rather than exceptional, the recommended starting point is:

```
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
  --checkpoint-cache-dir /Volumes/USB-SSD-RAID-0/mlx-lm/prompt-checkpoints \
  --checkpoint-save-exact disabled \
  --checkpoint-prefill-frontier-save disabled \
  --checkpoint-delta-chunk-tokens 8192 \
  --checkpoint-post-response-save-mode async \
  --checkpoint-async-save-shutdown-timeout 0 \
  --checkpoint-async-save-backlog-limit 2 \
  --generation-shutdown-timeout 0 \
  --checkpoint-shutdown-save-limit 0 \
  --request-max-tokens-floor 384000 \
  --prompt-concurrency 1 \
  --decode-concurrency 1 \
  --disable-batching \
  --loop-guard-ngram-size 64 \
  --loop-guard-repeats 3 \
  --loop-guard-min-tokens 256 \
  --decode-progress-interval-tokens 512 \
  --tool-call-max-tokens 8192 \
  --reasoning-loop-guard-min-chars 60 \
  --reasoning-loop-guard-repeats 4 \
  --reasoning-loop-guard-max-span-chars 2048 \
  --session-loop-history-size 256 \
  --session-loop-max-no-progress-turns 64 \
  --session-loop-repeated-output-limit 4 \
  --session-loop-repeated-action-limit 8
```

Do not pass `--model-name` for this OpenCode setup unless you have explicitly verified that you need request-facing model-name aliasing. The normal single-model local server workflow loads the model from `--model` and serves OpenCode requests through `/v1/chat/completions`.

OpenCode may still send a conservative `max_tokens` value such as 32000 even
when `limit.output` is set higher. `--request-max-tokens-floor 384000` raises
that request cap on the server side so long coding-agent turns are not cut off
early with `finish_reason=length`. Because that also allows an unclosed tool
call to run for a long time, keep `--tool-call-max-tokens 8192` enabled for
OpenCode serving. It stops a single tool-call span that grows past the limit
with `finish_reason=length` instead of waiting for the full raised token cap.
This local fork assumes a single-user, single-active-task server profile. The
session loop guard therefore uses one process-global local-session history pool,
not HTTP cookies, TCP connections, or per-client session IDs. With the
recommended `--session-loop-*` settings, the server records recent completions
across all HTTP requests and stops before generation when that global history
shows a non-tool loop: too many turns without visible assistant progress, or
repeated visible output/action signatures inside the bounded pool. It
intentionally does not count total tool-call turns at request-history or
process-global scope because legitimate tasks may issue many tools over a long
run. On detection it returns a short assistant message asking the client to stop
the same action loop, summarize state, and ask the user before continuing. This
is intentionally suited to personal OpenCode serving; do not use it as-is for
multi-user serving where independent conversations need separate loop histories.

The recommended server command intentionally leaves `--temp` and `--top-p` unset so request-side clients can control sampling. In local use, lower-temperature request settings helped reduce repetitive reasoning loops and “thought-loop” style failure modes while still preserving enough diversity for useful responses.

`--loop-guard-*` is a server-side fuse for exact repeated token loops during long decode, including repeated reasoning/thought spans. The default guard watches for repeated 8/16/32/64-token windows after 256 generated tokens; set `--loop-guard-ngram-size 0` to disable it. If the model still enters near-duplicate but non-exact loops, lower request sampling first (`temperature`, `top_p`) and add a small request-side `repetition_penalty` such as `1.05` to `1.10` when your client supports it.

`--reasoning-loop-guard-*` is a text-span guard for reasoning doom loops that
do not line up with exact token n-grams. The recommended settings follow the
same detection shape used by Liquid AI's
[Antidoom](https://www.liquid.ai/blog/antidoom) mining pass: a normalized
reasoning span/block of at least 60 characters repeating four times. This is an
inference-time fuse, not FTPO/LoRA training. On detection, the server stops the
current generation with `finish_reason=stop` and returns a visible assistant
message telling the client to stop the reasoning loop, summarize state, and ask
the user before continuing.

`--reasoning-max-tokens` is intentionally not part of the recommended
long-running OpenCode command. It defaults to `0` (disabled). Use it only as a
diagnostic hard fuse when you want to intentionally stop a single unclosed
reasoning span; values such as `8192` can terminate legitimate long coding-agent
turns before the task has completed.

`--decode-progress-interval-tokens 512` logs generation progress from the
generation worker. If prompt processing reaches 100% and no `decode first token`
line follows, the first decode step is stalled. If progress continues with
`state=tool` but the client shows no visible output, the model is producing an
unclosed tool call; the `--tool-call-max-tokens` fuse bounds that case. If the
log shows many short completions ending with `finish_reason=tool_calls`, the
client is repeatedly executing tools and sending the growing history back. The
server does not cap the total number of tool-call turns; it only bounds local
pool loops such as the same action signature repeating via
`--session-loop-repeated-action-limit`.

`--kv-bits 8` is not a prefill-compute speedup by itself. Its value is that GLM MLA int8 KV cache reduces long-context KV memory and keeps 200K+ prompts inside the intended memory envelope. The native DSA indexer score/top-k route remains compatible with this setting because it uses the DSA indexer cache, not the GLM MLA KV cache.

Prompt checkpointing remains the dominant TTFT optimization for repeated coding-agent prefixes. For latency-focused 200K+ serving, `--disable-batching` keeps requests on the single-request path that reuses disk prompt checkpoints and saves post-response continued/delta checkpoints asynchronously. Keep `--checkpoint-prefill-frontier-save disabled` for interactive OpenCode sessions: synchronous frontier saves during prefill can write multi-GB checkpoint files before the first decoded token, blocking the active response long enough to trip operation timeouts. Existing prefix/frontier/delta checkpoints are still eligible for lookup and reuse when this is disabled. Enable prefill frontier saves only for controlled cache-building runs where a long foreground save is acceptable. `--checkpoint-delta-chunk-tokens 8192` stores long post-response deltas as a chain of smaller delta checkpoint files; this avoids repeatedly rewriting one huge suffix and lets later saves extend the deepest reusable checkpoint. Use `--checkpoint-delta-chunk-tokens 0` only when you need the legacy single-file delta behavior for comparison. Disable final exact checkpoints for this long-running server profile: 190K-token exact checkpoints are around 11GB each on the tested setup and can spend tens of seconds writing only to be pruned immediately. The measured cold-prefill sweep now favors `--prefill-step-size 8192`, `--prefill-max-qk-tokens 67108864`, and adaptive GLM DSA prefill disabled (`--glm-dsa-adaptive-prefill-step-size 0`). The QK cap shrinks only the chunks whose query-by-context product would get too large; the 8192-token first chunk crosses the native sparse handoff immediately, then later chunks shrink automatically as the cap requires. Keep `MLX_LM_GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT` at its default/effective 131072 handoff for the Python selected-KV sparse path; lowering that handoff increased runtime and memory in the tested 128K runs. The vendored native DSA indexer route is enabled by default through `MLX_LM_GLM_DSA_NATIVE_INDEXER` and can replace the Python/MLX indexer score plus top-k path for supported GLM-5.2 M3 prefill chunks at context 4096 and above. Its single-query decode score probe is separate and remains disabled unless `MLX_LM_GLM_DSA_NATIVE_DECODE_INDEXER=1`; the 8K exact-hit 64-token A/B measured it slightly slower than the existing decode path. The vendored native sparse MLA route has its own lower handoff, `MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_MIN_CONTEXT` (default 6144). `MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV=1` lets it consume int8 GLM MLA KV cache by temporarily dequantizing the full latent KV cache for the native kernel; the recommended `MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_MAX_CONTEXT=262144` has been profiled through 204800 tokens with all chunks on the native sparse route and no dense fallback. Keep larger values bounded until your target context length is profiled. Do not force `MLX_LM_GLM_DSA_FAST_PREFILL_KEY_BLOCK=2048` unless you are profiling it; the default key block is 8192. If Metal recovery or memory pressure appears on your real prompt distribution, retry with `--prefill-step-size 4096` first, then 2048 and 1024.

`--mtp-speculative --num-draft-tokens 2` enables the built-in GLM DSA MTP layer
as an experimental speculative decode path. It is intentionally not in the
recommended OpenCode command yet. Server MTP mode disables batching and keeps
disk prompt checkpoints separate from target-only checkpoints, but it now uses a
dedicated `mtp-speculative` RAM prompt-cache namespace for repeated prompts. An
exact RAM hit trims one token and prefills only that suffix token so the MTP
layer can rebuild the hidden state it needs. If exact checkpoint saves are
enabled, MTP writes persistent exact checkpoints with an `mtp-speculative-...`
filename prefix and the `glm52-local-mtp-speculative` checkpoint namespace so
they can coexist with target-only checkpoints. Continued/delta disk saves remain
disabled for MTP, and the recommended long-running OpenCode command still keeps
`--checkpoint-save-exact disabled` to avoid large post-response writes. The
server logs
`prompt_cache_source=mtp-speculative-server-cache` on RAM hits and
`mtp speculative complete` with `drafted_tokens`, `accepted_tokens`,
`acceptance_rate`, `mean_accepted`, and `emitted_per_target_forward`; those
fields are the first sanity check before comparing wall-clock decode TPS.
GLM MTP cold prefill updates the MTP hidden state and cache without projecting
full-prompt vocabulary logits. The target path projects only the final position
of each prefill chunk. The completion log reports `target_prefill_tokens`,
`target_prefill_logits_skipped`, `mtp_prefill_tokens`, and
`mtp_prefill_logits_skipped`; the skipped counts confirm that the native GLM
prefill paths are active. MTP prefill shifts input tokens one position ahead of
the corresponding target hidden states and fuses the first draft proposal into
its final position, matching the DeepSeek-family proposer layout used by vLLM.
`mtp_prefill_shifted=true` and `mtp_prefill_first_draft_fused=true` confirm that
the aligned path is active and that one MTP forward was removed from the first
verification round. Fully accepted rounds also update the MTP cache through a
hidden/cache-only catch-up pass; `catchup_logits_skipped` counts the discarded
full-vocabulary projections avoided there. Greedy requests also report
`draft_logsumexp_skipped`, avoiding normalization work for MTP proposal logits
that are never returned. Without logits processors, greedy target verification
also batches draft-token comparison; `target_greedy_verify_batches` and
`target_greedy_verify_tokens` report that path. When the OpenAI request does not
ask for `logprobs` or `top_logprobs`, greedy MTP generation also skips target
full-vocabulary normalization. `target_logsumexp_skipped` counts those emitted
positions and `return_logprobs=false` confirms the practical server path.
Short target verification batches of up to eight tokens use selected-KV sparse
attention when the MLA cache is int8, even below the normal 131K Python sparse
prefill handoff. This prevents a missing native quantized-KV opt-in from
dequantizing and projecting the full context during MTP verification. The
recommended native quantized-KV route still takes priority when enabled. That
native route now compacts each verification query's causal top-k rows before
dequantization and passes the compact float KV to the fused sparse MLA kernel;
it no longer materializes the full float KV cache for every MTP target forward.
After 16 drafted tokens, the default adaptive guard falls back to regular
one-token target decode when observed MTP acceptance remains below `0.20`. This
fallback reuses the already-verified target cache and requires no re-prefill.
It stops MTP-layer execution completely so fallback decode returns to the
regular target-only path. The MTP cache remains reusable through its last
aligned prefix; post-response RAM insertion uses the shorter cache length and
the next tool turn prefills only the suffix beyond that point.
The completion log reports `adaptive_fallback`, the emitted-token position and
acceptance rate at the transition, subsequent target forward counts, and
`adaptive_fallback_mtp_cache_abandoned_tokens` for the suffix deliberately left
out of the reusable MTP cache.
Set `--mtp-adaptive-fallback-min-drafted-tokens 0` to disable the guard while
profiling raw MTP behavior.
GLM-5.2 MTP recycles the shared-head final-norm output between draft steps and,
when `index_share_for_mtp_iteration=true`, reuses the first draft step's DSA
top-k indices for the rest of that verification round. This matches the GLM-DSA
acceptance fix in [vLLM PR #45895](https://github.com/vllm-project/vllm/pull/45895);
`mtp_iteration_topk_reuses` confirms that the indexer-saving path is active.

`--checkpoint-async-save-backlog-limit 2` bounds queued post-response continued
and delta checkpoint saves. These saves can hold large prompt-cache snapshots in
unified memory while the next request is already decoding. If the client drives
many tool-call turns faster than checkpoints can be written, the server skips
new async checkpoint saves once the queue reaches the limit instead of letting
snapshot backlog grow into Metal out-of-memory aborts. Skipped saves are safe:
they only lose an opportunistic future warm prefix, not the current response.

Post-response continued/delta checkpoint saves run asynchronously by default so
the generation worker can accept the next request without waiting for disk I/O.
Use `--checkpoint-post-response-save-mode sync` only when debugging the older
synchronous path. Keep `--checkpoint-async-save-shutdown-timeout 0` and
`--generation-shutdown-timeout 0` for Ctrl+C responsiveness; raise them only
when shutdown should wait for active generation or checkpoint writes.

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
- delta hits restore a base prefix/frontier/continued checkpoint plus one or more chained suffix-only delta cache files, avoiding multi-GB exact checkpoint writes for 190K+ token conversations;
- disabled checkpointing reports `checkpoint_resolution=disabled`, `disk_cached_tokens=0`, and a full fresh prefill;
- mismatched tokens, incompatible GLM DSA metadata, incompatible GLM MLA KV settings, malformed checkpoints, or missing files fall back to a normal miss.

Benchmark-backed measurements on this local fork showed exact 4096/8192-token checkpoint hits dropping TTFT from roughly 23s/52s to roughly 0.2s. Controlled LCP runs are the clean way to measure partial reuse: for example, an 8192-token cached prefix plus a 2048-token suffix should report `expected_reused_prefix_tokens=8192`, `disk_cached_tokens=8192`, `fresh_prompt_tokens=2048`, and `checkpoint_expected_match=true`. Benchmark rows also include LCP lookup counters such as `checkpoint_lcp_block_matches`, `checkpoint_prefix_hashes`, and `checkpoint_cache_layout_rejections` so stale or incompatible checkpoint candidates are visible without scraping logs.

Use `--no-prompt-checkpoint` only for cold or disabled-baseline measurements. Use `--checkpoint-save-exact disabled` or `--no-save-exact-checkpoint` when you want to store only configured prefix/frontier checkpoints without also creating a final exact full-prompt checkpoint.
Shutdown-time RAM checkpoint flushes are disabled by default via `--checkpoint-shutdown-save-limit 0`; only enable them for short bounded caches, and keep `--checkpoint-shutdown-max-tokens` below the largest prompt size you are willing to serialize during Ctrl+C shutdown.

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

GLM DSA sparse prefill is enabled by default because the old dense fallback materialized full `(heads, query_length, context_length)` prefill tensors and could OOM well below the advertised long-context envelope. To avoid the Python selected-KV sparse path becoming pathologically slow too early, it waits until the effective context reaches one token below `MLX_LM_GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT` (default 131072), because generation prefill leaves the final prompt token for logits. For sparse chunks, MLA attention stays in latent space and avoids selected K/V projection; long chunks whose causal prefix already covers the full top-k set also skip the redundant selected-mask gather. Use `--fast-prefill disabled` only for short-context comparison runs. `--fast-prefill-query-chunk` controls selected-query microbatches, and `MLX_LM_GLM_DSA_FAST_PREFILL_KEY_BLOCK` controls the DSA indexer key block size. If the vendored native extension is built, `MLX_LM_GLM_DSA_NATIVE_INDEXER` can route supported GLM-5.2 M3 indexer score/top-k prefill chunks from context 4096 onward. This route remains usable with `--kv-bits 8`. `MLX_LM_GLM_DSA_NATIVE_DECODE_INDEXER=1` enables the experimental single-query decode indexer score kernel for measurement only; leave it unset for serving. `MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL` can route supported GLM MLA chunks from its own `MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_MIN_CONTEXT` threshold (default 6144), without waiting for the Python sparse handoff. The int8 GLM MLA KV path is opt-in through `MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV=1`; it keeps the cache stored as int8 but temporarily dequantizes the full latent KV tensor for native sparse MLA. Use `MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_MAX_CONTEXT` as a memory guard.

Benchmark rows include `glm_dsa_native_sparse_prefill_route_state`,
`glm_dsa_native_sparse_prefill_primary_fallback`, and
`glm_dsa_native_sparse_prefill_config_blocker`. With
`--native-sparse-quantized-kv enabled`, 32K cold prefill on the tested setup
dropped from about 395s to about 231s, replacing the long Python/MLX sparse
attention tail with native sparse MLA while leaving q projection as the next
large bottleneck. A later chunk-route sweep favored the 6144-token native sparse
handoff: 8K prefill-stop improved from about 55.85s at 8192 to 53.72s, and 16K
prefill-stop improved from about 110.28s at 8192 to 109.90s. Delaying the
handoff to 10240 regressed the 16K run to about 112.15s. With that 6144-token
handoff, raising the base prefill step from 2048 to 4096 reduced 16K
prefill-stop from about 109.90s to 105.44s and 32K prefill-stop from about
225.18s to 219.63s. Raising the base step again to 8192 with the same QK cap
reduced 16K to 104.11s, 32K to 207.10s, 64K to 436.05s, and 128K to 963.13s.
The larger first chunk avoids the early dense fallback and lowers total
q-projection time; on the 128K run q projection dropped from about 550.84s at
4096 to about 434.19s at 8192. The QK cap shrinks later chunks automatically,
and 128K peak memory stayed about 336.41GB. Peak memory at 16K through 64K rose
by about 0.53GB compared with the 4096-step runs.

Extending the native sparse quantized-KV guard beyond 128K kept the route stable
on the tested Mac Studio M3 Ultra 512GB setup. With `--prefill-step-size 8192`
and `--prefill-max-qk-tokens 67108864`, cold prefill measured 160K at about
1302.12s, 196608 tokens at about 1646.78s, and 204800 tokens at about 1739.78s.
All three runs stayed on native sparse MLA with dense chunks at 0; the 204800
run used 336 native chunks, peaked at about 340.64GB, and reported about
730.91s in q projection, 491.57s in native sparse attention, and 196.14s in
native indexer top-k.

The vendored native `glm_dsa_q8_vup_flat` kernel can be enabled separately for
quantized GLM DSA `unembed_out` projection when the fixed M3 GLM shape matches
64 heads, latent dim 512, value dim 256, affine int8 weights, and group size 64.
It is opt-in via `MLX_LM_GLM_DSA_NATIVE_Q8_VUP=1` or `--native-q8-vup enabled`
because the model-free microbench can be slower than MLX `quantized_matmul` on
some lengths. Benchmark rows report `glm_dsa_native_q8_vup_hits` and
`glm_dsa_native_q8_vup_fallback_reasons`.

For q4 `unembed_out` weights, `MLX_LM_GLM_DSA_NATIVE_Q4_VUP=1` /
`--native-q4-vup enabled` enables the matching `glm_dsa_q4_vup_flat` probe. It
is also disabled by default. On the tested 16K native sparse MLA profile it hit
the q4 route but left end-to-end TTFT effectively unchanged.

The vendored native q4 q projection probes are also opt-in:
`MLX_LM_GLM_DSA_NATIVE_Q4_QA=1` / `--native-q4-qa enabled` and
`MLX_LM_GLM_DSA_NATIVE_Q4_QB=1` / `--native-q4-qb enabled`. They are useful for
isolated profiling but are not part of the recommended server command yet. On
the tested 8K cold prefill, q projection split into about 29.7s q_a projection,
0.18s q_a RMSNorm, and 2.1s q_b projection; earlier q4 native probes did not
reduce end-to-end TTFT.
Use `--prefill-profile-isolate enabled` when reading q_projection sub-stage
timings: without it, MLX lazy evaluation can charge upstream work to q_a. On a
2K check, q_a changed from about 9.18s in normal profiling to about 0.23s with
isolated profiling.
With isolated profiling extended across indexer and attention stages, the tested
8K native sparse run spent about 0.49s in DSA indexer top-k and about 16.87s in
native sparse MLA attention, making sparse MLA attention the next attention-side
target.

The native sparse MLA kernel can also select an experimental tile with
`MLX_LM_GLM_DSA_SPARSE_MLA_TILE` or `--native-sparse-mla-tile`; unset/default
uses `bk256_dc32_wm8`. Available aliases are `bk128`, `bk256`,
`bk128_dc64`, `wm4`, `bk128_wm4`, and `bk128_dc64_wm4`. This is a profiling
knob, not a new recommended server default yet. For low-memory tile sweeps
without loading the full model, use
`benchmarks/glm52_sparse_mla_tile_microbench.py`. Initial synthetic sweeps at
512x8192/topk2048 and 2048x16384/topk2048 still favored the default `bk256`
tile.

The q_a probe can also select a tile with
`MLX_LM_GLM_DSA_NATIVE_Q4_QA_TILE` or `--native-q4-qa-tile`; unset/default uses
`bk64`. Available tiles are `bk32`, `bk64`, `bm16`, `bn16`, `bn64`, `bm64`,
`bm16bn64`, and `bm64bn64`. This is still an experimental measurement knob. On
the same 8K
benchmark build, native q4 q_a with `bk64` measured about 52.99s TTFT and
26.41s q_a projection versus about 53.26s TTFT and 26.58s q_a projection with
the native q4 q_a route disabled. The 2K run favored `bn64`, but the 8K run
regressed to about 53.41s TTFT and 26.71s q_a projection, so `bn64` is not a
long-context default. For low-memory tile sweeps without loading the full
model, use `benchmarks/glm52_q4_qa_tile_microbench.py`.

The q_b probe has a matching tile selector through
`MLX_LM_GLM_DSA_NATIVE_Q4_QB_TILE` or `--native-q4-qb-tile`; unset/default uses
`bm64` for the opt-in native q4 q_b path. Available tiles are `bk32`, `bk64`,
`bm16`, `bn16`, `bn64`, `bm64`, `bm16bn64`, `bm64bn64`, and `bk64bn64`. The
standalone `benchmarks/glm52_q4_qb_tile_microbench.py` measured only a small
tile-level margin, with `bm64` best on the 2048/8192 synthetic sweeps.

There is also an opt-in q_a dense-cache probe:
`MLX_LM_GLM_DSA_Q_A_DENSE_CACHE=1` / `--q-a-dense-cache enabled`. It
dequantizes each q4 `q_a_proj` weight to a dense fp16/bf16 matrix on first use
and reuses it for later prefill calls. This trades roughly 1.5-2GB of extra
resident memory for a warmed q_a projection path, so it is a measurement knob
rather than a recommended server setting. On the tested 8K repeat run, the
warmed path was effectively unchanged versus dense-cache disabled.

**Bottleneck hypothesis**
The bottleneck is still long-context prefill itself: later 32k chunks climbed to around 40s per 2048-token chunk. DSA/top-k and long-context attention/dequantization are the likely next places to profile, but checkpoint reuse is the practical answer for repeated coding-agent prefixes right now. Benchmark JSON now includes `checkpoint_prefill_chunk_summaries` plus slowest-chunk and route-count columns so native sparse MLA/indexer policy changes can be evaluated chunk by chunk.

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

They bound checkpoint file count, total checkpoint storage, and frontier checkpoint saves per generation run. Chunked delta saves intentionally trade fewer giant files for more smaller delta files, so raise the file limit if you lower `--checkpoint-delta-chunk-tokens` for very long 1M-token sessions.
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
- Long-context decode with GLM DSA top-k already gathers the selected quantized MLA latent KV entries before dequantization. A faster decode path likely needs fused selected gather/dequant/attention or MTP/speculative decode rather than another full-cache dequant guard.
- If generation behaves unexpectedly after changing model/runtime settings, clear the GLM-5.2 local runtime cache first.

### Prefill and decode benchmarks

See `docs/glm52-prefill-benchmark.md` for the GLM-5.2 prefill/decode benchmark commands, measured fields, and current batching notes. The benchmark script lives at `benchmarks/glm52_prefill_benchmark.py`. Use `--mode single` with `--max-tokens 1` for TTFT / checkpoint measurements, `--mode decode-context --max-tokens 64` for long-context decode TPS, and `--mode queued --max-tokens 8` or similar to expose admission wait time under serving load.

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
