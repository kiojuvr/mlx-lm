# Prompt Checkpoints

This fork stores trusted local GLM-5.2 prompt checkpoints and reuses the
longest exact token prefix it can validate. The match is deterministic
longest-common-prefix reuse, not fuzzy matching.

## Reuse behavior

* Exact full-prompt hits replay the last prompt token and skip almost all
  prefill.
* Prefix and frontier hits restore cached KV/DSA state for the shared prefix
  and prefill only the changed suffix.
* Continued checkpoints capture post-response cache state.
* Chained delta checkpoints restore a base checkpoint plus one or more
  suffix-only cache files, avoiding repeated multi-gigabyte exact writes for
  very long conversations.
* Disabled checkpointing reports `checkpoint_resolution=disabled`,
  `disk_cached_tokens=0`, and performs a full fresh prefill.
* Token mismatches, a different MLX version, incompatible GLM DSA metadata,
  incompatible GLM MLA KV settings, malformed checkpoints, and missing files
  become normal misses.

Prompt checkpointing validates token prefixes, cache structure, GLM DSA
metadata, GLM MLA KV settings, and the exact MLX version. It does not prove
complete model-weight, tokenizer, adapter, or artifact identity. Treat it as a
single-model local cache.

## Cache location

The shared runtime cache root is:

```text
~/.cache/mlx-lm/glm52-local/
```

Prompt checkpoints are stored in:

```text
~/.cache/mlx-lm/glm52-local/prompt-checkpoints/
```

The server and benchmark option `--checkpoint-cache-dir` redirects the
checkpoint files and manifest. It maps to
`MLX_LM_PROMPT_CHECKPOINT_CACHE_DIR`; it does not change model math, tensor
layout, or validation behavior.

Use an isolated directory for benchmark runs:

```sh
--checkpoint-cache-dir "$(mktemp -d)"
```

## When to invalidate the cache

Use an empty checkpoint directory after changing any of the following:

* model weights or model artifacts
* tokenizer files
* adapters
* quantization
* GLM implementation details
* KV-cache quantization or layout settings

An MLX version change does not require manual deletion. Older checkpoints fail
the version check, become normal cache misses, and are replaced after fresh
prefill.

To preserve the old cache for diagnosis, move the shared root:

```sh
mv ~/.cache/mlx-lm/glm52-local \
  ~/.cache/mlx-lm/glm52-local.bak.$(date +%Y%m%d_%H%M%S)
```

To remove only prompt checkpoints:

```sh
rm -rf ~/.cache/mlx-lm/glm52-local/prompt-checkpoints
```

If generation quality changes unexpectedly after a model or runtime change,
invalidate this cache before investigating model behavior.

## Recommended interactive policy

For latency-focused long-running OpenCode sessions:

```sh
--disable-batching \
--checkpoint-save-exact disabled \
--checkpoint-prefill-frontier-save disabled \
--checkpoint-delta-chunk-tokens 8192 \
--checkpoint-post-response-save-mode async \
--checkpoint-async-save-backlog-limit 2 \
--checkpoint-async-save-shutdown-timeout 0 \
--checkpoint-shutdown-save-limit 0
```

Synchronous frontier saves during prefill can write multi-gigabyte files
before the first decoded token and can block the response long enough to trip
client timeouts. Disabling new frontier saves does not prevent reuse of
existing prefix, frontier, or delta checkpoints.

Chunked delta saves extend the deepest reusable checkpoint without repeatedly
rewriting one large suffix. Set `--checkpoint-delta-chunk-tokens 0` only when
comparing against the legacy single-file delta behavior.

Final exact checkpoints are disabled in this profile because a 190K-token
checkpoint was about 11 GB on the tested system and could spend tens of seconds
writing before immediate pruning.

Post-response continued and delta saves run asynchronously by default. The
backlog limit bounds large cache snapshots retained in unified memory while
the next request is already decoding. When the queue is full, a new
opportunistic save is skipped; the current response is unaffected.

Use `--checkpoint-post-response-save-mode sync` only to diagnose the older
synchronous path. Keep the generation and asynchronous-save shutdown timeouts
at zero for responsive Ctrl+C behavior unless shutdown must wait for active
work.

Shutdown-time RAM checkpoint flushes are disabled by default with
`--checkpoint-shutdown-save-limit 0`. Enable them only for bounded caches, and
keep `--checkpoint-shutdown-max-tokens` below the largest prompt you are
willing to serialize during shutdown.

## Retention

The manifest/pruning layer defaults to:

```text
MLX_LM_PROMPT_CHECKPOINT_MAX_FILES=256
MLX_LM_PROMPT_CHECKPOINT_MAX_BYTES=256GiB
MLX_LM_PROMPT_CHECKPOINT_MAX_FRONTIERS_PER_RUN=16
```

These limits bound file count, total storage, and frontier saves per generation
run. Smaller delta chunks produce more files, so the file limit may need to be
raised for very long sessions. The byte limit accepts bytes and binary or
decimal suffixes such as `512GiB`, `1TiB`, and `2TB`.

## Validation and benchmarking

Benchmark-backed measurements on this fork showed exact 4096/8192-token hits
reducing TTFT from roughly 23/52 seconds to roughly 0.2 seconds. A controlled
8192-token cached prefix plus 2048-token suffix should report:

```text
expected_reused_prefix_tokens=8192
disk_cached_tokens=8192
fresh_prompt_tokens=2048
checkpoint_expected_match=true
```

Use `--no-prompt-checkpoint` only for a cold or disabled baseline. Use
`--checkpoint-save-exact disabled` when storing only configured
prefix/frontier checkpoints.

All controlled-LCP commands and the reported lookup counters are kept in
[Prefill and decode benchmarks](glm52-prefill-benchmark.md#longest-prefix-checkpoint-reuse).
