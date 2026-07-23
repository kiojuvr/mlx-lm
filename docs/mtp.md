# MTP Speculative Decoding

GLM DSA MTP speculative decoding is experimental. The recommended
long-running serving profile keeps it disabled while output stability and
acceptance are compared with target-only decode.

The model checkpoint must contain the native MTP-layer weights.

## Enabling MTP

For a single-request server:

```sh
python -m mlx_lm server \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --mtp-speculative \
  --num-draft-tokens 1
```

One draft token is the current recommendation for the tested checkpoint.
Recursive two-token drafting reduced acceptance in local measurements. On one
8K-context, 64-token synthetic run, the optimized one-token path reached about
17.45 tok/s versus 16.64 tok/s for target-only decode. The gain is modest and
depends on acceptance rate.

MTP server mode disables batching.

## Prompt-cache isolation

MTP and target-only caches are deliberately separate.

Repeated MTP prompts use a dedicated `mtp-speculative` RAM prompt-cache
namespace. An exact RAM hit trims one token and prefills that suffix token so
the MTP layer can rebuild the hidden state it needs. RAM hits log:

```text
prompt_cache_source=mtp-speculative-server-cache
```

When exact disk saves are enabled, MTP uses an
`mtp-speculative-...` filename prefix and the
`glm52-local-mtp-speculative` checkpoint namespace. These files can coexist
with target-only checkpoints.

Continued and delta disk saves remain disabled for MTP. The recommended
long-running profile also uses `--checkpoint-save-exact disabled` to avoid
large post-response writes. A combined target/MTP cache remains reusable after
vocabulary-head-free cache updates.

## Prefill alignment

GLM MTP cold prefill updates hidden state and cache without projecting
full-prompt vocabulary logits. The target path projects only the final position
of each prefill chunk.

MTP prefill shifts input tokens one position ahead of their corresponding
target hidden states and fuses the first draft proposal into the final
position. The following completion fields verify this path:

```text
target_prefill_tokens
target_prefill_logits_skipped
mtp_prefill_tokens
mtp_prefill_logits_skipped
mtp_prefill_shifted
mtp_prefill_first_draft_fused
```

Fully accepted rounds use a hidden/cache-only catch-up pass.
`catchup_logits_skipped` counts avoided full-vocabulary projections.

## Proposal and verification optimizations

Greedy MTP proposals skip normalization when proposal log probabilities are
not returned; `draft_logsumexp_skipped` reports this.

When no history-dependent logits processor is active, greedy target
verification batches the draft-token comparison. The completion reports
`target_greedy_verify_batches` and `target_greedy_verify_tokens`.

When the OpenAI request does not ask for `logprobs` or `top_logprobs`, target
normalization is also skipped. `target_logsumexp_skipped` and
`return_logprobs=false` confirm the normal server path.

Short target verification batches of up to eight tokens use selected-KV sparse
attention with float or int8 MLA caches, even below the normal 131K Python
sparse-prefill handoff. Quantized selected rows are dequantized once, then each
query uses the optimized one-token MLX attention path. This preserves one
batched target-model forward per speculative round without materializing the
full float KV cache.

For diagnosis only:

```sh
MLX_LM_MTP_SEQUENTIAL_QUANTIZED_VERIFY=1
```

This rebuilds verification with separate one-token target forwards and was
slower on the tested checkpoint.

GLM-5.2 MTP also recycles shared-head final-norm output between draft steps.
When `index_share_for_mtp_iteration=true`, it reuses the first draft step's DSA
top-k indices for the remaining verification round.
`mtp_iteration_topk_reuses` reports this route.

## Adaptive fallback

The production path includes a low-acceptance guard. Once the configured
minimum number of draft tokens has been observed, low acceptance switches the
rest of the request to regular one-token target decode without rebuilding the
target cache.

The fallback stops MTP-layer execution. The MTP cache remains reusable through
its last aligned prefix; post-response RAM insertion records that shorter
length so the next turn prefills only the suffix.

Completion logs include:

```text
adaptive_fallback
adaptive_fallback_at_emitted_tokens
adaptive_fallback_acceptance_rate
adaptive_fallback_mtp_cache_abandoned_tokens
```

Disable the guard only for raw draft-depth measurements:

```sh
--mtp-adaptive-fallback-min-drafted-tokens 0
```

## Observability

The `mtp speculative complete` log reports:

```text
drafted_tokens
accepted_tokens
acceptance_rate
mean_accepted
emitted_per_target_forward
```

Check acceptance and emitted tokens per target forward before interpreting
wall-clock decode TPS. Low acceptance can make MTP slower despite fewer target
forwards.

## Benchmarking

The benchmark supports checkpoint-disabled production-path measurements,
target-only versus MTP comparisons, and draft-depth sweeps. It reports
`mtp_speculative_*` fields for acceptance, prefill projection skipping,
verification, adaptive fallback, and top-k reuse.

All commands and current measurements live in the
[GLM DSA Decode Context Benchmark](glm52-prefill-benchmark.md#glm-dsa-decode-context-benchmark).
