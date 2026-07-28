# Serving GLM-5.2

This document covers the long-context, single-user serving profile for this
fork. For installation and the minimal server command, start with the
[top-level README](../README.md).

## Recommended long-context profile

The following profile favors repeated-prefix latency for long-running coding
agents. Set `MODEL` and `CHECKPOINT_DIR` to local paths before starting it.

```sh
export MODEL="$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-4.5bpw"
export CHECKPOINT_DIR="$HOME/.cache/mlx-lm/glm52-45bpw/prompt-checkpoints"

MLX_LM_PROMPT_CHECKPOINT_DEBUG=1 \
MLX_METAL_FAST_SYNCH=1 \
MLX_LM_GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT=131072 \
MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV=1 \
MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_MAX_CONTEXT=262144 \
python -m mlx_lm server \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --kv-bits 8 \
  --prompt-cache-size 1 \
  --prompt-cache-bytes 12GB \
  --temp 1.0 \
  --top-p 0.95 \
  --min-p 0.20 \
  --repetition-penalty 1.10 \
  --kv-group-size 64 \
  --quantized-kv-start 4096 \
  --prefill-step-size 8192 \
  --prefill-max-qk-tokens 67108864 \
  --glm-dsa-adaptive-prefill-step-size 0 \
  --checkpoint-cache-dir "$CHECKPOINT_DIR" \
  --checkpoint-save-exact disabled \
  --checkpoint-prefill-frontier-save disabled \
  --checkpoint-delta-chunk-tokens 8192 \
  --checkpoint-post-response-save-mode async \
  --checkpoint-async-save-shutdown-timeout 0 \
  --checkpoint-async-save-backlog-limit 2 \
  --generation-shutdown-timeout 0 \
  --checkpoint-shutdown-save-limit 0 \
  --chat-template-args '{"reasoning_effort":"max"}' \
  --prompt-concurrency 1 \
  --decode-concurrency 1 \
  --disable-batching \
  --cancel-active-on-new-request \
  --loop-guard-ngram-size 64 \
  --loop-guard-repeats 3 \
  --loop-guard-min-tokens 256 \
  --decode-progress-interval-tokens 512 \
  --tool-call-max-tokens 32768 \
  --reasoning-loop-guard-min-chars 60 \
  --reasoning-loop-guard-repeats 4 \
  --reasoning-loop-guard-max-span-chars 2048 \
  --session-loop-history-size 256 \
  --session-loop-max-no-progress-turns 64 \
  --session-loop-repeated-output-limit 64 \
  --session-loop-repeated-action-limit 8
```

Use `--host 0.0.0.0` only when access from a trusted local network is required.
Do not pass `--model-name` unless request-facing model aliasing is specifically
needed. The normal single-model workflow loads the model from `--model`.

The separate `glm52-45bpw` checkpoint directory is intentional. Prompt
checkpoint metadata does not prove complete model-weight identity, so a cache
created by the 3.5 bpw checkpoint must not be reused after this migration.
Keeping the former `glm52-local` directory untouched preserves it for
diagnosis or rollback.

## Optional GLM-5.2 Vision attachment

This fork's Vision path uses the same Alis model selected above. The 4.5 bpw
checkpoint retains the architecture and tokenizer artifacts used by the
validated 3.5 bpw Vision integration; rerun the standalone smoke command below
after migrating the language model.
Download only the four files used by the MLX Vision path:

```sh
VISION_DIR="$HOME/models/glm52-vision-projector"
mkdir -p "$VISION_DIR/moonvit"

uvx --from huggingface-hub hf download \
  baseten/GLM-5.2-Vision-NVFP4 \
  config.json mm_projector.safetensors \
  --local-dir "$VISION_DIR"

uvx --from huggingface-hub hf download \
  baseten/GLM-5.2-Vision-NVFP4 \
  preprocessor_config.json \
  --local-dir "$VISION_DIR/moonvit"

uvx --from huggingface-hub hf download \
  moonshotai/Kimi-K2.6 \
  model-00064-of-000064.safetensors \
  --local-dir "$VISION_DIR/moonvit"
```

Keep the filenames in each `hf download` command. Omitting them would select
the repositories' full language-model checkpoints, which are not required.
The Baseten `plugins/` directory and all other Kimi K2.6 weight shards are
also unnecessary for this MLX implementation. The required download is about
0.93 GB in total.

The resulting directory should contain:

```text
~/models/glm52-vision-projector/
├── config.json
├── mm_projector.safetensors
└── moonvit/
    ├── model-00064-of-000064.safetensors
    └── preprocessor_config.json
```

Add these arguments to the recommended server command:

```sh
--vision-projector "$HOME/models/glm52-vision-projector" \
--vision-disable-thinking \
--vision-temperature 0
```

The server then accepts Chat Completions `image_url` parts and Responses API
`input_image` parts. `data:image/...` URLs are enabled by default. HTTP(S)
fetching is deliberately unsupported; download remote images client-side and
send a bounded data URL. Use `--vision-allow-local-images` only for trusted
server-local paths.

`--vision-disable-thinking --vision-temperature 0` is the validated OpenWebUI
default for the Alis dynamic checkpoint. It avoids model-side failure modes
observed with sampled image generation: repeated reasoning, a second
`</think>` followed by a restarted answer, and short-period visible-output
loops. Text-only requests keep the profile's normal reasoning and sampling
defaults. A client can explicitly opt an image request back into thinking with
`"reasoning_effort": "high"`/`"max"` or
`"chat_template_kwargs": {"enable_thinking": true}`; the reasoning loop guard
remains active for that experimental path. Visible-output loop scanning is a
separate opt-in described below. An explicit request `temperature` overrides
`--vision-temperature`.

The defaults allow at most 8 images, 20 MiB and 40 million decoded pixels per
image, and 64 million decoded pixels in total. These are configurable with
`--vision-max-images`, `--vision-max-image-bytes`,
`--vision-max-image-pixels`, and `--vision-max-total-image-pixels`.
The server also rejects request bodies larger than 256 MiB before reading them;
adjust that independent ceiling with `--max-request-body-bytes`.

The Alis tokenizer has the required exact mapping:
`<|begin_of_image|>` = 154830, `<|image|>` = 154854, and
`<|end_of_image|>` = 154831. Structured image parts are converted to these
tokens before the repository's text-only chat template can emit its
no-multimodal reminder.

Vision requests use fresh KV caches and the sequential path, even if batching
is enabled for text requests. Token-keyed RAM and disk prompt caches are not
read or written for Vision because the same placeholder token sequence can
represent different images. MTP speculative and draft-model generation are not
supported for Vision requests.

Architecture, file provenance, standalone inference, and request examples are
documented in [GLM-5.2 Vision](../mlx_lm/GLM5V.md).

Before starting the long-lived server for the first time, the documented
standalone smoke command can cap the image to 128 merged tokens and report
prefill progress. The Vision adapter materializes its output before Alis
prefill, keeping the MoonViT and language-model graphs in separate Metal
commands. Vision requests additionally default to 16-token prefill chunks
through `--vision-prefill-step-size 16`; the larger text-only prefill setting
above remains unchanged.

## Why these settings are used

`--kv-bits 8` reduces the memory occupied by the GLM MLA cache. It is a
long-context memory optimization, not a prefill-compute speedup. The DSA
Indexer cache is independent and intentionally remains floating point.

The 4.5 bpw weights use about 424 GB (395 GiB) and leave less KV-cache
headroom than the former 3.5 bpw profile. On a 512 GB M3 Ultra, treat about
500K total tokens as the normal operating budget and about 600K as the
upper edge. Use the 3.5 bpw checkpoint when a roughly 1M-token context is more
important than the 4.5 bpw quality improvement.

The routed experts are NVFP4 (`mode="nvfp4"`, group size 16). The optional
affine 2/3-bit gate/up fusion therefore does not engage for these experts and
correctly falls back to MLX `gather_qmm`; the native weighted reduction remains
applicable. Attention projections remain 4-bit affine.

The measured cold-prefill profile favors `--prefill-step-size 8192`,
`--prefill-max-qk-tokens 67108864`, and
`--glm-dsa-adaptive-prefill-step-size 0`. The QK cap shrinks later chunks as
their query-by-context product grows. If Metal recovery or memory pressure
appears on a real prompt distribution, retry with a step size of 4096, then
2048 or 1024.

`--disable-batching` keeps long, repeated-prefix requests on the path that can
reuse disk prompt checkpoints and asynchronously save continued/delta
checkpoints. See [Prompt checkpoints](prompt-checkpoints.md) for the save and
invalidation rules.

`--cancel-active-on-new-request` gives this single-user profile
Vision-only last-request-wins behavior. If a browser stops an Open WebUI Vision
response but the Open WebUI proxy keeps its upstream MLX connection open, the
MLX server cannot observe the browser disconnect directly. A newly submitted
generation request then stops the old Vision generation and runs after its
current token finishes. Active text-only requests, including their tool-call
generation, are not cancellation targets. A Vision request that itself uses
tools remains a Vision cancellation target. Do not enable this option on a
shared multi-user server.

The server has no fixed output-token limit by default. Large OpenCode source
updates and substantial Vision responses therefore continue until a model stop,
a client-supplied limit, client cancellation, or a loop guard ends generation.
There is no generally correct server-wide output cap; safety comes from those
explicit termination mechanisms rather than an arbitrary default.

`--request-max-tokens-floor` is still available when a client sends an
explicitly small `max_tokens`/`max_completion_tokens`; it does not add a limit
when the client omits one. `--max-tokens` remains available only for deployments
that deliberately want a server-wide fallback cap.

OpenCode can send a conservative explicit `max_tokens` even when its configured
output limit is higher. For that client only, add
`--request-max-tokens-floor 384000`. Because the larger bound lets an unclosed
tool call run for a long time, keep `--tool-call-max-tokens 8192` and the loop
guards enabled. Do not add the floor to a general OpenWebUI profile.

For large source-code updates, the recommended profile leaves sampling at the
server defaults: `temperature=0.0` and `top_p=1.0`. Omitting `--temp` and
`--top-p` also lets a client override either value explicitly. The profile
continues to use `reasoning_effort=max` for complex agentic work. The available
server-side thinking defaults are:

```sh
--chat-template-args '{"reasoning_effort":"max"}'
--chat-template-args '{"reasoning_effort":"high"}'
--chat-template-args '{"enable_thinking":false}'
```

Request-level `temperature` and `top_p` override server defaults. Chat
Completions accepts top-level `"reasoning_effort": "high"`; the Responses API
accepts `"reasoning": {"effort": "high"}`. GLM-5.2 accepts `high` and `max`.

## Loop guards and progress

`--loop-guard-*` bounds the damage if an exact repeated token loop still
occurs. `--loop-guard-ngram-size` is the maximum checked period: the guard
watches 8-, 16-, 32-, and 64-token windows up to that maximum, plus the exact
configured value when it is nonstandard. Each check is scoped to one continuous
reasoning or visible-output span. State transitions reset it, and tool-call
payloads are excluded. Use `--tool-call-max-tokens` to bound an unclosed
tool-call span. Set `--loop-guard-ngram-size 0` only for controlled diagnosis.

`--reasoning-loop-guard-*` checks repeated normalized reasoning spans that do
not align with exact token n-grams, including Japanese text whose sentences are
not separated by spaces. On detection, it retains the visible recovery
message. Block-based matches must be consecutive at the current text tail and
must not exceed the configured maximum span. Visible assistant output has an independent
`--output-loop-guard-*` guard. It is disabled by default because valid code,
JSON, logs, and test fixtures can contain repeated text.

To opt into the initial Vision branch's visible-output scanning, add:

```sh
--output-loop-guard-min-chars 60 \
--output-loop-guard-repeats 4 \
--output-loop-guard-max-span-chars 2048
```

Existing `--reasoning-loop-guard-*` options remain accepted and continue to
guard reasoning, but no longer enable visible-output scanning. Enable the output
guard only when truncating legitimate repeated output is an acceptable tradeoff.

`--reasoning-max-tokens` is a separate hard fuse and defaults to disabled; it
is intentionally omitted from the normal long-running profile.

The session loop guard uses one process-global local-session history pool. It
can detect repeated output, lack of visible progress, and period-1 through
period-4 tool-turn patterns. It does not count the total number of legitimate
tool calls. Because its history is process-global, this profile is not suitable
for independent multi-user conversations.

`--decode-progress-interval-tokens 512` makes the generation worker report
progress. If prompt processing reaches 100% but no first-token line follows,
the first decode step is stalled. If progress continues with `state=tool` but
the client shows no output, the model is likely producing an unclosed tool
call. Many short completions ending in `finish_reason=tool_calls` indicate that
the client is repeatedly executing tools and returning the growing history.

## OpenCode provider configuration

OpenCode can use the server through its OpenAI-compatible provider:

```json
{
  "provider": {
    "mlx-lm": {
      "name": "mlx-lm (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "mlx-lm",
        "timeout": 7200000,
        "chunkTimeout": 7200000
      },
      "models": {
        "default_model": {
          "name": "GLM-5.2-Alis-MLX-Dynamic-4.5bpw",
          "limit": {
            "context": 524288,
            "output": 384000
          }
        }
      }
    }
  }
}
```

OpenCode sends the model key `default_model`; `name` is only a display label.
Do not place numeric `temperature` or `top_p` values under the model entry.
OpenCode expects the model-level `temperature` field to be a boolean or
omitted.

The server exposes:

* `GET /v1/models`
* `POST /v1/chat/completions`
* `POST /v1/completions`
* `POST /v1/responses`

## Continuous batching profile

For shorter mixed workloads where aggregate throughput matters more than
single-request TTFT and disk checkpoint reuse, continuous batching can be
useful:

```sh
MLX_METAL_FAST_SYNCH=1 \
python -m mlx_lm server \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --kv-bits 8 \
  --kv-group-size 64 \
  --quantized-kv-start 4096 \
  --prefill-step-size 2048 \
  --prompt-concurrency 2 \
  --decode-concurrency 2
```

Compatible GLM MLA int8 caches can share the continuous batching path.
Quantized and unquantized GLM MLA caches are not merged in the same active
batch, so a fresh short request may wait behind an incompatible quantized
batch. Other model families using `--kv-bits` remain on the single-request
path unless they implement compatible quantized batching.

Avoid prompt concurrency 4 as a default: local measurements showed little TPS
gain, roughly doubled TTFT, and added 6–9 GB peak memory. Decode concurrency
higher than prompt concurrency can also cause mixed-cache rejection churn for
fresh long-prefix workloads.

## Related documentation

* [Prompt checkpoints](prompt-checkpoints.md)
* [Native GLM kernels](native-kernels.md)
* [MTP speculative decoding](mtp.md)
* [Prefill and decode benchmarks](glm52-prefill-benchmark.md)
