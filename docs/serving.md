# Serving GLM-5.2

This document covers the long-context, single-user serving profile for this
fork. For installation and the minimal server command, start with the
[top-level README](../README.md).

## Recommended long-context profile

The following profile favors repeated-prefix latency for long-running coding
agents. Set `MODEL` and `CHECKPOINT_DIR` to local paths before starting it.

```sh
export MODEL="$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw"
export CHECKPOINT_DIR="$HOME/.cache/mlx-lm/glm52-local/prompt-checkpoints"

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
  --request-max-tokens-floor 384000 \
  --temp 1.0 \
  --top-p 0.95 \
  --chat-template-args '{"reasoning_effort":"max"}' \
  --prompt-concurrency 1 \
  --decode-concurrency 1 \
  --disable-batching \
  --repetition-penalty 1.05 \
  --repetition-context-size 1024 \
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
  --session-loop-repeated-output-limit 64 \
  --session-loop-repeated-action-limit 8
```

Use `--host 0.0.0.0` only when access from a trusted local network is required.
Do not pass `--model-name` unless request-facing model aliasing is specifically
needed. The normal single-model workflow loads the model from `--model`.

## Optional GLM-5.2 Vision attachment

This fork's Vision path is validated against the same Alis model used above.
If the downloaded projector directory contains:

```text
~/models/glm52-vision-projector/
├── config.json
├── mm_projector.safetensors
└── moonvit/
    ├── model-00064-of-000064.safetensors
    └── preprocessor_config.json
```

add one argument to the recommended server command:

```sh
--vision-projector "$HOME/models/glm52-vision-projector"
```

The server then accepts Chat Completions `image_url` parts and Responses API
`input_image` parts. `data:image/...` URLs are enabled by default. HTTP(S)
fetching is deliberately unsupported; download remote images client-side and
send a bounded data URL. Use `--vision-allow-local-images` only for trusted
server-local paths.

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

OpenCode can send a conservative `max_tokens` even when its configured output
limit is higher. `--request-max-tokens-floor 384000` raises that request cap.
Because an unclosed tool call could then run for a long time,
`--tool-call-max-tokens 8192` bounds a single tool-call span.

The recommended sampling defaults follow the GLM-5.2 guide:
`temperature=1.0`, `top_p=0.95`, and `reasoning_effort=max` for complex agentic
work. The available server-side thinking defaults are:

```sh
--chat-template-args '{"reasoning_effort":"max"}'
--chat-template-args '{"reasoning_effort":"high"}'
--chat-template-args '{"enable_thinking":false}'
```

Request-level `temperature` and `top_p` override server defaults. Chat
Completions accepts top-level `"reasoning_effort": "high"`; the Responses API
accepts `"reasoning": {"effort": "high"}`. GLM-5.2 accepts `high` and `max`.

## Loop guards and progress

The repetition penalty reduces the chance of a repeated-token loop.
`--loop-guard-*` bounds the damage if an exact repeated token loop still
occurs. It watches repeated 8/16/32/64-token windows after the configured
minimum generation length. Set `--loop-guard-ngram-size 0` only for controlled
diagnosis.

`--reasoning-loop-guard-*` detects repeated normalized reasoning spans that do
not align with exact token n-grams. On detection, generation stops and returns
a visible message asking the client to summarize state before continuing.
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
          "name": "GLM-5.2-Alis-MLX-Dynamic-3.5bpw",
          "limit": {
            "context": 1048576,
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
