# GLM-5.2 Vision (initial support)

This path attaches the frozen Kimi-K2.6 MoonViT tower and Baseten's trained
projector to an existing MLX GLM-5.2 language model. It does not require or
download the full `baseten/GLM-5.2-Vision-NVFP4` checkpoint.

## Architecture and weight mapping

For each image, the official Kimi-K2.6 NaViT processor creates 14×14 RGB
patches and a `(t, h, w)` grid. MoonViT applies:

1. `Conv2d(3, 1152, kernel=14, stride=14)` and interpolated learned 2D
   position embeddings.
2. 27 pre-norm transformer blocks, each with 16-head 2D RoPE attention and
   an `1152 → 4304 → 1152` tanh-approximate GELU MLP.
3. Final LayerNorm, temporal mean pooling, and a 2×2 spatial merge. Every
   output token therefore contains four 1152-wide patch vectors.
4. The trained projector:
   `LayerNorm(1152) → flatten(4×1152) → Linear(4608,4608) → GELU →
   Linear(4608,6144)`.

The projector contains exactly 49,558,272 BF16 parameters:

| Downloaded key | MLX key | Shape |
| --- | --- | --- |
| `mm_projector.pre_norm.{weight,bias}` | unchanged | `[1152]` |
| `mm_projector.linear_1.{weight,bias}` | unchanged | `[4608,4608]`, `[4608]` |
| `mm_projector.linear_2.{weight,bias}` | unchanged | `[6144,4608]`, `[6144]` |

Upstream Kimi projector names `proj.0` and `proj.2` are also accepted and map
to `linear_1` and `linear_2`. The MoonViT weights all live in Kimi-K2.6
`model-00064-of-000064.safetensors`; the only layout conversion is its
PyTorch Conv2d weight from OIHW to MLX OHWI.

The image feature count is `(h / 2) × (w / 2)`. A single GLM `<|image|>`
token (ID 154854) is expanded to that count, and those token embeddings are
replaced with projected image features. `<|begin_of_image|>` and
`<|end_of_image|>` remain ordinary text tokens.

## Required files

- The default local language model:
  `~/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-4.5bpw`.
- `mm_projector.safetensors` and its GLM5V `config.json`.
- Kimi-K2.6 `model-00064-of-000064.safetensors` (833.8 MB).
- Kimi-K2.6 `preprocessor_config.json` (recommended).

No other Kimi weight shard is used. In particular, Kimi's own projector and
language-model weights are not needed.

## Reused implementation

The MLX implementation is adapted from the Kimi module on MLX-VLM main
(commit `6c2f660`), specifically:

- `mlx_vlm/models/kimi_k25/vision.py` for MoonViT, 2D RoPE, and patch order.
- `mlx_vlm/models/kimi_k25/processing_kimi_k25.py` for image preprocessing.
- `mlx_vlm/models/kimi_k25/kimi_k25.py` for projector and embedding insertion.

Kimi-K2.6 still declares model type `kimi_k25`, so this is also its current
MLX-VLM implementation. The port was checked against Kimi-K2.6's official
`modeling_kimi_k25.py`, `kimi_k25_vision_processing.py`, and
`media_utils.py`; the initial image path uses the official three-column
`(t,h,w)` grid throughout.

The real BF16 weight files also load strictly. An independent FP32 comparison
against Transformers' native Kimi-K2.6 PyTorch implementation gave a maximum
absolute difference of `8.47e-6` for the complete projected 6144-wide output.
The corresponding offline golden can be rerun without PyTorch in the test
environment:

```bash
GLM5V_TEST_PROJECTOR_DIR="$HOME/models/glm52-vision-projector" \
  python -m unittest \
  tests.test_glm5v.TestGLM5Vision.test_real_weights_match_official_pytorch_golden
```

## One request

```bash
python -m mlx_lm.examples.glm52_vision \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-image-tokens 128 \
  --prefill-step-size 16 \
  --max-tokens 8 \
  --no-thinking \
  --verbose
```

The example defaults to the Alis model and projector paths above. Override
`--model`, `--projector`, or `--moonvit` only when the files live elsewhere.
`--max-image-tokens` retains Kimi's resize algorithm while lowering its input
patch budget; it is useful for a bounded first run and can be omitted for the
normal-resolution path. The example prints the selected image-token count and
each language-model prefill chunk to stderr. The conservative 16-token Vision
prefill step is intentional: it completed the real Alis request on the
validated M3 Ultra, whereas a single 128-token external-embedding chunk stalled
in one Metal command. Text-only serving retains its independently tuned larger
prefill steps.

The same path is available as Python APIs through
`mlx_lm.glm5v.GLM5VisionAdapter`. `adapter.prepare(...)` returns prompt IDs
and `input_embeddings` suitable for `mlx_lm.generate`. It explicitly
materializes MoonViT/projector output and the completed input buffer before
language-model prefill. This evaluation boundary prevents MLX from joining the
27-layer vision graph and the 78-layer Alis graph into one oversized Metal
command.

## OpenAI-compatible server

Add `--vision-projector` to the normal Alis server command. With the downloaded
layout used here, the MoonViT shard and preprocessor config are inferred:

```bash
python -m mlx_lm server \
  --model "$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-4.5bpw" \
  --vision-projector "$HOME/models/glm52-vision-projector" \
  --vision-disable-thinking \
  --vision-temperature 0 \
  --disable-batching \
  --host 127.0.0.1 \
  --port 8000
```

Chat Completions accepts OpenAI-style `image_url` parts. Data URLs are enabled
by default:

```json
{
  "model": "default_model",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "text", "text": "Describe this image."}
    ]
  }]
}
```

The Responses API accepts the equivalent `input_image` part. HTTP(S) fetching
is deliberately unsupported; clients should download remote images and send a
bounded data URL. Server-local paths require `--vision-allow-local-images` and
are disabled by default. Each structured image part is rewritten to
`<|begin_of_image|><|image|><|end_of_image|>` before the Alis chat template is
rendered, avoiding that text-only template's image-rejection reminder without
changing files in the 395 GiB model directory.

The server defaults limit each request to 8 images, each image to 20 MiB and
40 million decoded pixels, and all decoded images together to 64 million
pixels. Adjust them with `--vision-max-images`, `--vision-max-image-bytes`,
`--vision-max-image-pixels`, and `--vision-max-total-image-pixels`.
The HTTP server additionally rejects request bodies over 256 MiB before
reading them. Use `--max-request-body-bytes` to change that independent cap.

Image requests always use the sequential generation path and a fresh KV cache.
They are never loaded from or inserted into token-keyed RAM/disk prompt caches:
different images have identical placeholder token IDs, so such reuse would be
incorrect. Text-only requests retain the existing batching and checkpoint
behavior. The server also caps only Vision prefill chunks at 16 tokens by
default; use `--vision-prefill-step-size` to override that initial-support
safety limit.

For the tested Alis dynamic checkpoint,
`--vision-disable-thinking --vision-temperature 0` is the recommended
interactive-server setting. Text-only requests still use the normal
chat-template reasoning and sampling defaults. Image requests can explicitly
restore thinking with request-level `reasoning_effort` or
`chat_template_kwargs.enable_thinking`; this is useful for diagnosis, but the
model can repeat reasoning or emit an extra `</think>` after an otherwise
complete answer. The server suppresses a duplicate closing control token and
can guard repeated reasoning with `--reasoning-loop-guard-*`. Visible-output
scanning is independent and disabled by default; enable
`--output-loop-guard-*` explicitly only when the risk of truncating legitimate
repeated code, JSON, logs, or fixtures is acceptable. Exact token-loop
detection excludes tool-call payloads, which are bounded separately by
`--tool-call-max-tokens`. An explicit request `temperature` overrides the
Vision-specific default.

## Initial limitations

- Image inference only; video preprocessing is not exposed yet, although the
  MoonViT tower retains its `t ≤ 4` temporal path.
- Batch size one and expanded-placeholder prompts only.
- Image requests do not support MTP speculative or draft-model decoding because
  those paths do not accept the image `input_embeddings`.
- NVFP4 ModelOpt weights from the Baseten checkpoint are not an MLX
  quantization format. This integration uses the existing Alis MLX text model;
  it never loads or downloads the Baseten language-model shards.
