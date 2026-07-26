# mlx-lm GLM-5.2 Local Fork

A GLM-5.2-focused fork of [`mlx-lm`](https://github.com/ml-explore/mlx-lm) for running very large GLM-5.2 models locally on Apple Silicon.

This fork is optimized for long-context, single-user coding-agent workloads. It is not intended to be a general-purpose replacement for upstream `mlx-lm`.

## Highlights

* GLM-5.2 / `glm_moe_dsa` model support
* OpenAI-compatible Chat Completions and Responses APIs
* GLM MLA int8 KV cache support
* Exact longest-prefix prompt checkpoint reuse
* Asynchronous continued and delta checkpoint saves
* Optional native Metal kernels for GLM DSA and routed MoE
* Continuous batching support for compatible GLM MLA int8 caches
* Repetition and loop guards for long-running agent sessions
* Experimental GLM DSA MTP speculative decoding

## Recommended configuration

The quality-oriented target configuration is:

* Apple Silicon Mac
* Mac Studio M3 Ultra with 512 GB unified memory
* MLX 0.32.0
* CPython 3.13
* [`avlp12/GLM-5.2-Alis-MLX-Dynamic-4.5bpw`](https://huggingface.co/avlp12/GLM-5.2-Alis-MLX-Dynamic-4.5bpw)

The 4.5 bpw checkpoint is the quality-oriented profile. Its routed experts use
NVFP4 and its weights occupy about 395 GiB on disk. On a 512 GB system, use
int8 KV cache and plan for about 500K tokens of comfortable context (about
600K maximum), rather than the roughly 1M-token budget of the 3.5 bpw
checkpoint. Smaller-memory systems require a smaller model.

## Installation

Clone the fork's default branch:

```sh
git clone https://github.com/kiojuvr/mlx-lm.git

cd mlx-lm
```

Create a virtual environment with `uv`:

```sh
uv python install 3.13.14
uv venv --python 3.13.14 .venv
source .venv/bin/activate
```

Install the known-good runtime versions:

```sh
uv pip install \
  mlx==0.32.0 \
  transformers==5.12.1 \
  safetensors==0.8.0 \
  numpy==2.4.6 \
  tokenizers==0.22.2 \
  sentencepiece==0.2.1 \
  protobuf==7.35.1 \
  huggingface-hub==1.20.1
```

Install this checkout:

```sh
uv pip install -e .
```

The native Metal kernels are optional. See [Native kernels](docs/native-kernels.md) for the Xcode and build requirements.

## Basic server

Set the path to the local model:

```sh
export MODEL="$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-4.5bpw"
```

Start a local OpenAI-compatible server:

```sh
MLX_METAL_FAST_SYNCH=1 \
python -m mlx_lm server \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --kv-bits 8 \
  --kv-group-size 64 \
  --quantized-kv-start 4096 \
  --prefill-step-size 8192 \
  --prefill-max-qk-tokens 67108864
```

The API base URL is:

```text
http://127.0.0.1:8000/v1
```

Use `--host 0.0.0.0` only when the server must be reachable from a trusted local network.

For the complete long-context OpenCode profile, including prompt checkpoints and loop guards, see [Serving GLM-5.2](docs/serving.md).

## Prompt checkpoints

This fork can store and reuse the longest validated token prefix of earlier requests.

A matching checkpoint restores the cached GLM DSA and MLA state and prefills only the changed suffix. Exact, prefix, frontier, continued, and chained delta checkpoints are supported.

The default cache root is:

```text
~/.cache/mlx-lm/glm52-local/
```

Prompt checkpoints are stored under:

```text
~/.cache/mlx-lm/glm52-local/prompt-checkpoints/
```

Use an empty cache directory after changing model weights, tokenizer files, adapters, quantization, GLM implementation details, or KV-cache settings.
Checkpoints also record the exact MLX version; an MLX upgrade automatically
turns older entries into cache misses so they are regenerated safely.
When migrating from the 3.5 bpw checkpoint, point
`--checkpoint-cache-dir` at a new directory such as
`~/.cache/mlx-lm/glm52-45bpw/prompt-checkpoints`; the model-weight change is
not covered by the automatic MLX-version check.

See [Prompt checkpoints](docs/prompt-checkpoints.md) for validation, retention, and invalidation rules.

## OpenCode and other clients

OpenCode and other OpenAI-compatible clients can connect to:

```text
http://127.0.0.1:8000/v1
```

The server supports:

* `GET /v1/models`
* `POST /v1/chat/completions`
* `POST /v1/completions`
* `POST /v1/responses`

See [Serving GLM-5.2](docs/serving.md) for an OpenCode provider configuration and the recommended long-running agent settings.

## Documentation

* [Serving GLM-5.2](docs/serving.md)
* [Prompt checkpoints](docs/prompt-checkpoints.md)
* [Native GLM kernels](docs/native-kernels.md)
* [MTP speculative decoding](docs/mtp.md)
* [Prefill and decode benchmarks](docs/glm52-prefill-benchmark.md)

## Important limitations

* This is a GLM-5.2-specialized fork.
* The recommended serving profile assumes a trusted, single-user local environment.
* The process-global session loop history is not suitable for independent multi-user conversations.
* GLM MLA KV quantization currently supports int8 cache storage.
* The DSA Indexer cache intentionally remains floating point.
* Native kernels support only specific GLM-5.2 shapes and quantization layouts.
* MTP speculative decoding remains experimental.
* Prompt checkpoints must be invalidated after incompatible model or runtime
  changes not already covered by the automatic MLX-version check.

## Upstream project

General `mlx-lm` documentation, model conversion, training, LoRA, and support for other model families are documented in the upstream project:

[`ml-explore/mlx-lm`](https://github.com/ml-explore/mlx-lm)

This repository intentionally specializes parts of the runtime for GLM-5.2 and is not structured as an upstream pull request.

## License and acknowledgements

This repository retains the upstream MIT license.

The vendored GLM custom kernels include code derived from oMLX under the Apache License 2.0. See the license and README files under:

```text
mlx_lm/custom_kernels/glm_moe_dsa/
```
