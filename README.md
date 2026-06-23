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

### Recommended target model

This branch has been tested with:

    avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw

Example local path when downloaded through LM Studio:

    ~/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw

### Generate example

```
MLX_METAL_FAST_SYNCH=1 python -m mlx_lm generate \
  --model "$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw" \
  --prompt "Hello. Briefly introduce yourself." \
  --max-tokens 32 \
  --kv-bits 8 \
  --kv-group-size 64 \
  --quantized-kv-start 4096 \
  --temp 0.4 \
  --top-p 0.95
```

### Recommended GLM-5.2 serving settings

For long-running local GLM-5.2 serving on Apple silicon, the recommended starting point is:

```
MLX_METAL_FAST_SYNCH=1 python -m mlx_lm server \
  --model "$HOME/.lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw" \
  --host 0.0.0.0 \
  --port 8000 \
  --temp 0.4 \
  --top-p 0.95 \
  --kv-bits 8 \
  --kv-group-size 64 \
  --quantized-kv-start 4096 \
  --prefill-step-size 2048 \
  --prompt-concurrency 2 \
  --decode-concurrency 2
```

Do not pass `--model-name` for this OpenCode setup unless you have explicitly verified that you need request-facing model-name aliasing. The normal single-model local server workflow loads the model from `--model` and serves OpenCode requests through `/v1/chat/completions`.

`--temp 0.4` and `--top-p 0.95` are recommended as conservative default sampling settings for coding-agent and long-context workflows. In local use, lower-temperature sampling helped reduce repetitive reasoning loops and “thought-loop” style failure modes while still preserving enough diversity for useful responses.

`--kv-bits 8` is not a cold-prefill speedup by itself. Its value is that GLM MLA int8 KV cache can now be used with continuous batching, which makes long-context queued serving more practical and gives memory/concurrency headroom.

Prompt checkpointing remains the dominant TTFT optimization for repeated coding-agent prefixes. For this machine, `--prompt-concurrency 2` and `--decode-concurrency 2` were the best default balance. Avoid setting decode concurrency higher than prompt concurrency for fresh long-prefix workloads unless you are comfortable with mixed-cache rejection churn.

Keep prompt checkpointing enabled. It is the dominant TTFT win for repeated prefixes: 4096/8192 exact hits dropped from ~23s/~52s to ~0.2s. 
**Interpretation** 
kv_bits=8 does not materially improve cold prefill speed. It helps by enabling continuous batching with long GLM MLA contexts and gives modest memory headroom, especially as contexts grow. It is most useful for long-running local serving where KV memory and concurrency matter. Avoid prompt-concurrency=4 as a default. It barely improves TPS at 4096, but roughly doubles TTFT and adds 6 to 9GB peak memory. Also avoid decode-concurrency > prompt-concurrency for fresh long prefixes unless you are comfortable with mixed-cache rejection churn; q8/p2/c4 hit 90 rejections and p95 wait ~195s. 
**Bottleneck Hypothesis** 
The bottleneck is still long-context prefill itself: later 32k chunks climbed to ~40s per 2048-token chunk. DSA/top-k and long-context attention/dequantization are the likely next places to profile, but checkpoint reuse is the practical answer for repeated coding-agent prefixes right now.

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

When changing model weights, quantization, tokenizer, adapters, GLM implementation details, or KV quantization settings, invalidate the local runtime cache by moving or deleting the shared root:

    mv ~/.cache/mlx-lm/glm52-local \
       ~/.cache/mlx-lm/glm52-local.bak.$(date +%Y%m%d_%H%M%S)

#### Prompt checkpoint retention limits

The prompt checkpoint manifest/pruning layer uses these default limits:

    MLX_LM_PROMPT_CHECKPOINT_MAX_FILES=256
    MLX_LM_PROMPT_CHECKPOINT_MAX_BYTES=128GiB
    MLX_LM_PROMPT_CHECKPOINT_MAX_FRONTIERS_PER_RUN=16

They bound checkpoint file count, total checkpoint storage, and frontier checkpoint saves per generation run.

### Important limitations

- This is a GLM-5.2-specialized fork, not a generic `mlx-lm` runtime.
- GLM MLA KV quantization currently supports only `--kv-bits 8`.
- DSA indexer cache remains floating-point and is intentionally not quantized.
- With GLM MLA `--kv-bits 8`, the server can use the continuous `BatchGenerator` path. It will not merge quantized and unquantized GLM MLA caches in the same active batch, so fresh short requests may wait for an incompatible quantized batch instead of being quantized earlier than `--quantized-kv-start`. Compatible queued requests can still bypass that waiting request and join the active batch.
- Other model families with `--kv-bits` still use the single-request `stream_generate` path unless they grow batch-compatible quantized cache support.
- Prompt checkpointing is trusted single-model local cache reuse. It validates prefix, cache structure, GLM DSA metadata, and GLM MLA KV settings, but it does not prove full model weight, tokenizer, adapter, or artifact identity.
- Long-context decode currently dequantizes the full MLA latent cache on read. Sparse or block-wise dequantization is future work.
- If generation behaves unexpectedly after changing model/runtime settings, clear the GLM-5.2 local runtime cache first.

### Prefill benchmark

See `docs/glm52-prefill-benchmark.md` for the GLM-5.2 prefill benchmark command, measured fields, and current batching notes. The benchmark script lives at `benchmarks/glm52_prefill_benchmark.py`. Use `--mode single` with `--max-tokens 1` for TTFT / checkpoint measurements, and `--mode queued --max-tokens 8` or similar to expose admission wait time under serving load.

### Smoke test result

A short smoke test with `avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw` on Mac Studio M3 Ultra 512GB completed successfully with:

    --kv-bits 8 --kv-group-size 64 --quantized-kv-start 4096

Observed short-prompt result:

    Prompt: 19 tokens, ~6.9 tokens/sec
    Generation: 32 tokens, ~21 tokens/sec
    Peak memory: ~329 GB

Short prompts do not meaningfully demonstrate long-context KV memory savings or prompt checkpoint benefit. Use longer prompts to evaluate those paths.

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
