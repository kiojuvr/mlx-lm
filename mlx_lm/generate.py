# Copyright © 2023-2024 Apple Inc.

import argparse
import contextlib
import copy
import functools
import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from transformers import PreTrainedTokenizer

from .models import cache
from .models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    TokenBuffer,
    load_prompt_cache,
)
from .sample_utils import make_sampler
from .tokenizer_utils import TokenizerWrapper
from .utils import does_model_support_input_embeddings, load

DEFAULT_PROMPT = "hello"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.0
DEFAULT_TOP_K = 0
DEFAULT_XTC_PROBABILITY = 0.0
DEFAULT_XTC_THRESHOLD = 0.0
DEFAULT_MIN_TOKENS_TO_KEEP = 1
DEFAULT_SEED = None
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_QUANTIZED_KV_START = 5000
GLM_DSA_MTP_ENV = "MLX_LM_GLM_DSA_MTP"
PROMPT_CHECKPOINT_DEBUG_ENV = "MLX_LM_PROMPT_CHECKPOINT_DEBUG"
PROMPT_CHECKPOINT_FRONTIER_MIN_TOKENS = 8192
PROMPT_CHECKPOINT_FRONTIER_STRIDE_TOKENS = 16384
PROMPT_CHECKPOINT_MAX_FRONTIERS_PER_RUN_ENV = (
    "MLX_LM_PROMPT_CHECKPOINT_MAX_FRONTIERS_PER_RUN"
)
PROMPT_CHECKPOINT_MAX_FRONTIERS_PER_RUN = 16
DEFAULT_PREFILL_MAX_QK_TOKENS = 67_108_864


def _prompt_checkpoint_debug(message):
    if os.environ.get(PROMPT_CHECKPOINT_DEBUG_ENV) != "1":
        return
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s:%(name)s:%(message)s",
        )
    logging.getLogger(__name__).info("prompt checkpoint: %s", message)


_GLM_DSA_CHUNK_COUNTER_KEYS = (
    "fast_prefill_hits",
    "native_sparse_prefill_hits",
    "native_indexer_hits",
)
_GLM_DSA_CHUNK_REASON_KEYS = (
    "fallback_reasons",
    "native_sparse_prefill_fallback_reasons",
    "native_indexer_fallback_reasons",
)
_GLM_DSA_CHUNK_STAGE_KEYS = (
    "q_projection",
    "kv_cache_update",
    "dsa_indexer_topk",
    "native_indexer_scores",
    "native_indexer_topk",
    "latent_kv_dequantization",
    "latent_kv_projection",
    "native_sparse_kv_dequantization",
    "sparse_gather",
    "attention",
    "native_sparse_attention",
)
_GLM_DSA_DECODE_STAGE_KEYS = (
    "input_layernorm",
    "q_projection",
    "q_a_projection",
    "q_a_layernorm",
    "q_b_projection",
    "kv_cache_update",
    "dsa_indexer_topk",
    "latent_kv_dequantization",
    "latent_kv_projection",
    "native_q8_vup",
    "native_q4_vup",
    "attention",
    "o_projection",
    "post_attention_layernorm",
    "mlp",
    "total_decode_attention",
    "total_decode_layer",
    "total_decode_model",
)


def _glm_dsa_prefill_profile_snapshot():
    module = sys.modules.get("mlx_lm.models.glm_moe_dsa")
    getter = getattr(module, "get_glm_dsa_prefill_profile", None)
    if getter is None:
        return None
    try:
        profile = getter(reset=False)
    except Exception:
        return None
    snapshot = {
        key: int(profile.get(key, 0)) for key in _GLM_DSA_CHUNK_COUNTER_KEYS
    }
    for key in _GLM_DSA_CHUNK_REASON_KEYS:
        snapshot[key] = dict(profile.get(key, {}))
    stages = profile.get("stages", {})
    snapshot["stages"] = {
        stage: {
            "seconds": float(stages.get(stage, {}).get("seconds", 0.0)),
            "count": int(stages.get(stage, {}).get("count", 0)),
        }
        for stage in _GLM_DSA_CHUNK_STAGE_KEYS
    }
    return snapshot


def _counter_delta(before, after, key):
    return int(after.get(key, 0)) - int(before.get(key, 0))


def _counter_dict_delta(before, after, key):
    before_values = before.get(key, {})
    after_values = after.get(key, {})
    delta = {}
    for reason in sorted(set(before_values) | set(after_values)):
        value = int(after_values.get(reason, 0)) - int(before_values.get(reason, 0))
        if value:
            delta[reason] = value
    return delta


def _glm_dsa_prefill_profile_chunk_fields(before, after):
    if before is None or after is None:
        return {}
    fields = {}
    for key in _GLM_DSA_CHUNK_COUNTER_KEYS:
        value = _counter_delta(before, after, key)
        if value:
            fields[f"glm_dsa_{key}"] = value
    for key in _GLM_DSA_CHUNK_REASON_KEYS:
        value = _counter_dict_delta(before, after, key)
        if value:
            fields[f"glm_dsa_{key}"] = value
    before_stages = before.get("stages", {})
    after_stages = after.get("stages", {})
    for stage in _GLM_DSA_CHUNK_STAGE_KEYS:
        before_stage = before_stages.get(stage, {})
        after_stage = after_stages.get(stage, {})
        seconds = float(after_stage.get("seconds", 0.0)) - float(
            before_stage.get("seconds", 0.0)
        )
        count = int(after_stage.get("count", 0)) - int(
            before_stage.get("count", 0)
        )
        if seconds:
            fields[f"glm_dsa_{stage}_seconds"] = seconds
        if count:
            fields[f"glm_dsa_{stage}_count"] = count
    return fields


def _glm_dsa_decode_profile_snapshot():
    module = sys.modules.get("mlx_lm.models.glm_moe_dsa")
    getter = getattr(module, "get_glm_dsa_decode_profile", None)
    if getter is None:
        return None
    try:
        profile = getter(reset=False)
    except Exception:
        return None
    stages = profile.get("stages", {})
    return {
        "stages": {
            stage: {
                "seconds": float(stages.get(stage, {}).get("seconds", 0.0)),
                "count": int(stages.get(stage, {}).get("count", 0)),
            }
            for stage in _GLM_DSA_DECODE_STAGE_KEYS
        },
    }


def _glm_dsa_decode_profile_fields(before, after):
    if before is None or after is None:
        return {}
    fields = {}
    before_stages = before.get("stages", {})
    after_stages = after.get("stages", {})
    for stage in _GLM_DSA_DECODE_STAGE_KEYS:
        before_stage = before_stages.get(stage, {})
        after_stage = after_stages.get(stage, {})
        seconds = float(after_stage.get("seconds", 0.0)) - float(
            before_stage.get("seconds", 0.0)
        )
        count = int(after_stage.get("count", 0)) - int(
            before_stage.get("count", 0)
        )
        if seconds:
            fields[f"glm_dsa_decode_{stage}_seconds"] = seconds
        if count:
            fields[f"glm_dsa_decode_{stage}_count"] = count
    return fields


def _format_prefill_chunk_fields(fields):
    parts = []
    for key, value in fields.items():
        if isinstance(value, dict):
            value = json.dumps(value, sort_keys=True, separators=(",", ":"))
        elif isinstance(value, float):
            value = f"{value:.6f}"
        parts.append(f"{key}={value}")
    return " ".join(parts)


def _as_token_list(tokens):
    if tokens is None:
        return None
    if isinstance(tokens, mx.array):
        tokens = tokens.tolist()
    return [int(t) for t in tokens]


def _checkpoint_store_lengths(lengths, total_tokens):
    if not lengths:
        return []
    result = []
    seen = set()
    for value in lengths:
        try:
            length = int(value)
        except (TypeError, ValueError):
            continue
        if length <= 0 or length >= total_tokens or length in seen:
            continue
        seen.add(length)
        result.append(length)
    result.sort()
    return result


def _checkpoint_frontier_lengths(
    total_tokens,
    min_tokens=PROMPT_CHECKPOINT_FRONTIER_MIN_TOKENS,
    stride_tokens=PROMPT_CHECKPOINT_FRONTIER_STRIDE_TOKENS,
):
    try:
        min_tokens = int(min_tokens)
        stride_tokens = int(stride_tokens)
    except (TypeError, ValueError):
        return []
    if total_tokens <= 1 or min_tokens <= 0 or stride_tokens <= 0:
        return []

    first = max(2, min_tokens)
    if first >= total_tokens:
        return []

    frontiers = {first}
    start = ((first + stride_tokens - 1) // stride_tokens) * stride_tokens
    for length in range(start, total_tokens, stride_tokens):
        if length >= first:
            frontiers.add(length)
    return sorted(frontiers)


def _prompt_checkpoint_env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _effective_prefill_step_size(
    requested_step_size: int,
    remaining_tokens: int,
    processed_tokens: int,
    prefill_max_qk_tokens: Optional[int],
    adaptive_step_size: Optional[int] = None,
) -> int:
    if adaptive_step_size is not None and adaptive_step_size > 0:
        requested_step_size = max(requested_step_size, int(adaptive_step_size))
    step_size = min(requested_step_size, remaining_tokens)
    if prefill_max_qk_tokens is None or prefill_max_qk_tokens <= 0:
        return step_size
    if step_size <= 1:
        return step_size

    max_qk_tokens = max(1, int(prefill_max_qk_tokens))
    while (
        step_size > 1
        and step_size * max(processed_tokens + step_size, 1) > max_qk_tokens
    ):
        step_size = max(1, max_qk_tokens // max(processed_tokens + step_size, 1))
    return max(1, step_size)


def _model_config_value(model, key):
    configs = [
        getattr(model, "config", None),
        getattr(model, "args", None),
    ]
    nested_model = getattr(model, "model", None)
    if nested_model is not None:
        configs.extend(
            [
                getattr(nested_model, "config", None),
                getattr(nested_model, "args", None),
            ]
        )
    for config in configs:
        if isinstance(config, dict):
            value = config.get(key)
        else:
            value = getattr(config, key, None)
        if value is not None:
            return value
    return getattr(model, key, None)


def _glm_dsa_adaptive_prefill_step_size(
    model,
    *,
    requested_step_size: int,
    processed_tokens: int,
    remaining_tokens: int,
    adaptive_step_size: int = 0,
    adaptive_after_tokens: int = 0,
    adaptive_min_remaining_tokens: int = 0,
) -> Optional[int]:
    """Return an opt-in larger prefill step for GLM DSA models."""
    if adaptive_step_size <= 0:
        return None
    if _model_config_value(model, "model_type") != "glm_moe_dsa":
        return None
    if processed_tokens < max(0, adaptive_after_tokens):
        return None
    if remaining_tokens < max(0, adaptive_min_remaining_tokens):
        return None
    return max(int(requested_step_size), int(adaptive_step_size))


def _checkpoint_limit_frontier_lengths(lengths, max_frontiers):
    if max_frontiers is None or max_frontiers < 0 or len(lengths) <= max_frontiers:
        return list(lengths)
    if max_frontiers == 0:
        return []
    if max_frontiers == 1:
        return [lengths[-1]]
    limited = [lengths[0]]
    limited.extend(lengths[-(max_frontiers - 1) :])
    return sorted(set(limited))


def str2bool(string):
    return string.lower() not in ["false", "f"]


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM inference script")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The path to the local model directory or Hugging Face repo. "
            f"If no model is specified, then {DEFAULT_MODEL} is used."
        ),
        default=None,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--extra-eos-token",
        type=str,
        default=(),
        nargs="+",
        help="Add tokens in the list of eos tokens that stop generation.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    parser.add_argument(
        "--prefill-response",
        default=None,
        help="Prefill response to be used for the chat template",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--min-p", type=float, default=DEFAULT_MIN_P, help="Sampling min-p"
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Sampling top-k"
    )
    parser.add_argument(
        "--xtc-probability",
        type=float,
        default=DEFAULT_XTC_PROBABILITY,
        help="Probability of XTC sampling to happen each next token",
    )
    parser.add_argument(
        "--xtc-threshold",
        type=float,
        default=0.0,
        help="Thresold the probs of each next token candidate to be sampled by XTC",
    )
    parser.add_argument(
        "--min-tokens-to-keep",
        type=int,
        default=DEFAULT_MIN_TOKENS_TO_KEEP,
        help="Minimum tokens to keep for min-p sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="PRNG seed",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use the raw prompt without the tokenizer's chat template.",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--chat-template-config",
        help="Additional config for `apply_chat_template`. Should be a dictionary of"
        " string keys to values represented as a JSON decodable string.",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="Log verbose output when 'True' or 'T' or only print the response when 'False' or 'F'",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--prompt-cache-file",
        type=str,
        default=None,
        help="A file containing saved KV caches to avoid recomputing them",
    )
    parser.add_argument(
        "--no-prompt-checkpoint",
        action="store_true",
        help="Disable automatic trusted local prompt checkpoint load/save.",
    )
    parser.add_argument(
        "--quantize-activations",
        "-qa",
        action="store_true",
        help="Quantize activations using the same quantization config as the corresponding layer.",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        help="Number of bits for KV cache quantization. Defaults to no quantization.",
        default=None,
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        help="Group size for KV cache quantization.",
        default=64,
    )
    parser.add_argument(
        "--quantized-kv-start",
        help="When --kv-bits is set, start quantizing the KV cache "
        "from this step onwards.",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding.",
        default=None,
    )
    parser.add_argument(
        "--mtp-speculative",
        action="store_true",
        help=(
            "Use the model's built-in GLM DSA MTP layer for speculative "
            "decoding. This is experimental and requires a GLM-5.2 checkpoint "
            "with native MTP weights."
        ),
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
        default=3,
    )
    return parser


# A stream on the default device just for generation
generation_stream = mx.new_thread_local_stream(mx.default_device())


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        try:
            yield
        finally:
            pass
    else:
        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
        )
        max_rec_size = mx.device_info()["max_recommended_working_set_size"]
        if model_bytes > 0.9 * max_rec_size:
            model_mb = model_bytes // 2**20
            max_rec_mb = max_rec_size // 2**20
            print(
                f"[WARNING] Generating with a model that requires {model_mb} MB "
                f"which is close to the maximum recommended size of {max_rec_mb} "
                "MB. This can be slow. See the documentation for possible work-arounds: "
                "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
            )
        old_limit = mx.set_wired_limit(max_rec_size)
        try:
            yield
        finally:
            if streams is not None:
                for s in streams:
                    mx.synchronize(s)
            else:
                mx.synchronize()
            mx.set_wired_limit(old_limit)


@dataclass
class GenerationResponse:
    """
    The output of :func:`stream_generate`.

    Args:
        text (str): The next segment of decoded text. This can be an empty string.
        token (int): The next token.
        from_draft (bool): Whether the token was generated by the draft model.
        logprobs (mx.array): A vector of log probabilities.
        prompt_tokens (int): The number of tokens in the prompt.
        prompt_tps (float): The prompt processing tokens-per-second.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        peak_memory (float): The peak memory used so far in GB.
        finish_reason (str): The reason the response is being sent: "length", "stop" or `None`
    """

    text: str
    token: int
    logprobs: mx.array
    from_draft: bool
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None


def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


def model_supports_mtp_speculative(model: nn.Module) -> bool:
    return (
        getattr(model, "mtp", None) is not None
        and hasattr(model, "forward_with_hidden")
        and hasattr(model, "make_mtp_cache")
        and hasattr(model, "mtp_logits")
    )


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    prefill_max_qk_tokens: Optional[int] = None,
    glm_dsa_adaptive_prefill_step_size: int = 0,
    glm_dsa_adaptive_prefill_after_tokens: int = 0,
    glm_dsa_adaptive_prefill_min_remaining_tokens: int = 0,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    input_embeddings: Optional[mx.array] = None,
    prompt_checkpoint: bool = True,
    prompt_checkpoint_full_prompt: Optional[Sequence[int]] = None,
    prompt_checkpoint_initial_cached_tokens: int = 0,
    prompt_checkpoint_initial_cache_source: str = "server-cache",
    prompt_checkpoint_store_prefix_lengths: Optional[Sequence[int]] = None,
    prompt_checkpoint_allow_existing_cache: bool = False,
    prompt_checkpoint_save_exact: bool = True,
    prompt_checkpoint_frontier_min_tokens: int = (
        PROMPT_CHECKPOINT_FRONTIER_MIN_TOKENS
    ),
    prompt_checkpoint_frontier_stride_tokens: int = (
        PROMPT_CHECKPOINT_FRONTIER_STRIDE_TOKENS
    ),
    prompt_checkpoint_rendered_prompt: Optional[Union[str, bytes]] = None,
    prompt_checkpoint_decode_prefix: Optional[
        Callable[[Sequence[int]], Optional[str]]
    ] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place.
        prefill_step_size (int): Step size for processing the prompt.
        prefill_max_qk_tokens (int, optional): If set, cap each prompt prefill
          chunk so ``chunk_tokens * effective_context_tokens`` stays below this
          value. This lets long-context dense fallback shrink before it OOMs.
        glm_dsa_adaptive_prefill_step_size (int): Optional larger base prefill
          step for ``glm_moe_dsa`` models. The context-aware QK cap still
          applies after this step is selected. Use ``0`` to disable.
        glm_dsa_adaptive_prefill_after_tokens (int): Minimum processed prompt
          tokens before the GLM DSA adaptive step can activate.
        glm_dsa_adaptive_prefill_min_remaining_tokens (int): Minimum remaining
          prompt tokens required for the GLM DSA adaptive step.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
           when ``kv_bits`` is non-None. Default: ``0``.
        prompt_progress_callback (Callable[[int, int], None]): A call-back which takes the
           prompt tokens processed so far and the total number of prompt tokens.
        input_embeddings (mx.array, optional): Input embeddings to use instead of or in
          conjunction with prompt tokens. Default: ``None``.
        prompt_checkpoint (bool): If ``True``, automatically load/save trusted
          local GLM-5.2 prompt checkpoints under
          ``~/.cache/mlx-lm/glm52-local/prompt-checkpoints``. Exact and
          longest-prefix hits skip matching prefill; misses fall back to normal
          prefill.
        prompt_checkpoint_full_prompt (Sequence[int], optional): Full tokenized
          prompt used for server-managed prefix lookup when ``prompt`` is only
          the uncached suffix.
        prompt_checkpoint_initial_cached_tokens (int): Number of leading full
          prompt tokens already represented by ``prompt_cache``.
        prompt_checkpoint_initial_cache_source (str): Human-readable source for
          ``prompt_checkpoint_initial_cached_tokens``. Server code uses this to
          distinguish RAM prompt cache reuse from disk rendered-checkpoint reuse
          in checkpoint accounting logs.
        prompt_checkpoint_store_prefix_lengths (Sequence[int], optional):
          Stable prefix lengths to save while prefill reaches those frontiers.
        prompt_checkpoint_allow_existing_cache (bool): Treat a supplied
          ``prompt_cache`` as server-managed and allow disk checkpoint lookup to
          replace it when a longer disk prefix exists.
        prompt_checkpoint_save_exact (bool): If ``True``, save the final exact
          prompt checkpoint after prefill. Set ``False`` to store configured
          prefix/frontier checkpoints without adding a full-prompt exact hit.
        prompt_checkpoint_frontier_min_tokens (int): First automatic long-prompt
          frontier to save. Defaults to 8192 tokens.
        prompt_checkpoint_frontier_stride_tokens (int): Token stride for
          automatic frontiers after the first one. Defaults to 16384 tokens.
        prompt_checkpoint_rendered_prompt (str or bytes, optional): Rendered
          prompt bytes used to attach ds4-style rendered-prefix lookup metadata
          to saved prefix/frontier checkpoints.
        prompt_checkpoint_decode_prefix (Callable, optional): Function used to
          decode token prefixes for rendered-prefix metadata. The decoded text
          must be an exact byte prefix of ``prompt_checkpoint_rendered_prompt``.

    Yields:
        Tuple[mx.array, mx.array]: One token and a vector of log probabilities.
    """
    checkpoint_full_prompt = _as_token_list(prompt_checkpoint_full_prompt)
    if checkpoint_full_prompt is None and input_embeddings is None:
        checkpoint_full_prompt = _as_token_list(prompt)
    prompt_checkpoint_existing_cache = (
        prompt_cache is not None
        and prompt_checkpoint_allow_existing_cache
        and input_embeddings is None
        and checkpoint_full_prompt is not None
        and len(checkpoint_full_prompt) > 0
    )

    if input_embeddings is not None:
        if not does_model_support_input_embeddings(model):
            raise ValueError("Model does not support input embeddings.")
        elif len(prompt) > 0 and len(prompt) != len(input_embeddings):
            raise ValueError(
                f"When providing input_embeddings, their sequence length ({len(input_embeddings)}) "
                f"must match the sequence length of the prompt ({len(prompt)}), or the "
                "prompt must be empty."
            )
    elif len(prompt) == 0 and not prompt_checkpoint_existing_cache:
        raise ValueError(
            "Either input_embeddings or prompt (or both) must be provided."
        )
    if (
        kv_bits is not None
        and kv_bits != 8
        and cache.model_has_glm_mla_kv_cache(model)
    ):
        raise ValueError("GLM MLA KV quantization supports only --kv-bits 8")

    tokens = None
    total_prompt_tokens = len(input_embeddings) if input_embeddings is not None else (
        len(checkpoint_full_prompt) if checkpoint_full_prompt is not None else len(prompt)
    )
    prompt_checkpoint_exact_tokens = checkpoint_full_prompt
    prompt_checkpoint_path = None
    prompt_checkpoint_hit = False
    prompt_checkpoint_hit_kind = None
    prompt_checkpoint_cached_tokens = 0
    prompt_checkpoint_disk_cached_tokens = 0
    prompt_checkpoint_resolution = "miss"
    prompt_checkpoint_lookup_stats = {
        "files_scanned": 0,
        "candidate_files_scanned": 0,
        "candidate_lengths_scanned": 0,
        "prefix_hashes_computed": 0,
        "matched_candidates": 0,
        "manifest_entries": 0,
        "manifest_loaded": False,
        "manifest_missing": False,
        "manifest_malformed": False,
        "manifest_bootstrap": False,
        "manifest_missing_entries_removed": 0,
        "lcp_manager_entries": 0,
        "lcp_manager_token_lengths": 0,
        "lcp_manager_block_lengths": 0,
        "lcp_block_hashes_computed": 0,
        "lcp_block_hash_matches": 0,
    }
    prompt_checkpoint_lookup_seconds = 0.0
    prompt_checkpoint_rendered_bytes = cache.rendered_prompt_bytes(
        prompt_checkpoint_rendered_prompt
    )
    prompt_checkpoint_initial_cached_tokens = max(
        0, min(int(prompt_checkpoint_initial_cached_tokens), total_prompt_tokens)
    )
    prompt_checkpoint_initial_cache_source = str(
        prompt_checkpoint_initial_cache_source or "server-cache"
    )
    if not prompt_checkpoint_existing_cache or prompt_checkpoint_initial_cached_tokens <= 0:
        prompt_checkpoint_initial_cache_source = "none"
    prompt_checkpoint_initial_cache_is_disk = (
        prompt_checkpoint_initial_cache_source.startswith("disk-")
    )

    def _prompt_checkpoint_initial_resolution():
        if prompt_checkpoint_initial_cache_source.startswith("disk-rendered-"):
            return prompt_checkpoint_initial_cache_source.replace(
                "disk-rendered-", "rendered-", 1
            )
        if prompt_checkpoint_initial_cache_is_disk:
            return prompt_checkpoint_initial_cache_source
        return "server-cache"

    def _prompt_checkpoint_initial_cache_counts():
        if not prompt_checkpoint_existing_cache:
            return 0, 0
        if prompt_checkpoint_initial_cache_is_disk:
            return 0, prompt_checkpoint_initial_cached_tokens
        return prompt_checkpoint_initial_cached_tokens, 0

    if (
        prompt_checkpoint_existing_cache
        and len(prompt) == 0
        and total_prompt_tokens > 0
        and prompt_checkpoint_initial_cached_tokens >= total_prompt_tokens
    ):
        if not cache.can_trim_prompt_cache(prompt_cache):
            raise ValueError("Fully cached prompt cannot be resumed with this cache.")
        if cache.trim_prompt_cache(prompt_cache, 1) != 1:
            raise ValueError("Fully cached prompt cannot be resumed with this cache.")
        prompt = mx.array(checkpoint_full_prompt[-1:])
        prompt_checkpoint_initial_cached_tokens = total_prompt_tokens - 1
    (
        prompt_checkpoint_server_cached_tokens,
        prompt_checkpoint_initial_disk_cached_tokens,
    ) = _prompt_checkpoint_initial_cache_counts()
    if prompt_checkpoint_existing_cache:
        prompt_checkpoint_cached_tokens = prompt_checkpoint_initial_cached_tokens
        prompt_checkpoint_disk_cached_tokens = prompt_checkpoint_initial_disk_cached_tokens

    def _checkpoint_cache_token_length_for_prefix(prefix_length):
        return (
            prefix_length - 1
            if prefix_length == total_prompt_tokens
            else prefix_length
        )

    def _expected_checkpoint_cache_layout(prefix_length):
        return cache.expected_prompt_cache_layout_signature(
            model,
            max_kv_size=max_kv_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
            cache_token_length=_checkpoint_cache_token_length_for_prefix(
                prefix_length
            ),
        )

    _prompt_checkpoint_debug(
        "request "
        f"total_prompt_tokens={total_prompt_tokens} "
        f"server_cached_tokens={prompt_checkpoint_server_cached_tokens} "
        f"disk_cached_tokens={prompt_checkpoint_disk_cached_tokens} "
        f"initial_cached_tokens={prompt_checkpoint_initial_cached_tokens if prompt_checkpoint_existing_cache else 0} "
        f"initial_cache_source={prompt_checkpoint_initial_cache_source} "
        f"prefill_step_size={prefill_step_size}"
    )

    if not prompt_checkpoint:
        prompt_checkpoint_resolution = "disabled"
        _prompt_checkpoint_debug("checkpoint disabled")
    elif input_embeddings is not None:
        prompt_checkpoint_resolution = "input-embeddings"
        _prompt_checkpoint_debug("skip input_embeddings active")
    elif prompt_cache is not None and not prompt_checkpoint_allow_existing_cache:
        prompt_checkpoint_resolution = "explicit-cache"
        _prompt_checkpoint_debug("skip explicit prompt_cache active")
    elif prompt_checkpoint_existing_cache:
        if prompt_checkpoint_initial_cache_is_disk:
            prompt_checkpoint_resolution = _prompt_checkpoint_initial_resolution()
        else:
            prompt_checkpoint_resolution = "server-cache"
        _prompt_checkpoint_debug(
            "initial prompt_cache coexistence "
            f"cached_tokens={prompt_checkpoint_initial_cached_tokens} "
            f"cache_source={prompt_checkpoint_initial_cache_source}"
        )
    elif total_prompt_tokens <= 1:
        prompt_checkpoint_resolution = "too-short"
        _prompt_checkpoint_debug(
            f"skip prefix too short prefix_length={total_prompt_tokens}"
        )

    if (
        prompt_checkpoint
        and input_embeddings is None
        and (prompt_cache is None or prompt_checkpoint_allow_existing_cache)
        and checkpoint_full_prompt is not None
        and total_prompt_tokens > 1
    ):
        prompt_checkpoint_path = cache.prompt_checkpoint_file(
            prompt_checkpoint_exact_tokens
        )
        prompt_checkpoint_basename = os.path.basename(prompt_checkpoint_path)
        _prompt_checkpoint_debug(
            "lookup "
            f"file={prompt_checkpoint_basename} "
            f"prefix_length={total_prompt_tokens} "
            f"server_cached_tokens={prompt_checkpoint_server_cached_tokens} "
            f"disk_cached_tokens={prompt_checkpoint_disk_cached_tokens} "
            f"initial_cached_tokens={prompt_checkpoint_initial_cached_tokens if prompt_checkpoint_existing_cache else 0} "
            f"initial_cache_source={prompt_checkpoint_initial_cache_source}"
        )
        lookup_t0 = time.perf_counter()
        all_candidates, prompt_checkpoint_lookup_stats = (
            cache.find_prompt_checkpoint_prefix(
                checkpoint_full_prompt,
                expected_cache_layout_by_length=(
                    _expected_checkpoint_cache_layout
                ),
                return_stats=True,
            )
        )
        prompt_checkpoint_lookup_seconds = time.perf_counter() - lookup_t0
        _prompt_checkpoint_debug(
            "lookup result "
            f"file={prompt_checkpoint_basename} "
            f"prefix_length={total_prompt_tokens} "
            f"files_scanned={prompt_checkpoint_lookup_stats['files_scanned']} "
            f"candidates_scanned={prompt_checkpoint_lookup_stats['candidate_files_scanned']} "
            f"candidate_lengths={prompt_checkpoint_lookup_stats['candidate_lengths_scanned']} "
            f"matched_candidates={prompt_checkpoint_lookup_stats['matched_candidates']} "
            f"prefix_hashes={prompt_checkpoint_lookup_stats['prefix_hashes_computed']} "
            f"manifest_entries={prompt_checkpoint_lookup_stats.get('manifest_entries', 0)} "
            f"manifest_loaded={int(prompt_checkpoint_lookup_stats.get('manifest_loaded', False))} "
            f"manifest_bootstrap={int(prompt_checkpoint_lookup_stats.get('manifest_bootstrap', False))} "
            f"manifest_missing_entries_removed={prompt_checkpoint_lookup_stats.get('manifest_missing_entries_removed', 0)} "
            f"lcp_index_entries={prompt_checkpoint_lookup_stats.get('lcp_manager_entries', 0)} "
            f"lcp_token_lengths={prompt_checkpoint_lookup_stats.get('lcp_manager_token_lengths', 0)} "
            f"lcp_block_lengths={prompt_checkpoint_lookup_stats.get('lcp_manager_block_lengths', 0)} "
            f"lcp_block_hashes={prompt_checkpoint_lookup_stats.get('lcp_block_hashes_computed', 0)} "
            f"lcp_block_matches={prompt_checkpoint_lookup_stats.get('lcp_block_hash_matches', 0)} "
            f"cache_layout_rejections={prompt_checkpoint_lookup_stats.get('cache_layout_rejections', 0)} "
            f"lookup_seconds={prompt_checkpoint_lookup_seconds:.6f}"
        )
        candidates = [
            candidate
            for candidate in all_candidates
            if candidate[0] > prompt_checkpoint_initial_cached_tokens
        ]
        if candidates:
            rejected = False
            for candidate_length, candidate_prefix, candidate_path in candidates:
                candidate_basename = os.path.basename(candidate_path)
                checkpoint_cache_token_length = (
                    _checkpoint_cache_token_length_for_prefix(candidate_length)
                )
                expected_glm_mla_kv_quantization = (
                    cache.expected_glm_mla_kv_quantization_metadata(
                        model,
                        cache_token_length=checkpoint_cache_token_length,
                        kv_bits=kv_bits,
                        kv_group_size=kv_group_size,
                        quantized_kv_start=quantized_kv_start,
                    )
                )
                expected_glm_mla_kv_settings = (
                    cache.expected_glm_mla_kv_settings_metadata(
                        model,
                        kv_bits=kv_bits,
                        kv_group_size=kv_group_size,
                        quantized_kv_start=quantized_kv_start,
                    )
                )
                expected_cache_layout = _expected_checkpoint_cache_layout(
                    candidate_length
                )
                load_t0 = time.perf_counter()
                try:
                    prompt_cache, checkpoint_metadata = cache.load_prompt_checkpoint(
                        candidate_path,
                        prefix_tokens=candidate_prefix,
                        checkpoint_namespace=cache.DEFAULT_PROMPT_CHECKPOINT_NAMESPACE,
                        model=model,
                        expected_glm_mla_kv_quantization=(
                            expected_glm_mla_kv_quantization
                        ),
                        expected_glm_mla_kv_settings=expected_glm_mla_kv_settings,
                        expected_cache_layout=expected_cache_layout,
                        return_metadata=True,
                    )
                    load_seconds = time.perf_counter() - load_t0
                    if candidate_length == total_prompt_tokens:
                        prompt = mx.array(checkpoint_full_prompt[-1:])
                        prompt_checkpoint_cached_tokens = total_prompt_tokens - 1
                        prompt_checkpoint_hit_kind = "exact"
                    else:
                        prompt = mx.array(checkpoint_full_prompt[candidate_length:])
                        prompt_checkpoint_cached_tokens = candidate_length
                        prompt_checkpoint_hit_kind = checkpoint_metadata.get(
                            "checkpoint_label",
                            "prefix",
                        )
                        if prompt_checkpoint_hit_kind not in (
                            "prefix",
                            "frontier",
                            "continued",
                        ):
                            prompt_checkpoint_hit_kind = "prefix"
                    prompt_checkpoint_disk_cached_tokens = (
                        prompt_checkpoint_cached_tokens
                    )
                    prompt_checkpoint_resolution = prompt_checkpoint_hit_kind
                    prompt_checkpoint_hit = True
                    _prompt_checkpoint_debug(
                        f"{prompt_checkpoint_hit_kind} hit "
                        f"file={candidate_basename} "
                        f"prefix_length={candidate_length} "
                        f"cached_tokens={prompt_checkpoint_cached_tokens} "
                        f"load_seconds={load_seconds:.6f}"
                    )
                    try:
                        manifest_report = cache.update_prompt_checkpoint_manifest(
                            candidate_path,
                            prefix_length=candidate_length,
                            kind=prompt_checkpoint_hit_kind,
                            metadata=checkpoint_metadata,
                            hit=True,
                        )
                        _prompt_checkpoint_debug(
                            "manifest hit update "
                            f"file={candidate_basename} "
                            f"kind={manifest_report['kind']} "
                            f"hit_count={manifest_report['hit_count']} "
                            f"entries={manifest_report['entries']}"
                        )
                    except Exception as exc:
                        _prompt_checkpoint_debug(
                            "manifest hit update failure swallowed "
                            f"file={candidate_basename} "
                            f"error={type(exc).__name__}"
                        )
                    break
                except cache.PromptCacheCheckpointError:
                    load_seconds = time.perf_counter() - load_t0
                    rejected = True
                    # A rejected checkpoint is a safe cache miss here: never reuse it.
                    _prompt_checkpoint_debug(
                        "miss rejected "
                        f"file={candidate_basename} "
                        f"prefix_length={candidate_length} "
                        f"load_seconds={load_seconds:.6f} "
                        "error=PromptCacheCheckpointError"
                    )
            if not prompt_checkpoint_hit and not rejected:
                prompt_checkpoint_resolution = "miss"
                _prompt_checkpoint_debug(
                    "miss no usable prefix "
                    f"file={prompt_checkpoint_basename} "
                    f"prefix_length={total_prompt_tokens}"
                )
        elif all_candidates:
            prompt_checkpoint_resolution = "server-cache-covered"
            if prompt_checkpoint_initial_cache_is_disk:
                prompt_checkpoint_resolution = _prompt_checkpoint_initial_resolution()
            _prompt_checkpoint_debug(
                "candidate covered by initial prompt_cache "
                f"file={os.path.basename(all_candidates[0][2])} "
                f"prefix_length={all_candidates[0][0]} "
                f"cached_tokens={prompt_checkpoint_initial_cached_tokens} "
                f"cache_source={prompt_checkpoint_initial_cache_source}"
            )
        else:
            prompt_checkpoint_resolution = "miss"
            _prompt_checkpoint_debug(
                "miss file does not exist "
                f"file={prompt_checkpoint_basename} "
                f"prefix_length={total_prompt_tokens}"
            )

    if not prompt_checkpoint_hit and prompt_checkpoint_resolution in (
        "server-cache",
        "miss",
    ):
        if prompt_checkpoint_initial_cached_tokens > 0:
            if prompt_checkpoint_initial_cache_is_disk:
                prompt_checkpoint_resolution = _prompt_checkpoint_initial_resolution()
            else:
                prompt_checkpoint_resolution = "server-cache"
    fresh_prompt_tokens = max(total_prompt_tokens - prompt_checkpoint_cached_tokens, 0)
    fresh_prefill_tokens = max(fresh_prompt_tokens - 1, 0)
    _prompt_checkpoint_debug(
        "prefill summary "
        f"total_prompt_tokens={total_prompt_tokens} "
        f"server_cached_tokens={prompt_checkpoint_server_cached_tokens} "
        f"disk_cached_tokens={prompt_checkpoint_disk_cached_tokens} "
        f"initial_cached_tokens={prompt_checkpoint_initial_cached_tokens if prompt_checkpoint_existing_cache else 0} "
        f"initial_cache_source={prompt_checkpoint_initial_cache_source} "
        f"fresh_prompt_tokens={fresh_prompt_tokens} "
        f"fresh_prefill_tokens={fresh_prefill_tokens} "
        f"prefill_step_size={prefill_step_size} "
        f"prefill_max_qk_tokens={prefill_max_qk_tokens} "
        f"glm_dsa_adaptive_prefill_step_size={glm_dsa_adaptive_prefill_step_size} "
        f"glm_dsa_adaptive_prefill_after_tokens={glm_dsa_adaptive_prefill_after_tokens} "
        f"glm_dsa_adaptive_prefill_min_remaining_tokens={glm_dsa_adaptive_prefill_min_remaining_tokens} "
        f"resolution={prompt_checkpoint_resolution} "
        f"files_scanned={prompt_checkpoint_lookup_stats['files_scanned']} "
        f"candidates_scanned={prompt_checkpoint_lookup_stats['candidate_files_scanned']} "
        f"matched_candidates={prompt_checkpoint_lookup_stats['matched_candidates']} "
        f"cache_layout_rejections={prompt_checkpoint_lookup_stats.get('cache_layout_rejections', 0)} "
        f"manifest_entries={prompt_checkpoint_lookup_stats.get('manifest_entries', 0)} "
        f"manifest_bootstrap={int(prompt_checkpoint_lookup_stats.get('manifest_bootstrap', False))} "
        f"lookup_seconds={prompt_checkpoint_lookup_seconds:.6f}"
    )

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )

    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    prompt_checkpoint_pending_store_lengths = []
    prompt_checkpoint_store_labels = {}
    if prompt_checkpoint_path is not None and prompt_checkpoint_exact_tokens is not None:
        prompt_checkpoint_frontier_lengths = _checkpoint_frontier_lengths(
            total_prompt_tokens,
            prompt_checkpoint_frontier_min_tokens,
            prompt_checkpoint_frontier_stride_tokens,
        )
        max_frontiers = _prompt_checkpoint_env_int(
            PROMPT_CHECKPOINT_MAX_FRONTIERS_PER_RUN_ENV,
            PROMPT_CHECKPOINT_MAX_FRONTIERS_PER_RUN,
        )
        limited_frontier_lengths = _checkpoint_limit_frontier_lengths(
            prompt_checkpoint_frontier_lengths,
            max_frontiers,
        )
        if len(limited_frontier_lengths) != len(prompt_checkpoint_frontier_lengths):
            _prompt_checkpoint_debug(
                "frontier schedule capped "
                f"requested={len(prompt_checkpoint_frontier_lengths)} "
                f"scheduled={len(limited_frontier_lengths)} "
                f"max_frontiers={max_frontiers}"
            )
        for length in limited_frontier_lengths:
            prompt_checkpoint_store_labels[length] = "frontier"
        for length in _checkpoint_store_lengths(
            prompt_checkpoint_store_prefix_lengths,
            total_prompt_tokens,
        ):
            prompt_checkpoint_store_labels[length] = "prefix"
        prompt_checkpoint_pending_store_lengths = sorted(
            length
            for length in prompt_checkpoint_store_labels
            if length > prompt_checkpoint_cached_tokens
        )
        if prompt_checkpoint_pending_store_lengths:
            _prompt_checkpoint_debug(
                "store schedule "
                f"prefix_lengths={prompt_checkpoint_pending_store_lengths}"
            )

    def _expected_checkpoint_metadata(prefix_length):
        checkpoint_cache_token_length = _checkpoint_cache_token_length_for_prefix(
            prefix_length
        )
        return (
            cache.expected_glm_mla_kv_quantization_metadata(
                model,
                cache_token_length=checkpoint_cache_token_length,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
            ),
            cache.expected_glm_mla_kv_settings_metadata(
                model,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
            ),
            _expected_checkpoint_cache_layout(prefix_length),
        )

    def _update_manifest_for_checkpoint(
        checkpoint_path,
        prefix_length,
        label,
        checkpoint_metadata,
        *,
        hit=False,
    ):
        basename = os.path.basename(checkpoint_path)
        try:
            manifest_report = cache.update_prompt_checkpoint_manifest(
                checkpoint_path,
                prefix_length=prefix_length,
                kind=label,
                metadata=checkpoint_metadata,
                hit=hit,
            )
            _prompt_checkpoint_debug(
                "manifest update "
                f"file={basename} "
                f"kind={manifest_report['kind']} "
                f"entries={manifest_report['entries']} "
                f"size_bytes={manifest_report['size_bytes']} "
                f"hit_count={manifest_report['hit_count']}"
            )
        except Exception as exc:
            _prompt_checkpoint_debug(
                "manifest update failure swallowed "
                f"file={basename} "
                f"kind={label} "
                f"error={type(exc).__name__}"
            )

    def _prune_prompt_checkpoints(checkpoint_path, label):
        basename = os.path.basename(checkpoint_path)
        protected_files = (
            [basename] if label in ("frontier", "prefix", "continued") else []
        )
        try:
            prune_report = cache.prune_prompt_checkpoints(
                protected_files=protected_files
            )
            removed = prune_report["removed"]
            _prompt_checkpoint_debug(
                "manifest prune "
                f"removed={len(removed)} "
                f"total_files={prune_report['total_files']} "
                f"total_bytes={prune_report['total_bytes']} "
                f"max_files={prune_report['max_files']} "
                f"max_bytes={prune_report['max_bytes']} "
                f"protected={len(protected_files)}"
            )
            for removed_entry in removed[:8]:
                _prompt_checkpoint_debug(
                    "manifest prune removed "
                    f"file={removed_entry['filename']} "
                    f"kind={removed_entry['kind']} "
                    f"prefix_length={removed_entry['prefix_length']} "
                    f"size_bytes={removed_entry['size_bytes']} "
                    f"status={removed_entry['status']}"
                )
        except Exception as exc:
            _prompt_checkpoint_debug(
                "manifest prune failure swallowed "
                f"file={basename} "
                f"kind={label} "
                f"error={type(exc).__name__}"
            )

    def _rendered_prefix_for_checkpoint(prefix_tokens):
        if (
            prompt_checkpoint_rendered_bytes is None
            or prompt_checkpoint_decode_prefix is None
        ):
            return None
        try:
            rendered_prefix = prompt_checkpoint_decode_prefix(prefix_tokens)
        except Exception as exc:
            _prompt_checkpoint_debug(
                "rendered metadata skipped decode failure "
                f"prefix_length={len(prefix_tokens)} "
                f"error={type(exc).__name__}"
            )
            return None
        rendered_prefix_bytes = cache.rendered_prompt_bytes(rendered_prefix)
        if not rendered_prefix_bytes:
            _prompt_checkpoint_debug(
                "rendered metadata skipped empty prefix "
                f"prefix_length={len(prefix_tokens)}"
            )
            return None
        if not prompt_checkpoint_rendered_bytes.startswith(rendered_prefix_bytes):
            _prompt_checkpoint_debug(
                "rendered metadata skipped prefix mismatch "
                f"prefix_length={len(prefix_tokens)} "
                f"rendered_prefix_bytes={len(rendered_prefix_bytes)}"
            )
            return None
        return rendered_prefix

    def _checkpoint_save_metadata(prefix_tokens, label):
        metadata = {"checkpoint_label": label}
        rendered_prefix = _rendered_prefix_for_checkpoint(prefix_tokens)
        if rendered_prefix is None:
            return metadata
        metadata.update(
            cache.prompt_checkpoint_rendered_prefix_metadata(rendered_prefix)
        )
        metadata[cache.PROMPT_CHECKPOINT_PREFIX_TOKENS_METADATA_KEY] = (
            cache.prompt_checkpoint_prefix_tokens_metadata(prefix_tokens)
        )
        _prompt_checkpoint_debug(
            "rendered metadata attached "
            f"prefix_length={len(prefix_tokens)} "
            f"rendered_prefix_bytes={len(cache.rendered_prompt_bytes(rendered_prefix))}"
        )
        return metadata

    def _existing_frontier_checkpoint_is_valid(prefix_tokens, checkpoint_path, label):
        if label != "frontier" or not os.path.exists(checkpoint_path):
            return False
        prefix_length = len(prefix_tokens)
        (
            expected_quantization,
            expected_settings,
            expected_cache_layout,
        ) = _expected_checkpoint_metadata(prefix_length)
        basename = os.path.basename(checkpoint_path)
        try:
            _, checkpoint_metadata = cache.load_prompt_checkpoint(
                checkpoint_path,
                prefix_tokens=prefix_tokens,
                checkpoint_namespace=cache.DEFAULT_PROMPT_CHECKPOINT_NAMESPACE,
                model=model,
                expected_glm_mla_kv_quantization=expected_quantization,
                expected_glm_mla_kv_settings=expected_settings,
                expected_cache_layout=expected_cache_layout,
                return_metadata=True,
            )
        except cache.PromptCacheCheckpointError:
            _prompt_checkpoint_debug(
                "save frontier existing rejected "
                f"file={basename} "
                f"prefix_length={prefix_length}"
            )
            return False
        _prompt_checkpoint_debug(
            "save frontier skipped existing valid "
            f"file={basename} "
            f"prefix_length={prefix_length}"
        )
        _update_manifest_for_checkpoint(
            checkpoint_path,
            prefix_length,
            label,
            checkpoint_metadata,
        )
        _prune_prompt_checkpoints(checkpoint_path, label)
        return True

    def _save_prompt_checkpoint(prefix_tokens, checkpoint_path, label):
        if _existing_frontier_checkpoint_is_valid(
            prefix_tokens,
            checkpoint_path,
            label,
        ):
            return True
        save_t0 = time.perf_counter()
        try:
            cache.ensure_glm52_local_cache_dirs()
            checkpoint_metadata = cache.save_prompt_checkpoint(
                checkpoint_path,
                prompt_cache,
                prefix_tokens=prefix_tokens,
                checkpoint_namespace=cache.DEFAULT_PROMPT_CHECKPOINT_NAMESPACE,
                model=model,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
                metadata=_checkpoint_save_metadata(prefix_tokens, label),
            )
            save_seconds = time.perf_counter() - save_t0
            _prompt_checkpoint_debug(
                f"save {label} success "
                f"file={os.path.basename(checkpoint_path)} "
                f"prefix_length={len(prefix_tokens)} "
                f"save_seconds={save_seconds:.6f}"
            )
            _update_manifest_for_checkpoint(
                checkpoint_path,
                len(prefix_tokens),
                label,
                checkpoint_metadata,
            )
            _prune_prompt_checkpoints(checkpoint_path, label)
            return True
        except Exception as exc:
            save_seconds = time.perf_counter() - save_t0
            _prompt_checkpoint_debug(
                f"save {label} failure swallowed "
                f"file={os.path.basename(checkpoint_path)} "
                f"prefix_length={len(prefix_tokens)} "
                f"save_seconds={save_seconds:.6f} "
                f"error={type(exc).__name__}"
            )
            return False

    def _model_call(input_tokens: mx.array, input_embeddings: Optional[mx.array]):
        if input_embeddings is not None:
            return model(
                input_tokens, cache=prompt_cache, input_embeddings=input_embeddings
            )
        else:
            return model(input_tokens, cache=prompt_cache)

    def _step(input_tokens: mx.array, input_embeddings: Optional[mx.array] = None):
        nonlocal tokens

        with mx.stream(generation_stream):
            logits = _model_call(
                input_tokens=input_tokens[None],
                input_embeddings=(
                    input_embeddings[None] if input_embeddings is not None else None
                ),
            )

            logits = logits[:, -1, :]

            if logits_processors and len(input_tokens) > 0:
                tokens = (
                    mx.concat([tokens, input_tokens])
                    if tokens is not None
                    else input_tokens
                )
                for processor in logits_processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = sampler(logprobs)
            return sampled, logprobs.squeeze(0)

    with mx.stream(generation_stream):
        prompt_processed_tokens = prompt_checkpoint_cached_tokens
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
        while total_prompt_tokens - prompt_processed_tokens > 1:
            remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
            adaptive_prefill_step_size = _glm_dsa_adaptive_prefill_step_size(
                model,
                requested_step_size=prefill_step_size,
                processed_tokens=prompt_processed_tokens,
                remaining_tokens=remaining,
                adaptive_step_size=glm_dsa_adaptive_prefill_step_size,
                adaptive_after_tokens=glm_dsa_adaptive_prefill_after_tokens,
                adaptive_min_remaining_tokens=(
                    glm_dsa_adaptive_prefill_min_remaining_tokens
                ),
            )
            n_to_process = _effective_prefill_step_size(
                prefill_step_size,
                remaining,
                prompt_processed_tokens,
                prefill_max_qk_tokens,
                adaptive_step_size=adaptive_prefill_step_size,
            )
            for store_length in prompt_checkpoint_pending_store_lengths:
                if prompt_processed_tokens < store_length <= (
                    prompt_processed_tokens + n_to_process
                ):
                    n_to_process = store_length - prompt_processed_tokens
                    break
            chunk_start_tokens = prompt_processed_tokens
            chunk_profile_before = _glm_dsa_prefill_profile_snapshot()
            chunk_t0 = time.perf_counter()
            _model_call(
                input_tokens=prompt[:n_to_process][None],
                input_embeddings=(
                    input_embeddings[:n_to_process][None]
                    if input_embeddings is not None
                    else None
                ),
            )
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            chunk_seconds = time.perf_counter() - chunk_t0
            chunk_profile_after = _glm_dsa_prefill_profile_snapshot()
            chunk_fields = _glm_dsa_prefill_profile_chunk_fields(
                chunk_profile_before,
                chunk_profile_after,
            )
            chunk_profile_suffix = _format_prefill_chunk_fields(chunk_fields)
            prompt_processed_tokens += n_to_process
            _prompt_checkpoint_debug(
                "prefill chunk "
                f"start_tokens={chunk_start_tokens} "
                f"chunk_tokens={n_to_process} "
                f"processed_tokens={prompt_processed_tokens} "
                f"total_prompt_tokens={total_prompt_tokens} "
                f"prefill_step_size={prefill_step_size} "
                f"adaptive_prefill_step_size={adaptive_prefill_step_size or 0} "
                f"effective_prefill_step_size={n_to_process} "
                f"prefill_max_qk_tokens={prefill_max_qk_tokens} "
                f"chunk_seconds={chunk_seconds:.6f}"
                f"{' ' + chunk_profile_suffix if chunk_profile_suffix else ''}"
            )
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt = prompt[n_to_process:]
            input_embeddings = (
                input_embeddings[n_to_process:]
                if input_embeddings is not None
                else input_embeddings
            )
            while (
                prompt_checkpoint_pending_store_lengths
                and prompt_checkpoint_pending_store_lengths[0]
                <= prompt_processed_tokens
            ):
                store_length = prompt_checkpoint_pending_store_lengths.pop(0)
                if (
                    prompt_checkpoint_path is None
                    or prompt_checkpoint_exact_tokens is None
                    or store_length != prompt_processed_tokens
                ):
                    continue
                store_prefix = prompt_checkpoint_exact_tokens[:store_length]
                _save_prompt_checkpoint(
                    store_prefix,
                    cache.prompt_checkpoint_file(store_prefix),
                    prompt_checkpoint_store_labels.get(store_length, "prefix"),
                )
            mx.clear_cache()

        if (
            prompt_checkpoint_path is not None
            and prompt_checkpoint_exact_tokens is not None
            and prompt_checkpoint_hit_kind != "exact"
        ):
            if prompt_checkpoint_save_exact:
                _save_prompt_checkpoint(
                    prompt_checkpoint_exact_tokens,
                    prompt_checkpoint_path,
                    "exact",
                )
            else:
                _prompt_checkpoint_debug(
                    "save exact skipped disabled "
                    f"file={os.path.basename(prompt_checkpoint_path)} "
                    f"prefix_length={len(prompt_checkpoint_exact_tokens)}"
                )

        decode_first_step_started_at = time.perf_counter()
        _prompt_checkpoint_debug(
            "decode first step start "
            f"total_prompt_tokens={total_prompt_tokens} "
            f"cached_tokens={prompt_checkpoint_cached_tokens} "
            f"remaining_prompt_tokens={prompt.size}"
        )
        y, logprobs = _step(input_tokens=prompt, input_embeddings=input_embeddings)
        _prompt_checkpoint_debug(
            "decode first step scheduled "
            f"schedule_seconds={time.perf_counter() - decode_first_step_started_at:.6f}"
        )

    mx.async_eval(y, logprobs)
    n = 0
    while True:
        if n != max_tokens:
            if n == 0:
                _prompt_checkpoint_debug("decode lookahead step start")
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs)
            if n == 0:
                _prompt_checkpoint_debug("decode lookahead step scheduled")
        if n == 0:
            mx.eval(y)
            _prompt_checkpoint_debug(
                "decode first token ready "
                f"first_token_seconds={time.perf_counter() - decode_first_step_started_at:.6f}"
            )
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        if n == max_tokens:
            break
        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()
        y, logprobs = next_y, next_logprobs
        n += 1


def mtp_speculative_generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    num_draft_tokens: int = 2,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Generator[Tuple[mx.array, mx.array, bool], None, None]:
    """
    Generate with a model's built-in GLM DSA MTP layer.

    This is an opt-in experimental path for GLM-5.2 checkpoints that include
    native MTP weights. It keeps the target and MTP caches separate and verifies
    every drafted token with the target model before yielding it.
    """
    if not model_supports_mtp_speculative(model):
        raise ValueError(
            "MTP speculative decoding requires a model loaded with "
            f"{GLM_DSA_MTP_ENV}=1 and native MTP weights."
        )
    if prompt_cache is not None:
        raise ValueError(
            "MTP speculative decoding does not support prompt_cache yet."
        )
    if kv_bits is not None and kv_bits != 8 and cache.model_has_glm_mla_kv_cache(
        model
    ):
        raise ValueError("GLM MLA KV quantization supports only --kv-bits 8")

    prompt = prompt.astype(mx.uint32).reshape(-1)
    if prompt.size == 0:
        raise ValueError("MTP speculative decoding requires a non-empty prompt.")
    if max_tokens == 0:
        return
    if num_draft_tokens <= 0:
        raise ValueError("--num-draft-tokens must be positive.")

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
    use_logits_processors = bool(logits_processors)
    history = prompt if use_logits_processors else None
    prompt_progress_callback = prompt_progress_callback or (lambda *_args: None)

    target_cache = cache.make_prompt_cache(model)
    mtp_cache_holder = [model.make_mtp_cache()]
    if not cache.can_trim_prompt_cache(target_cache):
        types = {type(c).__name__ for c in target_cache if not c.is_trimmable()}
        raise ValueError(
            "MTP speculative decoding requires a trimmable target cache "
            f"(got {types})."
        )
    if not cache.can_trim_prompt_cache(mtp_cache_holder):
        types = {type(c).__name__ for c in mtp_cache_holder if not c.is_trimmable()}
        raise ValueError(
            "MTP speculative decoding requires a trimmable MTP cache "
            f"(got {types})."
        )

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _append_history(tokens):
        nonlocal history
        if not use_logits_processors:
            return
        tokens = tokens.astype(mx.uint32).reshape(-1)
        history = mx.concatenate([history, tokens])

    def _history_with(base_history, tokens):
        if not use_logits_processors:
            return None
        tokens = tokens.astype(mx.uint32).reshape(-1)
        return mx.concatenate([base_history, tokens])

    def _process_and_sample(tokens, logits):
        if logits.ndim == 1:
            logits = logits[None, :]
        elif logits.ndim == 3:
            logits = logits[:, -1, :]
        if use_logits_processors:
            for processor in logits_processors:
                logits = processor(tokens, logits)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler(logprobs).astype(mx.uint32).reshape(-1)[:1]
        return y, logprobs.squeeze(0)

    def _eval_target(logits, hidden):
        mx.eval(logits, hidden, [c.state for c in target_cache])

    def _eval_mtp(logits, hidden):
        mx.eval(logits, hidden, mtp_cache_holder[0].state)

    step_size = max(1, int(prefill_step_size or 1))
    target_logits = None
    target_hidden = None
    processed = 0
    prompt_progress_callback(0, prompt.size)
    with mx.stream(generation_stream):
        while processed < prompt.size:
            n_to_process = min(step_size, prompt.size - processed)
            chunk = prompt[processed : processed + n_to_process][None]
            target_logits, target_hidden = model.forward_with_hidden(
                chunk,
                cache=target_cache,
            )
            quantize_cache_fn(target_cache)
            mtp_logits, mtp_hidden, _topk = model.mtp_logits(
                chunk,
                target_hidden,
                cache=mtp_cache_holder[0],
            )
            quantize_cache_fn(mtp_cache_holder)
            _eval_target(target_logits, target_hidden)
            _eval_mtp(mtp_logits, mtp_hidden)
            processed += n_to_process
            prompt_progress_callback(processed, prompt.size)
            mx.clear_cache()

        current, logprobs = _process_and_sample(history, target_logits[:, -1, :])
        previous_hidden = target_hidden[:, -1:, :]
        mx.async_eval(current, logprobs, previous_hidden)

    emitted = 0
    unbounded = max_tokens < 0

    def _can_emit():
        return unbounded or emitted < max_tokens

    if not _can_emit():
        return

    mx.eval(current)
    _append_history(current)
    emitted += 1
    yield current.item(), logprobs, False

    while _can_emit():
        remaining = num_draft_tokens + 1 if unbounded else max_tokens - emitted
        num_draft = min(num_draft_tokens, max(remaining - 1, 0))
        if num_draft == 0:
            with mx.stream(generation_stream):
                logits, hidden = model.forward_with_hidden(
                    current[None],
                    cache=target_cache,
                )
                quantize_cache_fn(target_cache)
                next_token, next_logprobs = _process_and_sample(
                    history, logits[:, -1, :]
                )
                next_hidden = hidden[:, -1:, :]
                _eval_target(logits, hidden)
                mx.async_eval(next_token, next_logprobs, next_hidden)
            mx.eval(next_token)
            current = next_token
            previous_hidden = next_hidden
            _append_history(current)
            emitted += 1
            yield current.item(), next_logprobs, False
            continue

        draft_tokens = []
        draft_input = current
        draft_hidden = previous_hidden
        draft_history = history
        with mx.stream(generation_stream):
            for _ in range(num_draft):
                mtp_logits, mtp_hidden, _topk = model.mtp_logits(
                    draft_input[None],
                    draft_hidden,
                    cache=mtp_cache_holder[0],
                )
                quantize_cache_fn(mtp_cache_holder)
                proposal, _proposal_logprobs = _process_and_sample(
                    draft_history,
                    mtp_logits[:, -1, :],
                )
                _eval_mtp(mtp_logits, mtp_hidden)
                mx.async_eval(proposal, mtp_hidden)
                draft_tokens.append(proposal)
                draft_input = proposal
                draft_hidden = mtp_hidden[:, -1:, :]
                if use_logits_processors:
                    draft_history = _history_with(draft_history, proposal)

            target_inputs = mx.concatenate([current] + draft_tokens, axis=0)[None]
            logits, hidden = model.forward_with_hidden(
                target_inputs,
                cache=target_cache,
            )
            quantize_cache_fn(target_cache)

        accepted = 0
        target_tokens = []
        target_logprobs = []
        target_history = history
        for i in range(num_draft + 1):
            token, token_logprobs = _process_and_sample(
                target_history,
                logits[:, i, :],
            )
            mx.eval(token)
            target_tokens.append(token)
            target_logprobs.append(token_logprobs)
            if use_logits_processors:
                target_history = _history_with(target_history, token)
            if i == num_draft:
                break
            if token.item() != draft_tokens[i].item():
                break
            accepted += 1

        _eval_target(logits, hidden)

        cache_trim = num_draft - accepted
        if cache_trim:
            cache.trim_prompt_cache(target_cache, cache_trim)

        mtp_trim = max(num_draft - accepted - 1, 0)
        if mtp_trim:
            cache.trim_prompt_cache(mtp_cache_holder, mtp_trim)

        for i in range(accepted):
            if not _can_emit():
                break
            current = target_tokens[i]
            _append_history(current)
            emitted += 1
            yield current.item(), target_logprobs[i], True

        if not _can_emit():
            break

        current = target_tokens[accepted]
        previous_hidden = hidden[:, accepted : accepted + 1, :]
        _append_history(current)
        emitted += 1
        yield current.item(), target_logprobs[accepted], False

        if accepted == num_draft:
            catch_prev_hidden = hidden[:, num_draft - 1 : num_draft, :]
            with mx.stream(generation_stream):
                catch_logits, catch_hidden, _topk = model.mtp_logits(
                    draft_tokens[-1][None],
                    catch_prev_hidden,
                    cache=mtp_cache_holder[0],
                )
                quantize_cache_fn(mtp_cache_holder)
                _eval_mtp(catch_logits, catch_hidden)


def speculative_generate_step(
    prompt: mx.array,
    model: nn.Module,
    draft_model: nn.Module,
    *,
    num_draft_tokens: int = 2,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
) -> Generator[Tuple[mx.array, mx.array, bool], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        draft_model (nn.Module): The draft model for speculative decoding.
        num_draft_tokens (int, optional): The number of draft tokens for
          speculative decoding. Default: ``2``.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Callable[[mx.array], mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place. The cache must be trimmable.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
           when ``kv_bits`` is non-None. Default: ``0``.

    Yields:
        Tuple[mx.array, mx.array, bool]: One token, a vector of log probabilities,
          and a bool indicating if the token was generated by the draft model
    """
    if kv_bits is not None and kv_bits != 8 and (
        cache.model_has_glm_mla_kv_cache(model)
        or cache.model_has_glm_mla_kv_cache(draft_model)
    ):
        raise ValueError("GLM MLA KV quantization supports only --kv-bits 8")

    y = prompt.astype(mx.uint32)
    prev_tokens = None

    # Create the KV cache for generation
    if prompt_cache is None:
        model_cache = cache.make_prompt_cache(model)
        draft_cache = cache.make_prompt_cache(draft_model)
    else:
        model_cache = prompt_cache[: len(model.layers)]
        draft_cache = prompt_cache[len(model.layers) :]

    if not cache.can_trim_prompt_cache(model_cache):
        types = {type(c).__name__ for c in model_cache if not c.is_trimmable()}
        raise ValueError(
            f"Speculative decoding requires a trimmable prompt cache " f"(got {types})."
        )

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _process_and_sample(tokens, logits):
        if logits_processors:
            for processor in logits_processors:
                logits = processor(tokens, logits)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler(logprobs)
        return y, logprobs

    def _step(model, cache, y, n_predict=1):
        with mx.stream(generation_stream):
            logits = model(y[None], cache=cache)
            logits = logits[:, -n_predict:, :]

            quantize_cache_fn(cache)
            if logits_processors:
                nonlocal prev_tokens
                out_y, out_logprobs = [], []
                if n_predict > 1:
                    y = y[: -(n_predict - 1)]
                for i in range(n_predict):
                    prev_tokens = (
                        mx.concatenate([prev_tokens, y])
                        if prev_tokens is not None
                        else y
                    )
                    y, logprobs = _process_and_sample(prev_tokens, logits[:, i, :])
                    out_y.append(y)
                    out_logprobs.append(logprobs)
                return mx.concatenate(out_y, axis=0), mx.concatenate(
                    out_logprobs, axis=0
                )
            else:
                return _process_and_sample(None, logits.squeeze(0))

    def _prefill(model, cache, y):
        while y.size > 1:
            n_to_process = min(prefill_step_size, y.size - 1)
            model(y[:n_to_process][None], cache=cache)
            quantize_cache_fn(cache)
            mx.eval([c.state for c in cache])
            y = y[n_to_process:]
            mx.clear_cache()
        return y

    def _rewind_cache(num_draft, num_accept):
        cache.trim_prompt_cache(model_cache, num_draft - num_accept)
        cache.trim_prompt_cache(draft_cache, max(num_draft - num_accept - 1, 0))

    def _draft_generate(y, num_draft):
        if num_draft == 0:
            return mx.array([], mx.uint32)
        ys = []
        for _ in range(num_draft):
            y, _ = _step(draft_model, draft_cache, y)
            mx.async_eval(y)
            ys.append(y)
        return mx.concatenate(ys)

    with mx.stream(generation_stream):
        draft_y = _prefill(draft_model, draft_cache, y)
        y = _prefill(model, model_cache, y)

    ntoks = 0
    # Set these so the finally block doesn't raise
    num_draft = 0
    n = 0
    try:
        while True:
            num_draft = min(max_tokens - ntoks, num_draft_tokens)
            draft_tokens = _draft_generate(draft_y, num_draft)
            if prev_tokens is not None:
                prev_tokens = prev_tokens[: prev_tokens.size - y.size - num_draft + 1]
            y = mx.concatenate([y, draft_tokens])
            tokens, logprobs = _step(model, model_cache, y, num_draft + 1)
            mx.eval(tokens, draft_tokens)
            draft_tokens = draft_tokens.tolist()
            tokens = tokens.tolist()
            n = 0
            while n < num_draft:
                tn, dtn, lpn = tokens[n], draft_tokens[n], logprobs[n]
                if tn != dtn:
                    break
                n += 1
                ntoks += 1
                yield tn, lpn, True
                if ntoks == max_tokens:
                    break
            if ntoks < max_tokens:
                ntoks += 1
                yield tokens[n], logprobs[n], False

            if ntoks == max_tokens:
                break

            y = mx.array([tokens[n]], mx.uint32)
            draft_y = y

            # If we accepted all the draft tokens, include the last
            # draft token in the next draft step since it hasn't been
            # processed yet by the draft model
            if n == num_draft:
                draft_y = mx.concatenate(
                    [mx.array(draft_tokens[-1:], mx.uint32), draft_y]
                )

            if prev_tokens is not None:
                prev_tokens = prev_tokens[: -max(num_draft - n, 1)]
            _rewind_cache(num_draft, n)
    finally:
        _rewind_cache(num_draft, n)


def stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    max_tokens: int = 256,
    draft_model: Optional[nn.Module] = None,
    **kwargs,
) -> Generator[GenerationResponse, None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        model (nn.Module): The model to use for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        prompt (Union[str, mx.array, List[int]]): The input prompt string or
          integer tokens.
        max_tokens (int): The maximum number of tokens to generate.
          Default: ``256``.
        draft_model (Optional[nn.Module]): An optional draft model. If provided
          then speculative decoding is used. The draft model must use the same
          tokenizer as the main model. Default: ``None``.
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        GenerationResponse: An instance containing the generated text segment and
            associated metadata. See :class:`GenerationResponse` for details.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            # Try to infer if special tokens are needed
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer

    kwargs["max_tokens"] = max_tokens
    if (
        kwargs.get("prompt_checkpoint_rendered_prompt") is not None
        and kwargs.get("prompt_checkpoint_decode_prefix") is None
    ):

        def decode_checkpoint_prefix(tokens):
            decode_kwargs = [
                {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
                {"skip_special_tokens": False},
                {},
            ]
            for decode_kwarg in decode_kwargs:
                try:
                    return tokenizer.decode(tokens, **decode_kwarg)
                except TypeError:
                    continue
            return tokenizer.decode(tokens)

        kwargs["prompt_checkpoint_decode_prefix"] = decode_checkpoint_prefix

    mtp_speculative = bool(kwargs.pop("mtp_speculative", False))
    if mtp_speculative and draft_model is not None:
        raise ValueError("mtp_speculative cannot be combined with draft_model.")

    if draft_model is None and not mtp_speculative:
        kwargs.pop("num_draft_tokens", None)
        token_generator = generate_step(prompt, model, **kwargs)
        # from_draft always false for non-speculative generation
        token_generator = (
            (token, logprobs, False) for token, logprobs in token_generator
        )
    elif mtp_speculative:
        if kwargs.get("prompt_checkpoint", True):
            _prompt_checkpoint_debug("skip mtp speculative path active")
        else:
            _prompt_checkpoint_debug("checkpoint disabled")
        for key in list(kwargs):
            if key == "prompt_checkpoint" or key.startswith("prompt_checkpoint_"):
                kwargs.pop(key, None)
        if kwargs.get("prompt_cache") is not None:
            raise ValueError("mtp_speculative does not support prompt_cache yet.")
        if kwargs.get("input_embeddings") is not None:
            raise ValueError("mtp_speculative does not support input_embeddings.")
        kwargs.pop("prompt_cache", None)
        kwargs.pop("input_embeddings", None)
        kwargs.pop("glm_dsa_adaptive_prefill_step_size", None)
        kwargs.pop("glm_dsa_adaptive_prefill_after_tokens", None)
        kwargs.pop("glm_dsa_adaptive_prefill_min_remaining_tokens", None)
        kwargs.pop("prefill_max_qk_tokens", None)
        kwargs.pop("max_kv_size", None)
        token_generator = mtp_speculative_generate_step(prompt, model, **kwargs)
    else:
        if kwargs.get("prompt_checkpoint", True):
            _prompt_checkpoint_debug("skip speculative path active")
        else:
            _prompt_checkpoint_debug("checkpoint disabled")
        for key in list(kwargs):
            if key == "prompt_checkpoint" or key.startswith("prompt_checkpoint_"):
                kwargs.pop(key, None)
        kwargs.pop("glm_dsa_adaptive_prefill_step_size", None)
        kwargs.pop("glm_dsa_adaptive_prefill_after_tokens", None)
        kwargs.pop("glm_dsa_adaptive_prefill_min_remaining_tokens", None)
        kwargs.pop("max_kv_size", None)
        kwargs.pop("prompt_progress_callback", None)
        token_generator = speculative_generate_step(
            prompt, model, draft_model, **kwargs
        )
    with wired_limit(model, [generation_stream]):
        tic = time.perf_counter()
        for n, (token, logprobs, from_draft) in enumerate(token_generator):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt.size / prompt_time
                tic = time.perf_counter()
            if token in tokenizer.eos_token_ids:
                break

            detokenizer.add_token(token)
            if (n + 1) == max_tokens:
                break

            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=None,
            )

        detokenizer.finalize()
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=from_draft,
            prompt_tokens=prompt.size,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
        )


def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, List[int]],
    verbose: bool = False,
    **kwargs,
) -> str:
    """
    Generate a complete response from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (Union[str, List[int]]): The input prompt string or integer tokens.
       verbose (bool): If ``True``, print tokens and timing information.
           Default: ``False``.
       kwargs: The remaining options get passed to :func:`stream_generate`.
          See :func:`stream_generate` for more details.
    """
    if verbose:
        print("=" * 10)

    text = ""
    for response in stream_generate(model, tokenizer, prompt, **kwargs):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text

    if verbose:
        print()
        print("=" * 10)
        if len(text) == 0:
            print("No text generated for this prompt")
            return
        print(
            f"Prompt: {response.prompt_tokens} tokens, "
            f"{response.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {response.generation_tokens} tokens, "
            f"{response.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {response.peak_memory:.3f} GB")
    return text


def _left_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)
    return mx.array([[0] * (max_length - len(p)) + p for p in prompts])


def _right_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)
    return mx.array([p + [0] * (max_length - len(p)) for p in prompts])


@dataclass
class BatchStats:
    """
    An data object to hold generation stats.

    Args:
        prompt_tokens (int): The number of prompt tokens processed.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_time (float): The time in seconds spent in prompt processing.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        generation_time (float): The time in seconds spent in generation .
        peak_memory (float): The peak memory used so far in GB.
    """

    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0


def _make_cache(model, left_padding, max_kv_size):
    """
    Convert a list of regular caches into their corresponding
    batch-aware caches.
    """

    def to_batch_cache(c):
        if type(c) is KVCache:
            return BatchKVCache(left_padding)
        elif isinstance(c, ArraysCache):
            c.left_padding = mx.array(left_padding)
            return c
        elif isinstance(c, RotatingKVCache):
            if c.keep > 0:
                raise ValueError("RotatingKVCache with keep tokens is not supported.")
            return BatchRotatingKVCache(c.max_size, left_padding)
        elif isinstance(c, CacheList):
            return CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if hasattr(model, "make_cache"):
        cache = model.make_cache()
        return [to_batch_cache(c) for c in cache]
    else:
        if max_kv_size is not None:
            return [
                BatchRotatingKVCache(max_kv_size, left_padding) for _ in model.layers
            ]
        return [BatchKVCache(left_padding) for _ in model.layers]


def _merge_caches(caches):
    batch_cache = []

    if not caches:
        return batch_cache

    for i in range(len(caches[0])):
        if hasattr(caches[0][i], "merge"):
            batch_cache.append(caches[0][i].merge([c[i] for c in caches]))
        else:
            raise ValueError(
                f"{type(caches[0][i])} does not yet support batching with history"
            )
    return batch_cache


def _extend_cache(cache_a, cache_b):
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    if not _glm_mla_kv_caches_compatible(cache_a, cache_b):
        raise ValueError(
            "Cannot merge quantized and unquantized GLM MLA KV caches in one "
            "continuous batch."
        )
    for ca, cb in zip(cache_a, cache_b):
        ca.extend(cb)
    return cache_a


_GLM_MLA_FP_CACHE_TYPES = {"GlmMlaKVCache", "BatchGlmMlaKVCache"}
_GLM_MLA_QUANTIZED_CACHE_TYPES = {
    "QuantizedGlmMlaKVCache",
    "BatchQuantizedGlmMlaKVCache",
}


def _iter_nested_caches(caches):
    for c in caches or []:
        if isinstance(c, CacheList):
            yield from _iter_nested_caches(c.caches)
        else:
            yield c


def _glm_mla_kv_cache_modes(caches):
    modes = set()
    for c in _iter_nested_caches(caches):
        cache_type = type(c).__name__
        if cache_type in _GLM_MLA_FP_CACHE_TYPES:
            modes.add("fp")
        elif cache_type in _GLM_MLA_QUANTIZED_CACHE_TYPES:
            modes.add("quantized")
    return modes


def _glm_mla_kv_caches_compatible(cache_a, cache_b):
    modes_a = _glm_mla_kv_cache_modes(cache_a)
    modes_b = _glm_mla_kv_cache_modes(cache_b)
    return not (
        ("quantized" in modes_a and "fp" in modes_b)
        or ("fp" in modes_a and "quantized" in modes_b)
    )


def _glm_mla_kv_caches_mixed(cache_a, cache_b):
    modes_a = _glm_mla_kv_cache_modes(cache_a)
    modes_b = _glm_mla_kv_cache_modes(cache_b)
    return (
        ("quantized" in modes_a and "fp" in modes_b)
        or ("fp" in modes_a and "quantized" in modes_b)
    )


def _glm_mla_kv_cache_kind(caches):
    modes = _glm_mla_kv_cache_modes(caches)
    if not modes:
        return "none"
    if len(modes) == 1:
        return next(iter(modes))
    return "mixed"


def _build_trie(sequences):
    """Build an Aho-Corasick trie from the provided sequences

    See https://en.wikipedia.org/wiki/Aho–Corasick_algorithm .
    """
    trie = {}
    for idx, seq in enumerate(sequences):
        node = trie
        try:
            for tok in seq:
                node = node.setdefault(tok, {})
            node["__match__"] = (tuple(seq), idx)
        except TypeError:
            node = node.setdefault(seq, {})
            node["__match__"] = ((seq,), idx)

    # BFS to set failure links and propagate matches.
    queue = deque()
    for key, child in trie.items():
        if key == "__match__":
            continue
        child["__fail__"] = trie
        queue.append(child)
    while queue:
        parent = queue.popleft()
        for key, child in parent.items():
            if key in ("__fail__", "__match__"):
                continue
            queue.append(child)
            fail = parent["__fail__"]
            while key not in fail and fail is not trie:
                fail = fail["__fail__"]
            child["__fail__"] = fail[key] if key in fail else trie
            if "__match__" not in child and "__match__" in child["__fail__"]:
                child["__match__"] = child["__fail__"]["__match__"]
    return trie


def _step_trie(node, trie, x):
    """One step in the Aho-Corasick trie."""
    while x not in node and node is not trie:
        node = node["__fail__"]
    if x in node:
        node = node[x]
    return node


class SequenceStateMachine:
    """A state machine that uses one Aho-Corasick trie per state to efficiently
    track state across a generated sequence.

    The transitions are provided as state -> [(sequence, new_state)].

    Example:

        sm = SequenceStateMachine(
            transitions={
                "normal": [
                    (think_start_tokens, "reasoning"),
                    (tool_start_tokens, "tool"),
                    (eos, None),
                ],
                "reasoning": [
                    (think_end_tokens, "normal"),
                    (eos, None),
                ],
                "tool": [
                    (tool_end_tokens, None),
                    (eos, None)
                ],
            },
            initial="normal"
        )
    """

    def __init__(self, transitions={}, initial="normal"):
        self._initial = initial
        self._states = {}
        for src, edges in transitions.items():
            sequences, dst = zip(*edges)
            self._states[src] = (_build_trie(sequences), dst)
        if not self._states:
            self._states[initial] = (_build_trie([]), [])

    def __deepcopy__(self, memo):
        new = object.__new__(SequenceStateMachine)
        new._initial = self._initial
        new._states = self._states
        return new

    def make_state(self):
        return (self._initial, self._states[self._initial][0], self._states)

    @staticmethod
    def match(state, x):
        s, n, states = state
        n = _step_trie(n, states[s][0], x)

        seq = None
        match = n.get("__match__")
        if match is not None:
            seq = match[0]
            s = states[s][1][match[1]]
            n = states[s][0] if s is not None else None

        return (s, n, states), seq, s


class PromptProcessingBatch:
    """
    A batch processor for prompt tokens with support for incremental processing.

    This class handles batched prompt processing, managing KV caches and preparing
    tokens for generation. It supports extending, filtering, and splitting batches.
    """

    @dataclass
    class Response:
        uid: int
        progress: tuple
        end_of_segment: bool
        end_of_prompt: bool

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        caches: List[List[Any]],
        tokens: Optional[List[List[int]]] = None,
        prefill_step_size: int = 2048,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        fallback_sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        state_machines: Optional[List[SequenceStateMachine]] = None,
        max_tokens: Optional[List[int]] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
    ):
        self.model = model
        self.uids = uids
        self.prompt_cache = _merge_caches(caches)
        self.tokens = tokens if tokens is not None else [[] for _ in uids]

        self.prefill_step_size = prefill_step_size
        self.samplers = samplers if samplers is not None else []
        self.fallback_sampler = fallback_sampler or (lambda x: mx.argmax(x, axis=-1))
        self.logits_processors = (
            logits_processors if logits_processors is not None else []
        )
        self.state_machines = (
            state_machines
            if state_machines is not None
            else [SequenceStateMachine()] * len(uids)
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else [DEFAULT_MAX_TOKENS] * len(self.uids)
        )
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start

    def __len__(self):
        return len(self.uids)

    def extract_cache(self, idx: int) -> List[Any]:
        return [c.extract(idx) for c in self.prompt_cache]

    def extend(self, batch):
        if not any(self.samplers):
            self.samplers = [None] * len(self.uids)
        if not any(self.logits_processors):
            self.logits_processors = [None] * len(self.uids)
        samplers = batch.samplers if any(batch.samplers) else [None] * len(batch.uids)
        logits_processors = (
            batch.logits_processors
            if any(batch.logits_processors)
            else [None] * len(batch.uids)
        )

        self.uids.extend(batch.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, batch.prompt_cache)
        self.tokens.extend(batch.tokens)
        self.samplers.extend(samplers)
        self.logits_processors.extend(logits_processors)
        self.max_tokens.extend(batch.max_tokens)
        self.state_machines.extend(batch.state_machines)

    def _copy(self):
        new_batch = self.__class__.__new__(self.__class__)
        new_batch.model = self.model
        new_batch.uids = list(self.uids)
        new_batch.prompt_cache = copy.deepcopy(self.prompt_cache)
        new_batch.tokens = list(self.tokens)
        new_batch.prefill_step_size = self.prefill_step_size
        new_batch.samplers = list(self.samplers)
        new_batch.fallback_sampler = self.fallback_sampler
        new_batch.logits_processors = list(self.logits_processors)
        new_batch.state_machines = list(self.state_machines)
        new_batch.max_tokens = list(self.max_tokens)
        new_batch.kv_bits = self.kv_bits
        new_batch.kv_group_size = self.kv_group_size
        new_batch.quantized_kv_start = self.quantized_kv_start
        return new_batch

    def _maybe_quantize_cache(self):
        maybe_quantize_kv_cache(
            self.prompt_cache,
            quantized_kv_start=self.quantized_kv_start,
            kv_group_size=self.kv_group_size,
            kv_bits=self.kv_bits,
        )

    def split(self, indices: List[int]):
        indices = sorted(indices)
        indices_left = sorted(set(range(len(self.uids))) - set(indices))
        new_batch = self._copy()
        self.filter(indices_left)
        new_batch.filter(indices)

        return new_batch

    def filter(self, keep: List[int]):
        self.uids = [self.uids[idx] for idx in keep]
        if not keep:
            self.prompt_cache.clear()
        else:
            for c in self.prompt_cache:
                c.filter(keep)
        self.tokens = [self.tokens[idx] for idx in keep]
        if any(self.samplers):
            self.samplers = [self.samplers[idx] for idx in keep]
        else:
            self.samplers = [None] * len(keep)
        if any(self.logits_processors):
            self.logits_processors = [self.logits_processors[idx] for idx in keep]
        else:
            self.logits_processors = [[]] * len(keep)
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self.state_machines = [self.state_machines[idx] for idx in keep]

    def prompt(self, tokens: List[List[int]]):
        """
        Process prompt tokens through the model.

        Args:
            tokens: List of token sequences to process.
        """
        if len(self.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")

        if not tokens:
            return

        # Add the tokens to the self.tokens so they represent the tokens
        # contained in the KV Cache.
        for sti, ti in zip(self.tokens, tokens):
            sti += ti

        # Calculate if we need to pad
        lengths = [len(p) for p in tokens]
        max_length = max(lengths)
        padding = [max_length - l for l in lengths]
        max_padding = max(padding)

        # Prepare the caches and inputs. Right pad if needed otherwise just
        # cast to array.
        if max_padding > 0:
            tokens = _right_pad_prompts(tokens, max_length=max_length)
            for c in self.prompt_cache:
                c.prepare(lengths=lengths, right_padding=padding)
        else:
            tokens = mx.array(tokens)

        # Actual prompt processing loop
        while tokens.shape[1] > 0:
            n_to_process = min(self.prefill_step_size, tokens.shape[1])
            self.model(tokens[:, :n_to_process], cache=self.prompt_cache)
            self._maybe_quantize_cache()
            mx.eval([c.state for c in self.prompt_cache])
            mx.clear_cache()
            tokens = tokens[:, n_to_process:]

        # Finalize the cache if there was any padding
        if max_padding > 0:
            for c in self.prompt_cache:
                c.finalize()
            mx.eval([c.state for c in self.prompt_cache])
            mx.clear_cache()

    def generate(self, tokens: List[List[int]]):
        """
        Transition from prompt processing to generation.

        Args:
            tokens: Final tokens for each sequence to start generation.

        Returns:
            A GenerationBatch ready for token generation.
        """
        if any(len(t) > 1 for t in tokens):
            self.prompt([t[:-1] for t in tokens])
        last_token = mx.array([t[-1] for t in tokens])

        generation = GenerationBatch(
            self.model,
            self.uids,
            last_token,
            self.prompt_cache,
            self.tokens,
            self.samplers,
            self.fallback_sampler,
            self.logits_processors,
            self.state_machines,
            self.max_tokens,
            self.kv_bits,
            self.kv_group_size,
            self.quantized_kv_start,
        )

        self.uids = []
        self.prompt_cache = []
        self.tokens = []
        self.samplers = []
        self.logits_processors = []
        self.max_tokens = []

        return generation

    @classmethod
    def empty(
        cls,
        model: nn.Module,
        fallback_sampler: Callable[[mx.array], mx.array],
        prefill_step_size: int = 2048,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
    ):
        return cls(
            model=model,
            fallback_sampler=fallback_sampler,
            prefill_step_size=prefill_step_size,
            uids=[],
            caches=[],
            tokens=[],
            samplers=[],
            logits_processors=[],
            max_tokens=[],
            state_machines=[],
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )


class GenerationBatch:
    """
    A batched token generator that manages multiple sequences in parallel.

    This class handles the generation phase after prompt processing, managing
    KV caches, sampling, and stop sequence detection for multiple sequences.
    """

    @dataclass
    class Response:
        uid: int
        token: int
        logprobs: mx.array
        finish_reason: Optional[str]
        current_state: Optional[str]
        match_sequence: Optional[List[int]]
        prompt_cache: Optional[List[Any]]
        all_tokens: Optional[List[int]]

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        inputs: mx.array,
        prompt_cache: List[Any],
        tokens: List[List[int]],
        samplers: Optional[List[Callable[[mx.array], mx.array]]],
        fallback_sampler: Callable[[mx.array], mx.array],
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ],
        state_machines: List[SequenceStateMachine],
        max_tokens: List[int],
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
    ):
        self.model = model
        self.uids = uids
        self.prompt_cache = prompt_cache
        self.tokens = tokens

        self.samplers = samplers
        self.fallback_sampler = fallback_sampler
        self.logits_processors = logits_processors
        self.state_machines = state_machines
        self.max_tokens = max_tokens
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start

        if self.samplers and len(self.samplers) != len(self.uids):
            raise ValueError("Insufficient number of samplers provided")
        if self.logits_processors and len(self.logits_processors) != len(self.uids):
            raise ValueError("Insufficient number of logits_processors provided")

        self._current_tokens = None
        self._current_logprobs = []
        self._next_tokens = inputs
        self._next_logprobs = []
        self._token_context = [TokenBuffer(t) for t in tokens]
        self._num_tokens = [0] * len(self.uids)
        self._matcher_states = [m.make_state() for m in state_machines]

        if self.uids:
            self._step()

    def __len__(self):
        return len(self.uids)

    def extend(self, batch):
        """Extend this batch with another generation batch."""
        self.uids.extend(batch.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, batch.prompt_cache)
        self.tokens.extend(batch.tokens)
        self.samplers.extend(batch.samplers)
        self.logits_processors.extend(batch.logits_processors)
        self.max_tokens.extend(batch.max_tokens)
        self.state_machines.extend(batch.state_machines)
        if self._current_tokens is None:
            self._current_tokens = batch._current_tokens
            self._current_logprobs = batch._current_logprobs
        elif batch._current_tokens is not None:
            self._current_tokens = mx.concatenate(
                [self._current_tokens, batch._current_tokens]
            )
            self._current_logprobs.extend(batch._current_logprobs)
        if self._next_tokens is None:
            self._next_tokens = batch._next_tokens
            self._next_logprobs = batch._next_logprobs
        elif batch._next_tokens is not None:
            self._next_tokens = mx.concatenate([self._next_tokens, batch._next_tokens])
            self._next_logprobs.extend(batch._next_logprobs)
        self._token_context.extend(batch._token_context)
        self._num_tokens.extend(batch._num_tokens)
        self._matcher_states.extend(batch._matcher_states)

    def _maybe_quantize_cache(self):
        maybe_quantize_kv_cache(
            self.prompt_cache,
            quantized_kv_start=self.quantized_kv_start,
            kv_group_size=self.kv_group_size,
            kv_bits=self.kv_bits,
        )

    def _step(self) -> Tuple[List[int], List[mx.array]]:
        """
        Perform a single generation step.

        Returns:
            Tuple of token list and logprobs list.
        """
        self._current_tokens = self._next_tokens
        self._current_logprobs = self._next_logprobs
        inputs = self._current_tokens

        # Forward pass
        logits = self.model(inputs[:, None], cache=self.prompt_cache)
        logits = logits[:, -1, :]
        self._maybe_quantize_cache()

        # Logits processors
        token_context = []
        if any(self.logits_processors):
            # Update the token context that will be used by the logits processors
            token_context = [
                tc.update_and_fetch(inputs[i : i + 1])
                for i, tc in enumerate(self._token_context)
            ]
            processed_logits = []
            for e in range(len(self.uids)):
                sample_logits = logits[e : e + 1]
                for processor in self.logits_processors[e]:
                    sample_logits = processor(token_context[e], sample_logits)
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        # Normalize the logits
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # Sample
        if any(self.samplers):
            all_samples = []
            for e in range(len(self.uids)):
                sample_sampler = self.samplers[e] or self.fallback_sampler
                sampled = sample_sampler(logprobs[e : e + 1])
                all_samples.append(sampled)
            sampled = mx.concatenate(all_samples, axis=0)
        else:
            sampled = self.fallback_sampler(logprobs)

        # Assign the next step to member variables and start computing it
        # asynchronously
        self._next_tokens = sampled
        self._next_logprobs = list(logprobs)
        mx.async_eval(self._next_tokens, self._next_logprobs, token_context)

        # Eval the current tokens and current logprobs. After that also add
        # them to self.tokens so that it always represents the tokens contained
        # in the KV Cache.
        mx.eval(inputs, self._current_logprobs)
        inputs = inputs.tolist()
        for sti, ti in zip(self.tokens, inputs):
            sti.append(ti)
        return inputs, self._current_logprobs

    def extract_cache(self, idx: int) -> List[Any]:
        return [c.extract(idx) for c in self.prompt_cache]

    def filter(self, keep: List[int]):
        """Filter the batch to keep only the specified indices."""
        self.uids = [self.uids[idx] for idx in keep]
        if not keep:
            self.prompt_cache.clear()
        else:
            for c in self.prompt_cache:
                c.filter(keep)
        self.tokens = [self.tokens[idx] for idx in keep]
        if any(self.samplers):
            self.samplers = [self.samplers[idx] for idx in keep]
        if any(self.logits_processors):
            self.logits_processors = [self.logits_processors[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self.state_machines = [self.state_machines[idx] for idx in keep]

        self._next_tokens = self._next_tokens[keep] if keep else None
        self._next_logprobs = [self._next_logprobs[idx] for idx in keep]
        self._token_context = [self._token_context[idx] for idx in keep]
        self._num_tokens = [self._num_tokens[idx] for idx in keep]
        self._matcher_states = [self._matcher_states[idx] for idx in keep]

    def next(self) -> List[Response]:
        """
        Generate the next batch of tokens.

        Returns:
            List of Response objects for each sequence in the batch.
        """
        if not self.uids:
            return []

        tokens, logprobs = self._step()

        keep = []
        responses = []
        for i in range(len(self.uids)):
            finish_reason = None
            match_sequence = None

            self._num_tokens[i] += 1
            if self._num_tokens[i] >= self.max_tokens[i]:
                finish_reason = "length"

            self._matcher_states[i], match_sequence, current_state = (
                self.state_machines[i].match(self._matcher_states[i], tokens[i])
            )
            if match_sequence is not None and current_state is None:
                finish_reason = "stop"

            if finish_reason is not None:
                responses.append(
                    self.Response(
                        uid=self.uids[i],
                        token=tokens[i],
                        logprobs=logprobs[i],
                        finish_reason=finish_reason,
                        current_state=current_state,
                        match_sequence=match_sequence,
                        prompt_cache=self.extract_cache(i),
                        all_tokens=self.tokens[i],
                    )
                )
            else:
                keep.append(i)
                responses.append(
                    self.Response(
                        uid=self.uids[i],
                        token=tokens[i],
                        logprobs=logprobs[i],
                        finish_reason=None,
                        match_sequence=match_sequence,
                        current_state=current_state,
                        prompt_cache=None,
                        all_tokens=None,
                    )
                )

        if len(keep) < len(self.uids):
            self.filter(keep)

        return responses

    @classmethod
    def empty(
        cls,
        model: nn.Module,
        fallback_sampler: Callable[[mx.array], mx.array],
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
    ):
        return cls(
            model=model,
            fallback_sampler=fallback_sampler,
            uids=[],
            inputs=mx.array([], dtype=mx.uint32),
            prompt_cache=[],
            tokens=[],
            samplers=[],
            logits_processors=[],
            max_tokens=[],
            state_machines=[],
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )


class BatchGenerator:
    """
    A batch generator implements continuous batching.

    This class provides automatic management of prompt processing and generation
    batches, handling the transition between the two.

    It also allows for segmented prompt processing which guarantees that the
    generator will stop at these boundaries when processing an input.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        max_tokens: int = 128,
        stop_tokens: Optional[Sequence[Sequence[int]]] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[
            List[Callable[[mx.array, mx.array], mx.array]]
        ] = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        stream=None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.logits_processors = logits_processors or []
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = max(completion_batch_size, prefill_batch_size)
        self.max_kv_size = max_kv_size
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start

        self._stream = stream or generation_stream

        self._default_state_machine = SequenceStateMachine(
            {"normal": [(seq, None) for seq in stop_tokens]} if stop_tokens else {},
            initial="normal",
        )
        self._uid_count = 0
        self._prompt_batch = PromptProcessingBatch.empty(
            self.model,
            self.sampler,
            prefill_step_size=prefill_step_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )
        self._generation_batch = GenerationBatch.empty(
            self.model,
            self.sampler,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )
        self._unprocessed_sequences = deque()
        self._currently_processing = []
        self._insert_times = {}
        self._waiting_for_compatible_batch = set()
        self._admission_counters = {
            "glm_mla_quantized_batch_admitted": 0,
            "glm_mla_quantized_batch_rejected_mixed_cache": 0,
            "glm_mla_waited_for_compatible_batch": 0,
        }
        self._admission_events = []

        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        self._steps_counter = 0

        if mx.metal.is_available():
            self._old_wired_limit = mx.set_wired_limit(
                mx.device_info()["max_recommended_working_set_size"]
            )
        else:
            self._old_wired_limit = None

    @property
    def stream(self):
        return self._stream

    def close(self):
        if self._old_wired_limit is not None:
            mx.synchronize(self._stream)
            mx.set_wired_limit(self._old_wired_limit)
            self._old_wired_limit = None

    def __del__(self):
        self.close()

    @contextlib.contextmanager
    def stats(self, stats=None):
        stats = stats or BatchStats()
        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        tic = time.perf_counter()
        try:
            yield stats
        finally:
            toc = time.perf_counter()
            total_time = toc - tic
            gen_time = total_time - self._prompt_time_counter
            stats.prompt_tokens += self._prompt_tokens_counter
            stats.prompt_time += self._prompt_time_counter
            stats.prompt_tps = stats.prompt_tokens / stats.prompt_time
            stats.generation_tokens += self._gen_tokens_counter
            stats.generation_time += gen_time
            stats.generation_tps = stats.generation_tokens / stats.generation_time
            stats.peak_memory = max(stats.peak_memory, mx.get_peak_memory() / 1e9)
            stats.admission_stats = self.admission_stats
            stats.admission_events = list(self._admission_events)

    def insert(
        self,
        prompts: List[List[int]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        state_machines: Optional[List[SequenceStateMachine]] = None,
    ):
        return self.insert_segments(
            [[p] for p in prompts],
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            state_machines,
        )

    def insert_segments(
        self,
        segments: List[List[List[int]]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        state_machines: Optional[List[SequenceStateMachine]] = None,
    ):
        uids = []

        max_tokens = max_tokens or [self.max_tokens] * len(segments)
        all_tokens = all_tokens or [[] for _ in segments]
        samplers = samplers or [None] * len(segments)
        logits_processors = logits_processors or (
            [self.logits_processors] * len(segments)
        )
        state_machines = state_machines or (
            [self._default_state_machine] * len(segments)
        )

        caches = caches or [None] * len(segments)
        for i in range(len(segments)):
            if caches[i] is None:
                caches[i] = self._make_new_cache()

        for seq, m, c, at, s, lp, sm in zip(
            segments,
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            state_machines,
        ):
            seq = list(seq)
            if len(seq[-1]) != 1:
                seq.append(seq[-1][-1:])
                seq[-2] = seq[-2][:-1]
            uid = self._uid_count
            self._unprocessed_sequences.append(
                (uid, seq, m, c, at, s, lp, sm)
            )
            self._insert_times[uid] = time.perf_counter()
            uids.append(uid)
            self._uid_count += 1

        return uids

    def _make_new_cache(self):
        if self.max_kv_size is None:
            return cache.make_prompt_cache(self.model)

        return [
            (
                RotatingKVCache(max_size=self.max_kv_size)
                if isinstance(ci, KVCache)
                else ci
            )
            for ci in cache.make_prompt_cache(self.model)
        ]

    def _find_uids(self, uids):
        uids = set(uids)
        results = {}
        for i, uid_i in enumerate(self._generation_batch.uids):
            if uid_i in uids:
                results[uid_i] = (2, i)
        for i, uid_i in enumerate(self._prompt_batch.uids):
            if uid_i in uids:
                results[uid_i] = (1, i)
        for i, seq in enumerate(self._unprocessed_sequences):
            if seq[0] in uids:
                results[seq[0]] = (0, i)
        return results

    def extract_cache(self, uids):
        results = {}
        for uid, (stage, idx) in self._find_uids(uids).items():
            if stage == 0:
                results[uid] = self._unprocessed_sequences[idx][3:5]
            elif stage == 1:
                results[uid] = (
                    self._prompt_batch.extract_cache(idx),
                    self._prompt_batch.tokens[idx],
                )
            else:
                results[uid] = (
                    self._generation_batch.extract_cache(idx),
                    self._generation_batch.tokens[idx],
                )
        return results

    def remove(self, uids, return_prompt_caches=False):
        caches = {}
        if return_prompt_caches:
            caches = self.extract_cache(uids)

        keep = (
            set(range(len(self._unprocessed_sequences))),
            set(range(len(self._prompt_batch))),
            set(range(len(self._generation_batch))),
        )
        for stage, idx in self._find_uids(uids).values():
            keep[stage].remove(idx)

        if len(keep[0]) < len(self._unprocessed_sequences):
            removed_uids = {
                seq[0]
                for i, seq in enumerate(self._unprocessed_sequences)
                if i not in keep[0]
            }
            self._unprocessed_sequences = deque(
                x for i, x in enumerate(self._unprocessed_sequences) if i in keep[0]
            )
            for uid in removed_uids:
                self._insert_times.pop(uid, None)
                self._waiting_for_compatible_batch.discard(uid)
        if len(keep[1]) < len(self._prompt_batch):
            self._prompt_batch.filter(sorted(keep[1]))
            self._currently_processing = [
                x for i, x in enumerate(self._currently_processing) if i in keep[1]
            ]
        if len(keep[2]) < len(self._generation_batch):
            self._generation_batch.filter(sorted(keep[2]))

        return caches

    @property
    def prompt_cache_nbytes(self):
        total = sum(c.nbytes for p in self._unprocessed_sequences for c in p[3])
        total += sum(c.nbytes for c in self._prompt_batch.prompt_cache)
        total += sum(c.nbytes for c in self._generation_batch.prompt_cache)
        return total

    @property
    def admission_stats(self):
        active_cache = (
            self._prompt_batch.prompt_cache + self._generation_batch.prompt_cache
        )
        return {
            **self._admission_counters,
            "active_batch_cache_kind": _glm_mla_kv_cache_kind(active_cache),
            "active_batch_size": len(self._prompt_batch)
            + len(self._generation_batch),
            "queued_request_count": len(self._unprocessed_sequences),
        }

    def _record_admission_event(
        self,
        *,
        uid: int,
        event: str,
        candidate_cache_kind: str,
        active_cache_kind: str,
    ):
        now = time.perf_counter()
        self._admission_events.append(
            {
                "uid": uid,
                "event": event,
                "wait_seconds": now - self._insert_times.get(uid, now),
                "candidate_cache_kind": candidate_cache_kind,
                "active_batch_cache_kind": active_cache_kind,
                "active_batch_size": len(self._prompt_batch)
                + len(self._generation_batch),
                "queued_request_count": len(self._unprocessed_sequences),
            }
        )

    def _make_batch(self, sequences):
        uids = []
        caches = []
        tokens = []
        samplers = []
        logits_processors = []
        max_tokens = []
        state_machines = []
        for sequence in sequences:
            uids.append(sequence[0])
            caches.append(sequence[3])
            tokens.append(sequence[4])
            samplers.append(sequence[5])
            logits_processors.append(sequence[6])
            max_tokens.append(sequence[2])
            state_machines.append(sequence[7])
            self._insert_times.pop(sequence[0], None)
            self._waiting_for_compatible_batch.discard(sequence[0])
            self._currently_processing.append(
                [sequence[1], 0, sum(len(s) for s in sequence[1])]
            )

        return PromptProcessingBatch(
            model=self.model,
            uids=uids,
            caches=caches,
            tokens=tokens,
            prefill_step_size=self.prefill_step_size,
            samplers=samplers,
            fallback_sampler=self.sampler,
            logits_processors=logits_processors,
            state_machines=state_machines,
            max_tokens=max_tokens,
            kv_bits=self.kv_bits,
            kv_group_size=self.kv_group_size,
            quantized_kv_start=self.quantized_kv_start,
        )

    def _can_admit_sequence(self, sequence):
        candidate_cache = sequence[3]
        return _glm_mla_kv_caches_compatible(
            self._prompt_batch.prompt_cache,
            candidate_cache,
        ) and _glm_mla_kv_caches_compatible(
            self._generation_batch.prompt_cache,
            candidate_cache,
        )

    def _can_admit_with_selected_sequences(self, sequence, selected):
        candidate_cache = sequence[3]
        return all(
            _glm_mla_kv_caches_compatible(candidate_cache, selected_sequence[3])
            for selected_sequence in selected
        )

    def _select_compatible_sequences(self, limit: int):
        if limit <= 0:
            return []

        selected = []
        remaining = deque()
        active_cache = (
            self._prompt_batch.prompt_cache + self._generation_batch.prompt_cache
        )
        active_cache_kind = _glm_mla_kv_cache_kind(active_cache)

        for sequence in self._unprocessed_sequences:
            candidate_cache_kind = _glm_mla_kv_cache_kind(sequence[3])
            can_admit = (
                len(selected) < limit
                and self._can_admit_sequence(sequence)
                and self._can_admit_with_selected_sequences(sequence, selected)
            )
            if can_admit:
                selected.append(sequence)
                if (
                    active_cache_kind == "quantized"
                    or candidate_cache_kind == "quantized"
                ):
                    self._admission_counters[
                        "glm_mla_quantized_batch_admitted"
                    ] += 1
                self._record_admission_event(
                    uid=sequence[0],
                    event="admitted",
                    candidate_cache_kind=candidate_cache_kind,
                    active_cache_kind=active_cache_kind,
                )
            else:
                remaining.append(sequence)
                if _glm_mla_kv_caches_mixed(active_cache, sequence[3]) or any(
                    _glm_mla_kv_caches_mixed(
                        selected_sequence[3],
                        sequence[3],
                    )
                    for selected_sequence in selected
                ):
                    self._admission_counters[
                        "glm_mla_quantized_batch_rejected_mixed_cache"
                    ] += 1
                    if sequence[0] not in self._waiting_for_compatible_batch:
                        self._waiting_for_compatible_batch.add(sequence[0])
                        self._admission_counters[
                            "glm_mla_waited_for_compatible_batch"
                        ] += 1
                    self._record_admission_event(
                        uid=sequence[0],
                        event="rejected_mixed_cache",
                        candidate_cache_kind=candidate_cache_kind,
                        active_cache_kind=active_cache_kind,
                    )

        self._unprocessed_sequences = remaining
        return selected

    def _next(self):
        generation_responses = []
        prompt_responses = []

        # Generate tokens first
        if len(self._generation_batch) > 0:
            generation_responses = self._generation_batch.next()
            self._gen_tokens_counter += len(generation_responses)
            self._steps_counter += 1
            if self._steps_counter % 512 == 0:
                mx.clear_cache()

        # Exit early because we already have our hands full with decoding
        if len(self._generation_batch) >= self.completion_batch_size:
            return prompt_responses, generation_responses

        # Check if we have sequences and add them to the prompt batch
        n = min(
            self.prefill_batch_size - len(self._prompt_batch),
            self.completion_batch_size - len(self._generation_batch),
            len(self._unprocessed_sequences),
        )
        selected = self._select_compatible_sequences(n)
        if selected:
            self._prompt_batch.extend(self._make_batch(selected))

        # Split the prompt sequences to the ones moving to generation and the rest
        keep = []
        split = []
        for i, seq in enumerate(self._currently_processing):
            segments = seq[0]
            if len(segments) == 1 and len(segments[0]) == 1:
                split.append(i)
            else:
                keep.append(i)

        # Actually split off part of the prompt batch and start generation
        if split:
            last_inputs = [self._currently_processing[i][0][0] for i in split]
            progress = [(self._currently_processing[i][2],) * 2 for i in split]
            self._currently_processing = [self._currently_processing[i] for i in keep]
            gen_batch = self._prompt_batch.split(split).generate(last_inputs)
            for i, p in enumerate(progress):
                prompt_responses.append(
                    PromptProcessingBatch.Response(
                        gen_batch.uids[i],
                        p,
                        True,
                        True,
                    )
                )
            self._generation_batch.extend(gen_batch)

        # Extract the next prompts input
        prompts = []
        for i, seq in enumerate(self._currently_processing):
            response = PromptProcessingBatch.Response(
                self._prompt_batch.uids[i], 0, False, False
            )
            segments = seq[0]
            n = min(len(segments[0]), self.prefill_step_size)
            prompts.append(segments[0][:n])
            segments[0] = segments[0][n:]
            if len(segments[0]) == 0:
                segments.pop(0)
                response.end_of_segment = True
            seq[1] += len(prompts[-1])
            response.progress = (seq[1], seq[2])
            prompt_responses.append(response)

        # Process the prompts
        self._prompt_tokens_counter += sum(len(p) for p in prompts)
        tic = time.perf_counter()
        self._prompt_batch.prompt(prompts)
        toc = time.perf_counter()
        self._prompt_time_counter += toc - tic

        return prompt_responses, generation_responses

    def next(self):
        """
        Get the next batch of responses.

        Returns:
            Tuple of prompt processing responses and generation responses.
        """
        with mx.stream(self._stream):
            return self._next()

    def next_generated(self):
        """
        Return only generated tokens ignoring batch generation responses.

        Returns:
            List of GenerationBatch.Response objects
        """
        with mx.stream(self._stream):
            while True:
                prompt_responses, generation_responses = self._next()
                if not generation_responses and prompt_responses:
                    continue
                return generation_responses


@dataclass
class BatchResponse:
    """
    A data object to hold a batch generation response.

    Args:
        texts: (List[str]): The generated text for each prompt.
        stats (BatchStats): Statistics about the generation.
        caches: Optional prompt caches for each sequence.
        token_ids (Optional[List[List[int]]]): The generated token IDs for each
            prompt. Only present when ``return_token_ids=True``.
        logprobs (Optional[List[List[float]]]): The per-token log-probabilities
            of the sampled tokens for each prompt. Only present when
            ``return_logprobs=True``.
    """

    texts: List[str]
    stats: BatchStats
    caches: Optional[List[List[Any]]]
    token_ids: Optional[List[List[int]]] = None
    logprobs: Optional[List[List[float]]] = None


def batch_generate(
    model,
    tokenizer,
    prompts: List[List[int]],
    prompt_caches: Optional[List[List[Any]]] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    return_prompt_caches: bool = False,
    return_token_ids: bool = False,
    return_logprobs: bool = False,
    **kwargs,
) -> BatchResponse:
    """
    Generate responses for the given batch of prompts.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompts (List[List[int]]): The input prompts.
       prompt_caches (List[List[Any]], optional): Pre-computed prompt-caches
          for each input prompt. Note, unlike ``generate_step``, the caches
          won't be updated in-place.
       verbose (bool): If ``True``, print tokens and timing information.
          Default: ``False``.
       max_tokens (Union[int, List[int]): Maximum number of output tokens. This
          can be per prompt if a list is provided.
       return_prompt_caches (bool): Return the prompt caches in the batch
          responses. Default: ``False``.
       return_token_ids (bool): Return the generated token IDs in the batch
          responses. Default: ``False``.
       return_logprobs (bool): Return the per-token log-probability of the
          sampled token for each generated token. Useful for reinforcement
          learning (e.g. RLOO, PPO) where behavior log-probabilities are needed
          for importance weighting. Default: ``False``.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.
    """

    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        **kwargs,
    )
    num_samples = len(prompts)
    fin = 0
    if verbose:
        print(f"[batch_generate] Finished processing 0/{num_samples} ...", end="\r")

    if isinstance(max_tokens, int):
        max_tokens = [max_tokens] * len(prompts)

    uids = gen.insert(prompts, max_tokens, caches=prompt_caches)
    results = {uid: [] for uid in uids}
    logprob_results = {uid: [] for uid in uids} if return_logprobs else None
    prompt_caches = {}
    with gen.stats() as stats:
        while responses := gen.next_generated():
            for r in responses:
                if r.finish_reason is not None:
                    if return_prompt_caches:
                        prompt_caches[r.uid] = r.prompt_cache
                    if verbose:
                        fin += 1
                        print(
                            f"[batch_generate] Finished processing {fin}/{num_samples} ...",
                            end="\r",
                        )
                if r.finish_reason != "stop":
                    results[r.uid].append(r.token)
                    if return_logprobs:
                        logprob_results[r.uid].append(r.logprobs[r.token].item())
    gen.close()
    if verbose:
        print(f"[batch_generate] Finished processing {fin}/{num_samples}")

    # Return results in correct order
    texts = [tokenizer.decode(results[uid]) for uid in uids]
    caches = [prompt_caches[uid] for uid in uids] if return_prompt_caches else None
    token_ids = [results[uid] for uid in uids] if return_token_ids else None
    logprobs = [logprob_results[uid] for uid in uids] if return_logprobs else None
    if verbose:
        print(
            f"[batch_generate] Prompt: {stats.prompt_tokens} tokens, {stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {stats.generation_tokens} tokens, "
            f"{stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {stats.peak_memory:.3f} GB")
    return BatchResponse(texts, stats, caches, token_ids, logprobs)


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.seed is not None:
        mx.random.seed(args.seed)

    # Load the prompt cache and metadata if a cache file is provided
    using_cache = args.prompt_cache_file is not None
    if using_cache:
        prompt_cache, metadata = load_prompt_cache(
            args.prompt_cache_file,
            return_metadata=True,
        )
        if isinstance(prompt_cache[0], QuantizedKVCache):
            if args.kv_bits is not None and args.kv_bits != prompt_cache[0].bits:
                raise ValueError(
                    "--kv-bits does not match the kv cache loaded from --prompt-cache-file."
                )
            if args.kv_group_size != prompt_cache[0].group_size:
                raise ValueError(
                    "--kv-group-size does not match the kv cache loaded from --prompt-cache-file."
                )

    # Building tokenizer_config
    tokenizer_config = (
        {} if not using_cache else json.loads(metadata["tokenizer_config"])
    )
    tokenizer_config["trust_remote_code"] = args.trust_remote_code

    model_path = args.model
    if using_cache:
        if model_path is None:
            model_path = metadata["model"]
        elif model_path != metadata["model"]:
            raise ValueError(
                f"Providing a different model ({model_path}) than that "
                f"used to create the prompt cache ({metadata['model']}) "
                "is an error."
            )
    model_path = model_path or DEFAULT_MODEL
    if args.mtp_speculative:
        if args.draft_model is not None:
            raise ValueError("--mtp-speculative cannot be combined with --draft-model.")
        if using_cache:
            raise ValueError(
                "--mtp-speculative cannot be combined with --prompt-cache-file."
            )
        os.environ[GLM_DSA_MTP_ENV] = "1"

    model, tokenizer = load(
        model_path,
        adapter_path=args.adapter_path,
        tokenizer_config=tokenizer_config,
        model_config={"quantize_activations": args.quantize_activations},
        trust_remote_code=args.trust_remote_code,
    )
    for eos_token in args.extra_eos_token:
        tokenizer.add_eos_token(eos_token)

    template_kwargs = {}
    if args.chat_template_config is not None:
        template_kwargs = json.loads(args.chat_template_config)

    prompt = args.prompt.replace("\\n", "\n").replace("\\t", "\t")
    prompt = sys.stdin.read() if prompt == "-" else prompt
    if not args.ignore_chat_template and tokenizer.has_chat_template:
        if args.system_prompt is not None:
            messages = [{"role": "system", "content": args.system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})

        has_prefill = args.prefill_response is not None
        if has_prefill:
            messages.append({"role": "assistant", "content": args.prefill_response})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=has_prefill,
            add_generation_prompt=not has_prefill,
            **template_kwargs,
        )

        # Treat the prompt as a suffix assuming that the prefix is in the
        # stored kv cache.
        if using_cache:
            messages[-1]["content"] = "<query>"
            test_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=has_prefill,
                add_generation_prompt=not has_prefill,
            )
            prompt = prompt[test_prompt.index("<query>") :]
        prompt = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        prompt = tokenizer.encode(prompt)

    if args.draft_model is not None:
        draft_model, draft_tokenizer = load(args.draft_model)
        if draft_tokenizer.vocab_size != tokenizer.vocab_size:
            raise ValueError("Draft model tokenizer does not match model tokenizer.")
    else:
        draft_model = None
    sampler = make_sampler(
        args.temp,
        args.top_p,
        args.min_p,
        args.min_tokens_to_keep,
        top_k=args.top_k,
        xtc_probability=args.xtc_probability,
        xtc_threshold=args.xtc_threshold,
        xtc_special_tokens=tokenizer.encode("\n") + list(tokenizer.eos_token_ids),
    )
    response = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        sampler=sampler,
        max_kv_size=args.max_kv_size,
        prompt_cache=prompt_cache if using_cache else None,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        prompt_checkpoint=not args.no_prompt_checkpoint,
        draft_model=draft_model,
        mtp_speculative=args.mtp_speculative,
        num_draft_tokens=args.num_draft_tokens,
    )
    if not args.verbose:
        print(response)


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.generate...` directly is deprecated."
        " Use `mlx_lm.generate...` or `python -m mlx_lm generate ...` instead."
    )
    main()
