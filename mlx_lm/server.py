# Copyright © 2023-2024 Apple Inc.

import argparse
import copy
import json
import logging
import os
import pickle
import platform
import socket
import time
import uuid
from collections import deque
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty as QueueEmpty
from queue import Queue
from threading import Thread
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import mlx.core as mx
from huggingface_hub import scan_cache_dir

from ._version import __version__
from .generate import (
    BatchGenerator,
    DEFAULT_PREFILL_MAX_QK_TOKENS,
    SequenceStateMachine,
    _prompt_checkpoint_debug,
    stream_generate,
)
from .models.cache import (
    DEFAULT_PROMPT_CHECKPOINT_MAX_AGE_SECONDS,
    LRUPromptCache,
    PROMPT_CHECKPOINT_CACHE_DIR_ENV,
    PROMPT_CHECKPOINT_DELTA_BASE_FILENAME_METADATA_KEY,
    PROMPT_CHECKPOINT_DELTA_BASE_PREFIX_HASH_METADATA_KEY,
    PROMPT_CHECKPOINT_DELTA_BASE_PREFIX_LENGTH_METADATA_KEY,
    PROMPT_CHECKPOINT_DELTA_CACHE_START_TOKENS_METADATA_KEY,
    PROMPT_CHECKPOINT_DELTA_CACHE_TOKENS_METADATA_KEY,
    PROMPT_CHECKPOINT_PREFIX_TOKENS_METADATA_KEY,
    DEFAULT_PROMPT_CHECKPOINT_NAMESPACE,
    PromptCacheCheckpointError,
    can_trim_prompt_cache,
    concat_prompt_caches,
    expected_glm_mla_kv_quantization_metadata,
    expected_glm_mla_kv_settings_metadata,
    expected_prompt_cache_layout_signature,
    find_prompt_checkpoint_rendered_prefix,
    load_prompt_checkpoint_with_metadata_prefix,
    make_prompt_cache,
    model_has_glm_mla_kv_cache,
    prompt_checkpoint_file,
    prompt_checkpoint_prefix_tokens_metadata,
    prompt_prefix_hash,
    prompt_cache_token_length,
    prompt_checkpoint_rendered_prefix_metadata,
    prune_prompt_checkpoints,
    rendered_prompt_bytes,
    save_prompt_checkpoint,
    slice_prompt_cache,
    trim_prompt_cache,
    update_prompt_checkpoint_manifest,
)
from .sample_utils import make_logits_processors, make_sampler
from .utils import _parse_size, load, sharded_load


DEFAULT_PROMPT_CHECKPOINT_MIN_TOKENS = 512
DEFAULT_PROMPT_CHECKPOINT_COLD_MAX_TOKENS = 30_000
DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_TRIM_TOKENS = 32
DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_ALIGN_TOKENS = 2048
DEFAULT_PROMPT_CHECKPOINT_CONTINUED_INTERVAL_TOKENS = 10_000
DEFAULT_PROMPT_CHECKPOINT_SHUTDOWN_SAVE_LIMIT = 0
DEFAULT_PROMPT_CHECKPOINT_SHUTDOWN_MAX_TOKENS = 65_536
DEFAULT_GENERATION_SHUTDOWN_TIMEOUT_SECONDS = 0.0
DEFAULT_PROMPT_CHECKPOINT_ASYNC_SHUTDOWN_TIMEOUT_SECONDS = 0.0


def get_system_fingerprint():
    gpu_arch = mx.device_info()["architecture"]
    return f"{__version__}-{mx.__version__}-{platform.platform()}-{gpu_arch}"


class ToolCallFormatter:
    def __init__(self, tool_parser, tools, streaming=False):
        self._idx = 0
        self._tool_parser = tool_parser
        self._tools = tools
        self._streaming = streaming

    def _format(self, tc):
        tc_id = tc.pop("id", None) or str(uuid.uuid4())
        tc["arguments"] = json.dumps(tc["arguments"], ensure_ascii=False)
        out = {
            "function": tc,
            "type": "function",
            "id": tc_id,
        }
        if self._streaming:
            out["index"] = self._idx
            self._idx += 1
        return out

    def __call__(self, tool_calls):
        if not tool_calls:
            return []

        result = []
        for tool_text in tool_calls:
            try:
                parsed = self._tool_parser(tool_text, self._tools)
            except (ValueError, json.JSONDecodeError) as e:
                logging.warning(
                    f"Failed to parse tool call ({type(e).__name__}: {e}) — "
                    f"tool text was likely truncated mid-generation."
                )
                continue
            if not isinstance(parsed, list):
                parsed = [parsed]
            result.extend(self._format(tc) for tc in parsed)
        return result


def convert_chat(messages: List[dict], role_mapping: Optional[dict] = None):
    default_role_mapping = {
        "system_prompt": (
            "A chat between a curious user and an artificial intelligence "
            "assistant. The assistant follows the given rules no matter what."
        ),
        "system": "ASSISTANT's RULE: ",
        "user": "USER: ",
        "assistant": "ASSISTANT: ",
        "stop": "\n",
    }
    role_mapping = role_mapping or default_role_mapping

    prompt = ""
    for line in messages:
        role_prefix = role_mapping.get(line["role"], "")
        stop = role_mapping.get("stop", "")
        content = line.get("content", "")
        prompt += f"{role_prefix}{content}{stop}"

    prompt += role_mapping.get("assistant", "")
    return prompt.rstrip()


def process_message_content(messages):
    """
    Convert message content to a format suitable for `apply_chat_template`.

    The function operates on messages in place. It converts the 'content' field
    to a string instead of a list of text fragments.

    Args:
        message_list (list): A list of dictionaries, where each dictionary may
          have a 'content' key containing a list of dictionaries with 'type' and
          'text' keys.

    Raises:
        ValueError: If the 'content' type is not supported or if 'text' is missing.

    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text_fragments = [
                fragment["text"]
                for fragment in content
                if fragment.get("type") in ("text", "input_text", "output_text")
            ]
            if len(text_fragments) != len(content):
                raise ValueError("Only 'text' content type is supported.")
            message["content"] = "".join(text_fragments)
        elif content is None:
            message["content"] = ""

        if tool_calls := message.get("tool_calls"):
            for tool_call in tool_calls:
                if func := tool_call.get("function"):
                    if args := func.get("arguments"):
                        if isinstance(args, str):
                            func["arguments"] = json.loads(args)


@dataclass
class ModelDescription:
    model: str
    draft: str
    adapter: str


@dataclass
class SamplingArguments:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    xtc_probability: float
    xtc_threshold: float


@dataclass
class LogitsProcessorArguments:
    logit_bias: Optional[Dict[int, float]]
    repetition_penalty: float
    repetition_context_size: int
    presence_penalty: float
    presence_context_size: int
    frequency_penalty: float
    frequency_context_size: int


@dataclass
class GenerationArguments:
    model: ModelDescription
    sampling: SamplingArguments
    logits: LogitsProcessorArguments

    stop_words: List[str]

    max_tokens: int
    num_draft_tokens: int
    logprobs: bool
    top_logprobs: int
    seed: Optional[int]
    chat_template_kwargs: Optional[Dict[str, Any]]


@dataclass
class CompletionRequest:
    request_type: Literal["chat", "text"]

    prompt: str

    messages: List[Any]
    tools: Optional[List[Any]]
    role_mapping: Optional[Dict[str, Any]]


@dataclass
class GenerationContext:
    has_tool_calling: bool
    has_thinking: bool
    tool_parser: Callable[[str, Any], Dict]

    sequences: Dict[Tuple[int], str]

    prompt: List[int]
    prompt_cache_count: int = -1

    _should_stop: bool = False

    def stop(self):
        self._should_stop = True


def _resolve_request_max_tokens(body, cli_args):
    if body.get("max_completion_tokens", None) is not None:
        max_tokens = body["max_completion_tokens"]
        max_tokens_source = "max_completion_tokens"
    elif "max_tokens" in body:
        max_tokens = body["max_tokens"]
        max_tokens_source = "max_tokens"
    else:
        max_tokens = cli_args.max_tokens
        max_tokens_source = "cli_default"

    requested_max_tokens = max_tokens
    floor = getattr(cli_args, "request_max_tokens_floor", 0) or 0
    floor_applied = isinstance(max_tokens, int) and floor > 0 and max_tokens < floor
    if floor_applied:
        max_tokens = floor
    return max_tokens, max_tokens_source, requested_max_tokens, floor_applied


@dataclass
class RenderedPromptCheckpoint:
    prompt_cache: List[Any]
    prefix_tokens: List[int]
    suffix_tokens: List[int]
    kind: str
    rendered_prefix_bytes: int
    checkpoint_path: Optional[str] = None
    delta_base_path: Optional[str] = None
    delta_base_prefix_tokens: Optional[List[int]] = None

    @property
    def prompt(self):
        return self.prefix_tokens + self.suffix_tokens

    @property
    def cached_tokens(self):
        return len(self.prefix_tokens)


@dataclass
class _AsyncCheckpointSaveJob:
    label: str
    save: Callable[[], bool]


def _prompt_checkpoint_policy_int(args, name, default):
    value = getattr(args, name, default)
    if value is None:
        value = default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


def _prompt_checkpoint_max_age_seconds(args):
    return _prompt_checkpoint_policy_int(
        args,
        "checkpoint_max_age_seconds",
        DEFAULT_PROMPT_CHECKPOINT_MAX_AGE_SECONDS,
    )


def _prompt_checkpoint_save_exact_enabled(args):
    value = getattr(args, "checkpoint_save_exact", "enabled")
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off", "disabled"}


def _prompt_checkpoint_save_exact_for_prompt(args, token_count):
    if not _prompt_checkpoint_save_exact_enabled(args):
        return False
    cold_max_tokens = _prompt_checkpoint_policy_int(
        args,
        "checkpoint_cold_max_tokens",
        DEFAULT_PROMPT_CHECKPOINT_COLD_MAX_TOKENS,
    )
    try:
        token_count = max(0, int(token_count))
    except (TypeError, ValueError):
        return False
    return cold_max_tokens <= 0 or token_count <= cold_max_tokens


def _generation_shutdown_timeout_seconds(args):
    if args is None:
        return DEFAULT_GENERATION_SHUTDOWN_TIMEOUT_SECONDS
    try:
        return max(float(getattr(args, "generation_shutdown_timeout", 0.0)), 0.0)
    except (TypeError, ValueError):
        return DEFAULT_GENERATION_SHUTDOWN_TIMEOUT_SECONDS


def _prompt_checkpoint_post_response_save_mode(args):
    value = getattr(args, "checkpoint_post_response_save_mode", "async")
    value = str(value).strip().lower()
    if value not in {"async", "sync"}:
        return "async"
    return value


def _prompt_checkpoint_async_shutdown_timeout_seconds(args):
    if args is None:
        return DEFAULT_PROMPT_CHECKPOINT_ASYNC_SHUTDOWN_TIMEOUT_SECONDS
    try:
        return max(
            float(
                getattr(
                    args,
                    "checkpoint_async_save_shutdown_timeout",
                    DEFAULT_PROMPT_CHECKPOINT_ASYNC_SHUTDOWN_TIMEOUT_SECONDS,
                )
            ),
            0.0,
        )
    except (TypeError, ValueError):
        return DEFAULT_PROMPT_CHECKPOINT_ASYNC_SHUTDOWN_TIMEOUT_SECONDS


def _prompt_checkpoint_boundary_store_length(
    token_count,
    *,
    min_tokens=DEFAULT_PROMPT_CHECKPOINT_MIN_TOKENS,
    trim_tokens=DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_TRIM_TOKENS,
    align_tokens=DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_ALIGN_TOKENS,
):
    try:
        token_count = max(0, int(token_count))
        min_tokens = max(0, int(min_tokens))
        trim_tokens = max(0, int(trim_tokens))
        align_tokens = max(0, int(align_tokens))
    except (TypeError, ValueError):
        return 0
    if token_count == 0:
        return 0

    if token_count > min_tokens + trim_tokens:
        stable_length = token_count - trim_tokens
        if align_tokens > 0:
            stable_length -= stable_length % align_tokens
        if stable_length >= min_tokens:
            return stable_length
    return token_count


def _prompt_checkpoint_cold_prefix_length(args, token_count):
    min_tokens = _prompt_checkpoint_policy_int(
        args, "checkpoint_min_tokens", DEFAULT_PROMPT_CHECKPOINT_MIN_TOKENS
    )
    cold_max_tokens = _prompt_checkpoint_policy_int(
        args,
        "checkpoint_cold_max_tokens",
        DEFAULT_PROMPT_CHECKPOINT_COLD_MAX_TOKENS,
    )
    trim_tokens = _prompt_checkpoint_policy_int(
        args,
        "checkpoint_boundary_trim_tokens",
        DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_TRIM_TOKENS,
    )
    align_tokens = _prompt_checkpoint_policy_int(
        args,
        "checkpoint_boundary_align_tokens",
        DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_ALIGN_TOKENS,
    )

    try:
        token_count = max(0, int(token_count))
    except (TypeError, ValueError):
        return 0
    if token_count < min_tokens:
        return 0
    if cold_max_tokens > 0 and token_count > cold_max_tokens:
        return 0
    return _prompt_checkpoint_boundary_store_length(
        token_count,
        min_tokens=min_tokens,
        trim_tokens=trim_tokens,
        align_tokens=align_tokens,
    )


def _prompt_checkpoint_continued_step(args):
    interval_tokens = _prompt_checkpoint_policy_int(
        args,
        "checkpoint_continued_interval_tokens",
        DEFAULT_PROMPT_CHECKPOINT_CONTINUED_INTERVAL_TOKENS,
    )
    align_tokens = _prompt_checkpoint_policy_int(
        args,
        "checkpoint_boundary_align_tokens",
        DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_ALIGN_TOKENS,
    )
    if interval_tokens <= 0:
        return 0
    if align_tokens <= 0:
        return interval_tokens
    return ((interval_tokens + align_tokens - 1) // align_tokens) * align_tokens


def _prompt_checkpoint_continued_frontier_args(args):
    continued_step = _prompt_checkpoint_continued_step(args)
    if continued_step <= 0:
        return 0, 0
    return continued_step, continued_step


def _prompt_checkpoint_continued_store_length(args, token_count):
    continued_step = _prompt_checkpoint_continued_step(args)
    if continued_step <= 0:
        return 0
    min_tokens = _prompt_checkpoint_policy_int(
        args, "checkpoint_min_tokens", DEFAULT_PROMPT_CHECKPOINT_MIN_TOKENS
    )
    trim_tokens = _prompt_checkpoint_policy_int(
        args,
        "checkpoint_boundary_trim_tokens",
        DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_TRIM_TOKENS,
    )
    try:
        token_count = max(0, int(token_count))
    except (TypeError, ValueError):
        return 0
    if token_count < min_tokens:
        return 0
    stable_length = max(0, token_count - trim_tokens)
    store_length = (stable_length // continued_step) * continued_step
    return store_length if store_length >= min_tokens else 0


def _prompt_checkpoint_store_prefix_lengths(
    args,
    prompt,
    segments,
    segment_types,
    initial_cached_tokens=0,
):
    try:
        total_tokens = len(prompt)
    except TypeError:
        return []
    if total_tokens <= 1:
        return []

    try:
        initial_cached_tokens = max(0, int(initial_cached_tokens))
    except (TypeError, ValueError):
        initial_cached_tokens = 0

    lengths = []
    seen = set()

    def add(length):
        if (
            length is not None
            and initial_cached_tokens < length < total_tokens
            and length not in seen
        ):
            seen.add(length)
            lengths.append(length)

    segment_end = 0
    for segment, segment_type in zip(segments or (), segment_types or ()):
        segment_end += len(segment)
        if segment_type == "system":
            add(segment_end)

    add(_prompt_checkpoint_cold_prefix_length(args, total_tokens))
    lengths.sort()
    return lengths


class TokenLoopGuard:
    def __init__(self, ngram_size: int = 64, repeats: int = 3, min_tokens: int = 256):
        self.ngram_size = max(0, int(ngram_size))
        self.repeats = max(0, int(repeats))
        self.min_tokens = max(0, int(min_tokens))
        self.tokens = []

    @property
    def enabled(self):
        return self.ngram_size > 0 and self.repeats > 1

    def append(self, token: int) -> bool:
        self.tokens.append(int(token))
        return self.has_loop()

    def has_loop(self) -> bool:
        if not self.enabled or len(self.tokens) < self.min_tokens:
            return False
        ngram_sizes = [8, 16, 32, 64, self.ngram_size]
        seen = set()
        for ngram_size in ngram_sizes:
            if ngram_size in seen or ngram_size <= 0 or ngram_size > self.ngram_size:
                continue
            seen.add(ngram_size)
            window_size = ngram_size * self.repeats
            if len(self.tokens) < window_size:
                continue
            tail = self.tokens[-window_size:]
            first = tail[:ngram_size]
            if all(
                tail[i : i + ngram_size] == first
                for i in range(ngram_size, window_size, ngram_size)
            ):
                return True
        return False


@dataclass
class Response:
    text: str
    token: int
    state: str
    match: Tuple[int]
    logprob: float
    finish_reason: Optional[str]
    top_tokens: Tuple[Dict[str, Any]]


def _process_control_tokens(ctx, token_stream):
    buffer_size = max(len(s) for s in ctx.sequences)
    buffered_stream = deque()

    for tok in token_stream:
        buffered_stream.append(tok)
        if tok.match is not None:
            popped = [buffered_stream.pop() for _ in tok.match]
            for t in reversed(popped):
                buffered_stream.append(replace(t, text=""))
        if len(buffered_stream) >= buffer_size:
            yield buffered_stream.popleft()
    while len(buffered_stream) > 0:
        yield buffered_stream.popleft()


class TimeBudget:
    def __init__(self, budget=0.5, iterations=25, sync_frequency=10):
        self._is_distributed = mx.distributed.init().size() > 1
        self._budget = budget
        self._iterations = iterations
        self._sync_frequency = sync_frequency
        self._start = None
        self._current_iterations = None
        self._loops = 0
        self._time_spent = 0

    def __iter__(self):
        self._start = time.time()
        self._current_iterations = 0
        return self

    def __next__(self):
        if not self._is_distributed:
            if time.time() - self._start > self._budget:
                raise StopIteration()
            return None

        self._current_iterations += 1
        if self._current_iterations <= self._iterations:
            return None

        self._loops += 1
        self._time_spent += time.time() - self._start
        if self._loops % self._sync_frequency == 0:
            loop_time = mx.distributed.all_sum(self._time_spent).item()
            avg_loop_time = loop_time / (
                mx.distributed.init().size() * self._sync_frequency
            )
            factor = self._budget / avg_loop_time
            self._iterations = max(round(self._iterations * factor), 1)
            self._loops = 0
            self._time_spent = 0
        raise StopIteration()


class ModelProvider:
    def __init__(self, cli_args: argparse.Namespace):
        """Load models on demand and persist them across the whole process."""
        self.cli_args = cli_args
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.is_batchable = False

        group = mx.distributed.init()
        self.pipeline_group = group if group.size() > 1 and cli_args.pipeline else None
        self.tensor_group = (
            group if group.size() > 1 and not cli_args.pipeline else None
        )
        self.is_distributed = group.size() > 1

        # Maps model and adapter paths the actual paths to be used. Used to
        # map 'default_model' to the provided model by cli argument but could
        # be used for more in the future.
        self._model_map = {}
        self._adapter_map = {}
        self._draft_model_map = {}
        self._model_map["default_model"] = self.cli_args.model
        self._adapter_map["default_model"] = self.cli_args.adapter_path
        self._draft_model_map["default_model"] = self.cli_args.draft_model

        # Build the tokenizer config for later use in load
        self._tokenizer_config = {"trust_remote_code": cli_args.trust_remote_code}
        if cli_args.chat_template:
            self._tokenizer_config["chat_template"] = cli_args.chat_template

    def _load(self, model_path, adapter_path=None, draft_model_path=None):
        if self.is_distributed and (
            adapter_path is not None or draft_model_path is not None
        ):
            raise ValueError(
                "Loading with adapters or draft models not supported in distributed mode"
            )

        # Remove the old model if it exists.
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None

        # Load the model and tokenizer
        if self.is_distributed:
            model, tokenizer = sharded_load(
                model_path,
                pipeline_group=self.pipeline_group,
                tensor_group=self.tensor_group,
                tokenizer_config=self._tokenizer_config,
                trust_remote_code=self.cli_args.trust_remote_code,
            )
        else:
            model, tokenizer = load(
                model_path,
                adapter_path=adapter_path,
                tokenizer_config=self._tokenizer_config,
                trust_remote_code=self.cli_args.trust_remote_code,
            )

        # Use the default chat template if needed
        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template

        # Load the draft model for speculative decoding
        draft_model = None
        if draft_model_path is not None:
            draft_model, draft_tokenizer = load(draft_model_path)
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                logging.warning(
                    "Draft model tokenizer does not match model tokenizer. "
                    "Speculative decoding may not work as expected."
                )

        # Compute batchability
        is_batchable = draft_model is None
        is_batchable = is_batchable and all(
            hasattr(c, "merge") for c in make_prompt_cache(model)
        )
        if (
            self.cli_args.kv_bits is not None
            and self.cli_args.kv_bits != 8
            and model_has_glm_mla_kv_cache(model)
        ):
            raise ValueError("GLM MLA KV quantization supports only --kv-bits 8")

        # Update the member variables
        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.is_batchable = is_batchable

    def load_default(self):
        if self._model_map["default_model"] is not None:
            self.load("default_model", None, "default_model")

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        model_path = self._model_map.get(model_path, model_path)
        adapter_path = self._adapter_map.get(model_path, adapter_path)
        draft_model_path = self._draft_model_map.get(draft_model_path, draft_model_path)

        model_key = (model_path, adapter_path, draft_model_path)
        if self.model_key != model_key:
            self._load(*model_key)

        return self.model, self.tokenizer


def _make_sampler(args, tokenizer):
    return make_sampler(
        args.sampling.temperature,
        top_p=args.sampling.top_p,
        top_k=args.sampling.top_k,
        min_p=args.sampling.min_p,
        xtc_probability=args.sampling.xtc_probability,
        xtc_threshold=args.sampling.xtc_threshold,
        xtc_special_tokens=[
            tokenizer.eos_token_id,
            tokenizer.encode("\n"),
        ],
    )


def _make_logits_processors(args):
    return make_logits_processors(
        args.logits.logit_bias,
        args.logits.repetition_penalty,
        args.logits.repetition_context_size,
        args.logits.presence_penalty,
        args.logits.presence_context_size,
        args.logits.frequency_penalty,
        args.logits.frequency_context_size,
    )


def _format_top_logprobs(logprobs, top_n, tokenizer) -> Tuple[Dict[str, Any]]:
    """Returns info dicts for the top `top_n` tokens from `logprobs`"""
    if top_n <= 0:
        return ()
    sorted_indices = mx.argpartition(-logprobs, kth=top_n - 1)
    top_indices = sorted_indices[:top_n].tolist()
    top_probs = logprobs[top_indices].tolist()
    txts = tokenizer.convert_ids_to_tokens(top_indices)
    return tuple(
        {"id": i, "token": s, "logprob": g}
        for i, s, g in zip(top_indices, txts, top_probs)
    )


class ResponseGenerator:
    def __init__(self, model_provider: ModelProvider, prompt_cache: LRUPromptCache):
        self.model_provider = model_provider
        self.prompt_cache = prompt_cache
        self.requests = Queue()
        self._state_machine_cache = {}

        self._time_budget = TimeBudget()
        self._is_distributed = mx.distributed.init().size() > 1
        self._rank = mx.distributed.init().rank()
        self._stop = False
        self._shutdown_complete = False
        self._checkpoint_save_queue = Queue()
        self._checkpoint_save_stop = object()
        self._checkpoint_save_active = False
        self._checkpoint_save_thread = Thread(
            target=self._checkpoint_save_worker,
            daemon=True,
        )
        self._checkpoint_save_thread.start()
        self._generation_thread = Thread(target=self._generate, daemon=True)
        self._generation_thread.start()

    def stop_and_join(self):
        cli_args = getattr(getattr(self, "model_provider", None), "cli_args", None)
        generation_timeout = _generation_shutdown_timeout_seconds(cli_args)
        logging.info(
            "Shutdown requested: asking generation worker to stop. "
            "Waiting up to %.3fs for active generation to finish...",
            generation_timeout,
        )
        self._stop = True
        self._generation_thread.join(timeout=generation_timeout)
        is_alive = getattr(self._generation_thread, "is_alive", lambda: False)
        if is_alive():
            logging.warning(
                "Generation worker still active after %.3fs; continuing shutdown.",
                generation_timeout,
            )
        self._stop_checkpoint_save_worker(cli_args)
        self.shutdown()

    def join(self):
        self._generation_thread.join()

    def _checkpoint_save_worker(self):
        while True:
            job = self._checkpoint_save_queue.get()
            try:
                if job is self._checkpoint_save_stop:
                    return
                self._checkpoint_save_active = True
                _prompt_checkpoint_debug(
                    f"async checkpoint save start label={job.label}"
                )
                saved = job.save()
                _prompt_checkpoint_debug(
                    f"async checkpoint save complete label={job.label} "
                    f"saved={int(bool(saved))}"
                )
            except Exception as exc:
                _prompt_checkpoint_debug(
                    "async checkpoint save failure "
                    f"label={getattr(job, 'label', 'unknown')} "
                    f"error={type(exc).__name__}"
                )
            finally:
                if job is not self._checkpoint_save_stop:
                    self._checkpoint_save_active = False
                self._checkpoint_save_queue.task_done()

    def _enqueue_checkpoint_save(self, label: str, save: Callable[[], bool]):
        self._checkpoint_save_queue.put(_AsyncCheckpointSaveJob(label=label, save=save))
        _prompt_checkpoint_debug(
            f"async checkpoint save queued label={label} "
            f"pending={self._checkpoint_save_queue.qsize()}"
        )

    def _stop_checkpoint_save_worker(self, cli_args):
        checkpoint_queue = getattr(self, "_checkpoint_save_queue", None)
        checkpoint_thread = getattr(self, "_checkpoint_save_thread", None)
        if checkpoint_queue is None or checkpoint_thread is None:
            return

        checkpoint_queue.put(self._checkpoint_save_stop)
        shutdown_timeout = _prompt_checkpoint_async_shutdown_timeout_seconds(cli_args)
        pending_saves = max(checkpoint_queue.qsize() - 1, 0)
        if shutdown_timeout > 0:
            logging.info(
                "Waiting up to %.3fs for async prompt checkpoint saves; "
                "pending=%s.",
                shutdown_timeout,
                pending_saves,
            )
        checkpoint_thread.join(timeout=shutdown_timeout)
        active = bool(getattr(self, "_checkpoint_save_active", False))
        if checkpoint_thread.is_alive() and (
            active or pending_saves > 0 or shutdown_timeout > 0
        ):
            logging.warning(
                "Async prompt checkpoint save worker still active after %.3fs; "
                "continuing shutdown. pending=%s",
                shutdown_timeout,
                pending_saves,
            )

    def flush_shutdown_prompt_checkpoints(self):
        limit = _prompt_checkpoint_policy_int(
            self.cli_args,
            "checkpoint_shutdown_save_limit",
            DEFAULT_PROMPT_CHECKPOINT_SHUTDOWN_SAVE_LIMIT,
        )
        stats = {
            "enabled": limit > 0,
            "limit": limit,
            "max_tokens": _prompt_checkpoint_policy_int(
                self.cli_args,
                "checkpoint_shutdown_max_tokens",
                DEFAULT_PROMPT_CHECKPOINT_SHUTDOWN_MAX_TOKENS,
            ),
            "candidates": 0,
            "attempted": 0,
            "saved": 0,
            "skipped": 0,
            "skipped_too_large": 0,
        }
        if limit <= 0:
            _prompt_checkpoint_debug("shutdown flush skipped disabled")
            return stats
        if self.model_provider.draft_model is not None:
            _prompt_checkpoint_debug("shutdown flush skipped draft model active")
            return stats
        if self.model_provider.model is None or self.model_provider.tokenizer is None:
            _prompt_checkpoint_debug("shutdown flush skipped model not loaded")
            return stats
        if not hasattr(self.prompt_cache, "snapshot"):
            _prompt_checkpoint_debug("shutdown flush skipped cache snapshot unavailable")
            return stats

        current_model_key = self.model_provider.model_key
        candidates = [
            entry
            for entry in self.prompt_cache.snapshot(newest_first=True)
            if entry.get("model") == current_model_key
        ]
        candidates.sort(
            key=lambda entry: (
                len(entry.get("tokens") or ()),
                entry.get("last_hit_at") or entry.get("created_at") or 0,
            ),
            reverse=True,
        )
        stats["candidates"] = len(candidates)

        seen_checkpoint_files = set()
        for entry in candidates:
            if stats["saved"] >= limit:
                break
            tokens = entry.get("tokens") or []
            store_length = _prompt_checkpoint_continued_store_length(
                self.cli_args,
                len(tokens),
            )
            if store_length <= 0:
                stats["skipped"] += 1
                continue
            if stats["max_tokens"] > 0 and store_length > stats["max_tokens"]:
                stats["skipped"] += 1
                stats["skipped_too_large"] += 1
                _prompt_checkpoint_debug(
                    "shutdown flush skipped too large "
                    f"store_length={store_length} "
                    f"max_tokens={stats['max_tokens']}"
                )
                continue
            checkpoint_name = os.path.basename(
                prompt_checkpoint_file(tokens[:store_length])
            )
            if checkpoint_name in seen_checkpoint_files:
                stats["skipped"] += 1
                continue
            seen_checkpoint_files.add(checkpoint_name)
            stats["attempted"] += 1
            if self._save_continued_prompt_checkpoint(
                self.model_provider.tokenizer,
                entry["prompt_cache"],
                tokens,
                prompt_token_count=0,
                rendered_continuation=None,
            ):
                stats["saved"] += 1

        _prompt_checkpoint_debug(
            "shutdown flush complete "
            f"candidates={stats['candidates']} "
            f"attempted={stats['attempted']} "
            f"saved={stats['saved']} "
            f"skipped={stats['skipped']} "
            f"skipped_too_large={stats['skipped_too_large']} "
            f"max_tokens={stats['max_tokens']} "
            f"limit={limit}"
        )
        return stats

    def prune_shutdown_prompt_checkpoints(self):
        try:
            stats = prune_prompt_checkpoints(
                max_age_seconds=_prompt_checkpoint_max_age_seconds(self.cli_args),
            )
        except Exception as exc:
            _prompt_checkpoint_debug(
                "shutdown prune failure swallowed "
                f"error={type(exc).__name__}"
            )
            return None
        removed = stats.get("removed", [])
        if removed:
            _prompt_checkpoint_debug(
                "shutdown prune removed "
                f"count={len(removed)} "
                f"total_files={stats.get('total_files')} "
                f"total_bytes={stats.get('total_bytes')}"
            )
        return stats

    def shutdown(self):
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        logging.info("Shutdown sequence started: flushing prompt checkpoints.")
        self.flush_shutdown_prompt_checkpoints()
        self.prune_shutdown_prompt_checkpoints()
        logging.info("Shutdown sequence complete.")

    def _log_cache_stats(self):
        n_sequences = len(self.prompt_cache)
        n_bytes = self.prompt_cache.nbytes
        logging.info(f"Prompt Cache: {n_sequences} sequences, {n_bytes / 1e9:.2f} GB")
        for cache_type, stats in self.prompt_cache.stats_by_type().items():
            n_sequences = stats["n_sequences"]
            n_bytes = stats["n_bytes"]
            logging.info(
                f"- {cache_type}: {n_sequences} sequences, {n_bytes / 1e9:.2f} GB"
            )

    def _next_request(self, timeout=None):
        request = None
        if not self._is_distributed or self._rank == 0:
            try:
                if timeout is not None:
                    request = self.requests.get(timeout=timeout)
                else:
                    request = self.requests.get_nowait()
            except QueueEmpty:
                pass
        return self._share_request(request)

    def _share_object(self, obj):
        if not self._is_distributed:
            return obj

        if self._rank == 0:
            if obj is None:
                mx.eval(mx.distributed.all_sum(0))
                return None
            data = mx.array(pickle.dumps(obj))
            mx.eval(mx.distributed.all_sum(data.size))
            mx.eval(mx.distributed.all_sum(data))
            return obj
        else:
            size = mx.distributed.all_sum(0).item()
            if size == 0:
                return None
            data = mx.zeros(size, dtype=mx.uint8)
            data = mx.distributed.all_sum(data)
            return pickle.loads(data)

    def _share_request(self, request):
        if not self._is_distributed:
            return request

        shareable = request[1:] if request is not None else None
        shareable = self._share_object(shareable)
        if shareable is None:
            return None

        rq = request[0] if request is not None else Queue()
        return rq, *shareable

    def _render_prompt_text(self, tokenizer, request, args):
        if not hasattr(request, "request_type"):
            return None
        if request.request_type != "chat":
            return request.prompt

        messages = request.messages
        tools = request.tools
        role_mapping = request.role_mapping
        if not tokenizer.has_chat_template:
            return convert_chat(messages, role_mapping)

        process_message_content(messages)
        if tools and not tokenizer.has_tool_calling:
            logging.warning(
                "Received tools but model does not support tool calling. "
                "If you think this is an error, file an issue here: "
                "https://github.com/ml-explore/mlx-lm/issues"
            )
        chat_template_args = self.model_provider.cli_args.chat_template_args
        if args.chat_template_kwargs:
            chat_template_args = chat_template_args.copy()
            chat_template_args.update(args.chat_template_kwargs)
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            tools=tools,
            **chat_template_args,
        )

    @staticmethod
    def _encode_rendered_suffix(tokenizer, rendered_suffix):
        try:
            return tokenizer.encode(rendered_suffix, add_special_tokens=False)
        except TypeError:
            return tokenizer.encode(rendered_suffix)

    @staticmethod
    def _decode_checkpoint_prefix(tokenizer, prefix_tokens):
        decode_kwargs = [
            {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
            {"skip_special_tokens": False},
            {},
        ]
        for decode_kwarg in decode_kwargs:
            try:
                return tokenizer.decode(prefix_tokens, **decode_kwarg)
            except TypeError:
                continue
        return tokenizer.decode(prefix_tokens)

    @classmethod
    def _decode_checkpoint_tokens_bytes(cls, tokenizer, tokens):
        return rendered_prompt_bytes(cls._decode_checkpoint_prefix(tokenizer, tokens))

    @staticmethod
    def _initial_state_from_prompt(tokenizer, prompt):
        initial_state = "normal"
        if tokenizer.has_thinking:
            think_start = tokenizer.rfind_think_start(prompt)
            think_end = tokenizer.rfind_think_end(prompt)
            if think_start > think_end:
                initial_state = "reasoning"
        return initial_state

    @staticmethod
    def _checkpoint_metadata_int(metadata, key, default=0):
        try:
            return int(metadata.get(key, default))
        except (TypeError, ValueError):
            return default

    def _load_delta_prompt_checkpoint(self, checkpoint_path):
        expected_settings = expected_glm_mla_kv_settings_metadata(
            self.model_provider.model,
            kv_bits=self.cli_args.kv_bits,
            kv_group_size=self.cli_args.kv_group_size,
            quantized_kv_start=self.cli_args.quantized_kv_start,
        )
        delta_cache, target_tokens, metadata = (
            load_prompt_checkpoint_with_metadata_prefix(
                checkpoint_path,
                checkpoint_namespace=DEFAULT_PROMPT_CHECKPOINT_NAMESPACE,
                model=self.model_provider.model,
                expected_glm_mla_kv_settings=expected_settings,
                return_metadata=True,
            )
        )
        if metadata.get("checkpoint_label") != "delta":
            raise PromptCacheCheckpointError("delta checkpoint label mismatch")

        base_filename = metadata.get(PROMPT_CHECKPOINT_DELTA_BASE_FILENAME_METADATA_KEY)
        if not base_filename or base_filename != os.path.basename(base_filename):
            raise PromptCacheCheckpointError("delta checkpoint base filename mismatch")
        base_prefix_length = self._checkpoint_metadata_int(
            metadata,
            PROMPT_CHECKPOINT_DELTA_BASE_PREFIX_LENGTH_METADATA_KEY,
            0,
        )
        cache_start_tokens = self._checkpoint_metadata_int(
            metadata,
            PROMPT_CHECKPOINT_DELTA_CACHE_START_TOKENS_METADATA_KEY,
            0,
        )
        cache_tokens = self._checkpoint_metadata_int(
            metadata,
            PROMPT_CHECKPOINT_DELTA_CACHE_TOKENS_METADATA_KEY,
            0,
        )
        if (
            base_prefix_length <= 0
            or cache_start_tokens != base_prefix_length
            or cache_tokens <= base_prefix_length
            or cache_tokens > len(target_tokens)
        ):
            raise PromptCacheCheckpointError("delta checkpoint token bounds mismatch")
        base_tokens = target_tokens[:base_prefix_length]
        if (
            metadata.get(PROMPT_CHECKPOINT_DELTA_BASE_PREFIX_HASH_METADATA_KEY)
            != prompt_prefix_hash(base_tokens)
        ):
            raise PromptCacheCheckpointError("delta checkpoint base hash mismatch")

        delta_tokens = cache_tokens - base_prefix_length
        if prompt_cache_token_length(delta_cache) != delta_tokens:
            raise PromptCacheCheckpointError("delta checkpoint cache length mismatch")

        base_path = os.path.join(os.path.dirname(checkpoint_path), base_filename)
        expected_quantization = expected_glm_mla_kv_quantization_metadata(
            self.model_provider.model,
            cache_token_length=base_prefix_length,
            kv_bits=self.cli_args.kv_bits,
            kv_group_size=self.cli_args.kv_group_size,
            quantized_kv_start=self.cli_args.quantized_kv_start,
        )
        expected_cache_layout = expected_prompt_cache_layout_signature(
            self.model_provider.model,
            kv_bits=self.cli_args.kv_bits,
            kv_group_size=self.cli_args.kv_group_size,
            quantized_kv_start=self.cli_args.quantized_kv_start,
            cache_token_length=base_prefix_length,
        )
        base_cache, loaded_base_tokens, _ = (
            load_prompt_checkpoint_with_metadata_prefix(
                base_path,
                checkpoint_namespace=DEFAULT_PROMPT_CHECKPOINT_NAMESPACE,
                model=self.model_provider.model,
                expected_glm_mla_kv_quantization=expected_quantization,
                expected_glm_mla_kv_settings=expected_settings,
                expected_cache_layout=expected_cache_layout,
                return_metadata=True,
            )
        )
        if loaded_base_tokens != base_tokens:
            raise PromptCacheCheckpointError("delta checkpoint base tokens mismatch")

        return (
            concat_prompt_caches(base_cache, delta_cache),
            target_tokens[:cache_tokens],
            target_tokens,
            metadata,
            base_path,
            base_tokens,
        )

    def _load_rendered_prompt_checkpoint(self, tokenizer, rendered_prompt):
        if rendered_prompt is None or self.model_provider.draft_model is not None:
            return None
        rendered = rendered_prompt_bytes(rendered_prompt)
        if not rendered:
            return None

        def expected_cache_layout_for_rendered_candidate(prefix_length, kind):
            if kind == "delta":
                return None
            cache_token_length = (
                max(prefix_length - 1, 0)
                if kind == "exact"
                else prefix_length
            )
            return expected_prompt_cache_layout_signature(
                self.model_provider.model,
                kv_bits=self.cli_args.kv_bits,
                kv_group_size=self.cli_args.kv_group_size,
                quantized_kv_start=self.cli_args.quantized_kv_start,
                cache_token_length=cache_token_length,
            )

        candidates, lookup_stats = find_prompt_checkpoint_rendered_prefix(
            rendered,
            allowed_kinds=("prefix", "frontier", "continued", "delta", "exact"),
            expected_cache_layout_by_candidate=(
                expected_cache_layout_for_rendered_candidate
            ),
            return_stats=True,
        )
        _prompt_checkpoint_debug(
            "rendered lookup result "
            f"rendered_bytes={len(rendered)} "
            f"candidates={len(candidates)} "
            f"lcp_index_entries={lookup_stats.get('lcp_manager_entries', 0)} "
            f"lcp_rendered_lengths={lookup_stats.get('lcp_manager_rendered_lengths', 0)} "
            f"cache_layout_rejections={lookup_stats.get('cache_layout_rejections', 0)} "
            f"manifest_missing_entries_removed={lookup_stats.get('manifest_missing_entries_removed', 0)}"
        )
        for rendered_prefix_length, token_prefix_length, checkpoint_path, kind in (
            candidates
        ):
            if kind not in ("prefix", "frontier", "continued", "delta", "exact"):
                continue

            try:
                if kind == "delta":
                    (
                        prompt_cache,
                        prefix_tokens,
                        stored_prefix_tokens,
                        metadata,
                        delta_base_path,
                        delta_base_prefix_tokens,
                    ) = self._load_delta_prompt_checkpoint(checkpoint_path)
                else:
                    checkpoint_cache_token_length = (
                        max(token_prefix_length - 1, 0)
                        if kind == "exact"
                        else token_prefix_length
                    )
                    expected_quantization = expected_glm_mla_kv_quantization_metadata(
                        self.model_provider.model,
                        cache_token_length=checkpoint_cache_token_length,
                        kv_bits=self.cli_args.kv_bits,
                        kv_group_size=self.cli_args.kv_group_size,
                        quantized_kv_start=self.cli_args.quantized_kv_start,
                    )
                    expected_settings = expected_glm_mla_kv_settings_metadata(
                        self.model_provider.model,
                        kv_bits=self.cli_args.kv_bits,
                        kv_group_size=self.cli_args.kv_group_size,
                        quantized_kv_start=self.cli_args.quantized_kv_start,
                    )
                    expected_cache_layout = expected_prompt_cache_layout_signature(
                        self.model_provider.model,
                        kv_bits=self.cli_args.kv_bits,
                        kv_group_size=self.cli_args.kv_group_size,
                        quantized_kv_start=self.cli_args.quantized_kv_start,
                        cache_token_length=checkpoint_cache_token_length,
                    )
                    prompt_cache, prefix_tokens, metadata = (
                        load_prompt_checkpoint_with_metadata_prefix(
                            checkpoint_path,
                            checkpoint_namespace=DEFAULT_PROMPT_CHECKPOINT_NAMESPACE,
                            model=self.model_provider.model,
                            expected_glm_mla_kv_quantization=expected_quantization,
                            expected_glm_mla_kv_settings=expected_settings,
                            expected_cache_layout=expected_cache_layout,
                            return_metadata=True,
                        )
                    )
                    stored_prefix_tokens = prefix_tokens
                    delta_base_path = None
                    delta_base_prefix_tokens = None
            except PromptCacheCheckpointError:
                _prompt_checkpoint_debug(
                    "rendered candidate rejected load "
                    f"file={os.path.basename(checkpoint_path)} "
                    f"kind={kind}"
                )
                continue
            checkpoint_label = metadata.get("checkpoint_label", kind)
            if checkpoint_label not in (
                "prefix",
                "frontier",
                "continued",
                "delta",
                "exact",
            ):
                continue
            if kind == "exact" and checkpoint_label != "exact":
                _prompt_checkpoint_debug(
                    "rendered candidate rejected label mismatch "
                    f"file={os.path.basename(checkpoint_path)} "
                    f"manifest_kind={kind} "
                    f"checkpoint_label={checkpoint_label}"
                )
                continue

            if checkpoint_label == "exact":
                if len(stored_prefix_tokens) <= 1:
                    _prompt_checkpoint_debug(
                        "rendered candidate rejected exact too short "
                        f"file={os.path.basename(checkpoint_path)} "
                        f"prefix_tokens={len(stored_prefix_tokens)}"
                    )
                    continue
                prefix_tokens = stored_prefix_tokens[:-1]
                try:
                    decoded_prefix = self._decode_checkpoint_tokens_bytes(
                        tokenizer,
                        prefix_tokens,
                    )
                except Exception:
                    decoded_prefix = None
                if not decoded_prefix or not rendered.startswith(decoded_prefix):
                    _prompt_checkpoint_debug(
                        "rendered candidate rejected exact prefix decode mismatch "
                        f"file={os.path.basename(checkpoint_path)} "
                        f"stored_prefix_tokens={len(stored_prefix_tokens)} "
                        f"cached_tokens={len(prefix_tokens)}"
                    )
                    continue
                rendered_prefix_length = len(decoded_prefix)
            else:
                try:
                    decoded_prefix = self._decode_checkpoint_tokens_bytes(
                        tokenizer,
                        prefix_tokens,
                    )
                except Exception:
                    decoded_prefix = None
                if decoded_prefix != rendered[:rendered_prefix_length]:
                    _prompt_checkpoint_debug(
                        "rendered candidate rejected prefix decode mismatch "
                        f"file={os.path.basename(checkpoint_path)} "
                        f"kind={checkpoint_label} "
                        f"prefix_tokens={len(prefix_tokens)} "
                        f"rendered_prefix_bytes={rendered_prefix_length}"
                    )
                    continue

            rendered_suffix_bytes = rendered[rendered_prefix_length:]
            try:
                rendered_suffix = rendered_suffix_bytes.decode("utf-8")
            except UnicodeDecodeError:
                _prompt_checkpoint_debug(
                    "rendered candidate rejected suffix utf8 mismatch "
                    f"file={os.path.basename(checkpoint_path)} "
                    f"kind={checkpoint_label} "
                    f"rendered_prefix_bytes={rendered_prefix_length}"
                )
                continue
            suffix_tokens = self._encode_rendered_suffix(tokenizer, rendered_suffix)
            try:
                decoded_suffix = self._decode_checkpoint_tokens_bytes(
                    tokenizer,
                    suffix_tokens,
                )
            except Exception:
                decoded_suffix = None
            if decoded_suffix != rendered[rendered_prefix_length:]:
                _prompt_checkpoint_debug(
                    "rendered candidate rejected suffix decode mismatch "
                    f"file={os.path.basename(checkpoint_path)} "
                    f"kind={checkpoint_label} "
                    f"suffix_tokens={len(suffix_tokens)} "
                    f"rendered_suffix_bytes={len(rendered_suffix.encode('utf-8'))}"
                )
                continue
            try:
                update_prompt_checkpoint_manifest(
                    checkpoint_path,
                    prefix_length=len(stored_prefix_tokens),
                    kind=checkpoint_label,
                    metadata=metadata,
                    hit=True,
                )
            except Exception:
                pass
            _prompt_checkpoint_debug(
                "rendered hit "
                f"file={os.path.basename(checkpoint_path)} "
                f"kind={checkpoint_label} "
                f"stored_prefix_tokens={len(stored_prefix_tokens)} "
                f"cached_tokens={len(prefix_tokens)} "
                f"suffix_tokens={len(suffix_tokens)} "
                f"rendered_prefix_bytes={rendered_prefix_length}"
            )
            return RenderedPromptCheckpoint(
                prompt_cache=prompt_cache,
                prefix_tokens=prefix_tokens,
                suffix_tokens=suffix_tokens,
                kind=checkpoint_label,
                rendered_prefix_bytes=rendered_prefix_length,
                checkpoint_path=checkpoint_path,
                delta_base_path=delta_base_path,
                delta_base_prefix_tokens=delta_base_prefix_tokens,
            )
        return None

    def _save_continued_prompt_checkpoint(
        self,
        tokenizer,
        prompt_cache,
        cache_key,
        *,
        prompt_token_count,
        rendered_continuation=None,
    ):
        if self.model_provider.draft_model is not None:
            _prompt_checkpoint_debug("save continued skipped draft model active")
            return False
        store_length = _prompt_checkpoint_continued_store_length(
            self.cli_args,
            len(cache_key),
        )
        if store_length <= 0 or store_length <= prompt_token_count:
            _prompt_checkpoint_debug(
                "save continued skipped boundary "
                f"total_tokens={len(cache_key)} "
                f"prompt_tokens={prompt_token_count} "
                f"store_length={store_length}"
            )
            return False

        checkpoint_cache = copy.deepcopy(prompt_cache)
        extra_cache_tokens = prompt_cache_token_length(checkpoint_cache) - len(cache_key)
        if extra_cache_tokens > 0:
            if not can_trim_prompt_cache(checkpoint_cache):
                _prompt_checkpoint_debug(
                    "save continued skipped extra non-trimmable cache "
                    f"extra_tokens={extra_cache_tokens}"
                )
                return False
            if trim_prompt_cache(checkpoint_cache, extra_cache_tokens) != (
                extra_cache_tokens
            ):
                _prompt_checkpoint_debug(
                    "save continued skipped extra trim mismatch "
                    f"extra_tokens={extra_cache_tokens}"
                )
                return False

        trim_tokens = len(cache_key) - store_length
        if trim_tokens > 0:
            if not can_trim_prompt_cache(checkpoint_cache):
                _prompt_checkpoint_debug("save continued skipped non-trimmable cache")
                return False
            if trim_prompt_cache(checkpoint_cache, trim_tokens) != trim_tokens:
                _prompt_checkpoint_debug(
                    "save continued skipped trim mismatch "
                    f"trim_tokens={trim_tokens}"
                )
                return False

        prefix_tokens = cache_key[:store_length]
        metadata = {"checkpoint_label": "continued"}
        rendered_prefix = None
        if rendered_continuation is not None:
            rendered_continuation_bytes = rendered_prompt_bytes(rendered_continuation)
            try:
                rendered_prefix = self._decode_checkpoint_prefix(
                    tokenizer,
                    prefix_tokens,
                )
            except Exception:
                rendered_prefix = None
            rendered_prefix_bytes = rendered_prompt_bytes(rendered_prefix)
            if (
                not rendered_prefix_bytes
                or rendered_continuation_bytes is None
                or not rendered_continuation_bytes.startswith(rendered_prefix_bytes)
            ):
                rendered_prefix = None
        if rendered_prefix is not None:
            metadata.update(prompt_checkpoint_rendered_prefix_metadata(rendered_prefix))
            metadata[PROMPT_CHECKPOINT_PREFIX_TOKENS_METADATA_KEY] = (
                prompt_checkpoint_prefix_tokens_metadata(prefix_tokens)
            )

        checkpoint_path = prompt_checkpoint_file(prefix_tokens)
        try:
            checkpoint_metadata = save_prompt_checkpoint(
                checkpoint_path,
                checkpoint_cache,
                prefix_tokens=prefix_tokens,
                checkpoint_namespace=DEFAULT_PROMPT_CHECKPOINT_NAMESPACE,
                model=self.model_provider.model,
                kv_bits=self.cli_args.kv_bits,
                kv_group_size=self.cli_args.kv_group_size,
                quantized_kv_start=self.cli_args.quantized_kv_start,
                metadata=metadata,
            )
            update_prompt_checkpoint_manifest(
                checkpoint_path,
                prefix_length=store_length,
                kind="continued",
                metadata=checkpoint_metadata,
            )
            prune_prompt_checkpoints(
                protected_files=[os.path.basename(checkpoint_path)],
                max_age_seconds=_prompt_checkpoint_max_age_seconds(self.cli_args),
            )
            _prompt_checkpoint_debug(
                "save continued success "
                f"file={os.path.basename(checkpoint_path)} "
                f"prefix_length={store_length} "
                f"trim_tokens={trim_tokens} "
                f"rendered_metadata={int(rendered_prefix is not None)}"
            )
            return True
        except Exception as exc:
            _prompt_checkpoint_debug(
                "save continued failure swallowed "
                f"file={os.path.basename(checkpoint_path)} "
                f"prefix_length={store_length} "
                f"error={type(exc).__name__}"
            )
            return False

    def _save_delta_prompt_checkpoint(
        self,
        tokenizer,
        prompt_cache,
        cache_key,
        *,
        base_checkpoint,
        rendered_continuation=None,
    ):
        if self.model_provider.draft_model is not None:
            _prompt_checkpoint_debug("save delta skipped draft model active")
            return False
        if base_checkpoint is None:
            _prompt_checkpoint_debug("save delta skipped missing base checkpoint")
            return False

        base_path = base_checkpoint.delta_base_path or base_checkpoint.checkpoint_path
        base_tokens = (
            base_checkpoint.delta_base_prefix_tokens
            or base_checkpoint.prefix_tokens
        )
        if not base_path or not base_tokens or base_checkpoint.kind == "exact":
            _prompt_checkpoint_debug(
                "save delta skipped unsupported base "
                f"kind={getattr(base_checkpoint, 'kind', None)}"
            )
            return False

        base_length = len(base_tokens)
        target_length = len(cache_key)
        if target_length <= base_length:
            _prompt_checkpoint_debug(
                "save delta skipped boundary "
                f"target_length={target_length} "
                f"base_length={base_length}"
            )
            return False
        cache_length = prompt_cache_token_length(prompt_cache)
        if cache_length < target_length:
            _prompt_checkpoint_debug(
                "save delta skipped cache too short "
                f"cache_length={cache_length} "
                f"target_length={target_length}"
            )
            return False

        try:
            delta_cache = slice_prompt_cache(
                prompt_cache,
                base_length,
                target_length,
            )
        except Exception as exc:
            _prompt_checkpoint_debug(
                "save delta skipped slice failure "
                f"target_length={target_length} "
                f"base_length={base_length} "
                f"error={type(exc).__name__}"
            )
            return False

        prefix_tokens = cache_key[:target_length]
        metadata = {
            "checkpoint_label": "delta",
            PROMPT_CHECKPOINT_DELTA_BASE_FILENAME_METADATA_KEY: os.path.basename(
                base_path
            ),
            PROMPT_CHECKPOINT_DELTA_BASE_PREFIX_HASH_METADATA_KEY: prompt_prefix_hash(
                base_tokens
            ),
            PROMPT_CHECKPOINT_DELTA_BASE_PREFIX_LENGTH_METADATA_KEY: str(base_length),
            PROMPT_CHECKPOINT_DELTA_CACHE_START_TOKENS_METADATA_KEY: str(base_length),
            PROMPT_CHECKPOINT_DELTA_CACHE_TOKENS_METADATA_KEY: str(target_length),
            PROMPT_CHECKPOINT_PREFIX_TOKENS_METADATA_KEY: (
                prompt_checkpoint_prefix_tokens_metadata(prefix_tokens)
            ),
        }
        rendered_prefix = None
        if rendered_continuation is not None:
            rendered_continuation_bytes = rendered_prompt_bytes(rendered_continuation)
            try:
                rendered_prefix = self._decode_checkpoint_prefix(
                    tokenizer,
                    prefix_tokens,
                )
            except Exception:
                rendered_prefix = None
            rendered_prefix_bytes = rendered_prompt_bytes(rendered_prefix)
            if (
                not rendered_prefix_bytes
                or rendered_continuation_bytes is None
                or not rendered_continuation_bytes.startswith(rendered_prefix_bytes)
            ):
                rendered_prefix = None
        if rendered_prefix is not None:
            metadata.update(prompt_checkpoint_rendered_prefix_metadata(rendered_prefix))

        checkpoint_path = prompt_checkpoint_file(prefix_tokens)
        try:
            checkpoint_metadata = save_prompt_checkpoint(
                checkpoint_path,
                delta_cache,
                prefix_tokens=prefix_tokens,
                checkpoint_namespace=DEFAULT_PROMPT_CHECKPOINT_NAMESPACE,
                model=self.model_provider.model,
                kv_bits=self.cli_args.kv_bits,
                kv_group_size=self.cli_args.kv_group_size,
                quantized_kv_start=self.cli_args.quantized_kv_start,
                metadata=metadata,
            )
            update_prompt_checkpoint_manifest(
                checkpoint_path,
                prefix_length=target_length,
                kind="delta",
                metadata=checkpoint_metadata,
            )
            prune_prompt_checkpoints(
                protected_files=[
                    os.path.basename(base_path),
                    os.path.basename(checkpoint_path),
                ],
                max_age_seconds=_prompt_checkpoint_max_age_seconds(self.cli_args),
            )
            _prompt_checkpoint_debug(
                "save delta success "
                f"file={os.path.basename(checkpoint_path)} "
                f"target_length={target_length} "
                f"base_length={base_length} "
                f"delta_tokens={target_length - base_length} "
                f"rendered_metadata={int(rendered_prefix is not None)}"
            )
            return True
        except Exception as exc:
            _prompt_checkpoint_debug(
                "save delta failure swallowed "
                f"file={os.path.basename(checkpoint_path)} "
                f"target_length={target_length} "
                f"base_length={base_length} "
                f"error={type(exc).__name__}"
            )
            return False

    def _tokenize(self, tokenizer, request, args, return_rendered=False):
        """Tokenize a request and split the prompt into segments.

        Returns a tuple

          * prompt - Full list of tokens
          * segments - A list of lists of tokens. Up to 3 segments that
            correspond to system prompt, context, thinking tail.
          * segment_types - A string per segment indicating if the segment is a
            system prompt or a user prompt or nothing special.
          * initial state - A string that contains the initial state of the
            state machine (normal or thinking depending on whether we have tail
            or not)
        """
        rendered_prompt = None
        if request.request_type == "chat":
            messages = request.messages
            tools = request.tools
            role_mapping = request.role_mapping

            if tokenizer.has_chat_template:
                process_message_content(messages)
                if tools and not tokenizer.has_tool_calling:
                    logging.warning(
                        "Received tools but model does not support tool calling. "
                        "If you think this is an error, file an issue here: "
                        "https://github.com/ml-explore/mlx-lm/issues"
                    )

                chat_template_args = self.model_provider.cli_args.chat_template_args
                if args.chat_template_kwargs:
                    chat_template_args = chat_template_args.copy()
                    chat_template_args.update(args.chat_template_kwargs)
                template_kwargs = dict(tools=tools, **chat_template_args)
                prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **template_kwargs,
                )
                if return_rendered:
                    rendered_prompt = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        **template_kwargs,
                    )
            else:
                rendered_prompt = convert_chat(messages, role_mapping)
                prompt = tokenizer.encode(rendered_prompt)
                template_kwargs = None
        else:
            prompt = tokenizer.encode(request.prompt)
            if return_rendered:
                return prompt, [prompt], ["assistant"], "normal", request.prompt
            return prompt, [prompt], ["assistant"], "normal"

        # If we are here it means we have a chat request so we need to search
        # for segments for better cache management.

        # Choose the initial state among only reasoning or normal
        initial_state = "normal"
        if tokenizer.has_thinking:
            think_start = tokenizer.rfind_think_start(prompt)
            think_end = tokenizer.rfind_think_end(prompt)
            if think_start > think_end:
                initial_state = "reasoning"

        # It is not a user message so no segmentation needed.
        if messages[-1]["role"] != "user":
            if return_rendered:
                return prompt, [prompt], ["assistant"], initial_state, rendered_prompt
            return prompt, [prompt], ["assistant"], initial_state

        segments = []
        segment_types = []

        # Find where the system prompt ends and add it as a segment.
        num_system = 0
        sys_end = 0
        for m in messages:
            if m["role"] == "system":
                num_system += 1
            else:
                break
        if num_system > 0:
            if tokenizer.has_chat_template:
                sys_tokens = tokenizer.apply_chat_template(
                    messages[:num_system] + [{"role": "user", "content": ""}],
                    add_generation_prompt=False,
                    tokenize=True,
                    **template_kwargs,
                )
            else:
                sys_tokens = tokenizer.encode(
                    convert_chat(
                        messages[:num_system] + [{"role": "user", "content": ""}],
                        role_mapping,
                    )
                )
            sys_end = min(len(sys_tokens), len(prompt))
            for i, (a, b) in enumerate(zip(sys_tokens, prompt)):
                if a != b:
                    sys_end = i
                    break
            if sys_end > 0 and sys_end < len(prompt):
                segments.append(prompt[:sys_end])
                segment_types.append("system")

        # Find a tail segment that contains thinking tokens (small up to 11
        # tokens)
        tail_start = len(prompt)
        if tokenizer.has_thinking:
            think_start = tokenizer.rfind_think_start(prompt, start=tail_start - 11)
            if think_start >= 0:
                tail_start = think_start

        # Finalize the segments and return
        if sys_end < tail_start:
            segments.append(prompt[sys_end:tail_start])
            segment_types.append("user")
        if tail_start < len(prompt):
            segments.append(prompt[tail_start:])
            segment_types.append("assistant")
        if not segments:
            segments = [prompt]
            segment_types = ["assistant"]

        if return_rendered:
            return prompt, segments, segment_types, initial_state, rendered_prompt
        return prompt, segments, segment_types, initial_state

    def _make_state_machine(
        self, model_key, tokenizer, stop_words, initial_state="normal"
    ):
        """Make a new SequenceStateMachine or fetch it if we 've made it before.

        Return also a dictionary that maps the token sequences in the state
        machine to their strings.
        """
        cache_key = (model_key, tuple(stop_words), initial_state)
        rs = self._state_machine_cache.get(cache_key)
        if rs is not None:
            return rs

        # Will hold the state machine transitions and the sequences map to
        # strings.
        transitions = {}
        sequences = {}

        # Add all the stop sequences
        common_stops = []
        for t in tokenizer.eos_token_ids:
            sequences[(t,)] = tokenizer.convert_ids_to_tokens(t)
            common_stops.append(((t,), None))
        for w in stop_words:
            t = tuple(tokenizer.encode(w, add_special_tokens=False))
            sequences[t] = w
            common_stops.append((t, None))

        # From normal to stop
        transitions["normal"] = list(common_stops)

        # Reasoning related transitions
        if tokenizer.has_thinking:
            ts = tokenizer.think_start_tokens
            te = tokenizer.think_end_tokens
            transitions["normal"].append((ts, "reasoning"))
            transitions["reasoning"] = [(te, "normal")]
            transitions["reasoning"].extend(common_stops)
            sequences[ts] = tokenizer.think_start
            sequences[te] = tokenizer.think_end

        # Tool calling relating transitions
        if tokenizer.has_tool_calling:
            ts = tokenizer.tool_call_start_tokens
            te = tokenizer.tool_call_end_tokens
            transitions["normal"].append((ts, "tool"))
            transitions["tool"] = [(te, "normal")] if te else []
            transitions["tool"].extend(common_stops)
            sequences[ts] = tokenizer.tool_call_start
            if te:
                sequences[te] = tokenizer.tool_call_end

        sm = SequenceStateMachine(transitions, initial=initial_state)
        if len(self._state_machine_cache) > 100:
            self._state_machine_cache.clear()
        self._state_machine_cache[cache_key] = (sm, sequences)

        return sm, sequences

    def _is_batchable(self, args):
        if getattr(self.model_provider.cli_args, "disable_batching", False):
            return False
        kv_bits = self.model_provider.cli_args.kv_bits
        kv_batchable = kv_bits is None or (
            kv_bits == 8 and model_has_glm_mla_kv_cache(self.model_provider.model)
        )
        return (
            self.model_provider.is_batchable
            and args.seed is None
            and kv_batchable
        )

    def _generate(self):
        # Local thread stream that we 'll pass to the BatchGenerator to make
        # sure that all generation runs in the same stream as the
        # synchronization messages.
        generation_stream = mx.default_stream(mx.default_device())

        # Load the default model if it is given
        self.model_provider.load_default()

        current_model = None
        current_sampling = None
        current_tokenizer = None
        current_model_key = None
        batch_generator = None
        drain_batch = False
        batch_results = {}

        unprocessed_requests = []

        def get_next_request(timeout=None):
            if unprocessed_requests:
                return unprocessed_requests.pop()
            else:
                return self._next_request(timeout)

        if self._is_distributed:
            seed = mx.distributed.all_sum(mx.random.state[0]).view(mx.uint64).item()
            mx.random.seed(seed)

        while not self._stop:
            request = None
            if not drain_batch:
                timeout = (
                    None
                    if (batch_generator is not None and len(batch_results) > 0)
                    else 0.1
                )
                request = get_next_request(timeout=timeout)

            # We got a request
            if request is not None:
                rqueue, request, args = request

                # Can it be added to the current batch?
                if (
                    batch_generator is not None
                    and current_model == args.model
                    and self._is_batchable(args)
                ):
                    try:
                        prompt, segments, segment_types, initial_state = self._tokenize(
                            current_tokenizer, request, args
                        )
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    sm, sequences = self._make_state_machine(
                        self.model_provider.model_key,
                        tokenizer,
                        args.stop_words,
                        initial_state,
                    )

                    self._log_cache_stats()
                    cache, rest = self.prompt_cache.fetch_nearest_cache(
                        current_model_key, prompt
                    )
                    prompt_cache_count = len(prompt) - len(rest)
                    N = prompt_cache_count
                    while N > 0:
                        if N >= len(segments[0]):
                            N -= len(segments.pop(0))
                            segment_types.pop(0)
                        else:
                            segments[0] = segments[0][N:]
                            break

                    ctx = GenerationContext(
                        has_tool_calling=tokenizer.has_tool_calling,
                        has_thinking=tokenizer.has_thinking,
                        tool_parser=tokenizer.tool_parser,
                        sequences=sequences,
                        prompt=prompt,
                        prompt_cache_count=prompt_cache_count,
                    )
                    rqueue.put(ctx)

                    (uid,) = batch_generator.insert_segments(
                        segments=[segments],
                        max_tokens=[args.max_tokens],
                        caches=[cache],
                        all_tokens=[prompt[:prompt_cache_count]],
                        samplers=[_make_sampler(args, tokenizer)],
                        logits_processors=[_make_logits_processors(args)],
                        state_machines=[sm],
                    )
                    batch_results[uid] = {
                        "ctx": ctx,
                        "rqueue": rqueue,
                        "detokenizer": tokenizer.detokenizer,
                        "segment_types": segment_types[::-1],
                        "top_logprobs": args.top_logprobs,
                    }
                    # just making sure we don't leave a reference around
                    del cache

                    if self.model_provider.cli_args.prompt_cache_bytes is not None:
                        total = self.model_provider.cli_args.prompt_cache_bytes
                        active = batch_generator.prompt_cache_nbytes
                        self.prompt_cache.trim_to(n_bytes=total - active)
                    continue

                # No batch generator. Load the model and if it's not
                # batchable serve sequential, o/w make a batch generaotr and
                # serve batched
                elif batch_generator is None:
                    try:
                        model, tokenizer = self.model_provider.load(
                            args.model.model, args.model.adapter, args.model.draft
                        )
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    if not self._is_batchable(args):
                        self._serve_single((rqueue, request, args))
                        continue

                    current_model = args.model
                    current_tokenizer = tokenizer
                    current_model_key = self.model_provider.model_key
                    batch_results = {}
                    batch_generator = BatchGenerator(
                        model,
                        completion_batch_size=self.cli_args.decode_concurrency,
                        prefill_batch_size=self.cli_args.prompt_concurrency,
                        prefill_step_size=self.cli_args.prefill_step_size,
                        kv_bits=self.cli_args.kv_bits,
                        kv_group_size=self.cli_args.kv_group_size,
                        quantized_kv_start=self.cli_args.quantized_kv_start,
                        stream=generation_stream,
                    )
                    unprocessed_requests.append((rqueue, request, args))
                    continue

                # We have a batch but this request cannot be added to the
                # batch so drain it to process the request.
                else:
                    drain_batch = True
                    unprocessed_requests.append((rqueue, request, args))
                    continue

            # No request so serve from the current batch
            elif batch_generator is not None:
                if len(batch_results) == 0:
                    if drain_batch:
                        current_model = None
                        current_sampling = None
                        current_tokenizer = None
                        current_model_key = None
                        batch_generator.close()
                        batch_generator = None
                        drain_batch = False
                    continue

                uids_to_remove = []
                for _ in self._time_budget:
                    prompt_responses, gen_responses = batch_generator.next()
                    if not prompt_responses and not gen_responses:
                        break

                    # Progress report for prompt processing
                    for r in prompt_responses:
                        result = batch_results[r.uid]
                        result["rqueue"].put(r.progress)
                        if result["ctx"]._should_stop:
                            uids_to_remove.append(r.uid)

                    # Save the caches at end of segments
                    eos_ids = [
                        r.uid
                        for r in prompt_responses
                        if r.end_of_segment
                        and not r.end_of_prompt
                        and batch_results[r.uid]["segment_types"]
                    ]
                    caches = batch_generator.extract_cache(eos_ids)
                    for uid, (cache, cache_key) in caches.items():
                        self.prompt_cache.insert_cache(
                            self.model_provider.model_key,
                            cache_key[:],
                            cache,
                            cache_type=batch_results[uid]["segment_types"].pop(),
                        )
                    del caches

                    for r in gen_responses:
                        result = batch_results[r.uid]
                        result["detokenizer"].add_token(r.token)
                        result["rqueue"].put(
                            Response(
                                result["detokenizer"].last_segment,
                                r.token,
                                r.current_state,
                                r.match_sequence,
                                r.logprobs[r.token].item(),
                                r.finish_reason,
                                _format_top_logprobs(
                                    r.logprobs,
                                    result["top_logprobs"],
                                    current_tokenizer,
                                ),
                            )
                        )

                        if r.finish_reason is not None:
                            result["rqueue"].put(None)
                            self.prompt_cache.insert_cache(
                                current_model_key,
                                r.all_tokens[:],
                                r.prompt_cache,
                                cache_type="assistant",
                            )
                            del batch_results[r.uid]

                        if result["ctx"]._should_stop:
                            uids_to_remove.append(r.uid)

                uids_to_remove = self._share_object(uids_to_remove)
                if uids_to_remove:
                    batch_generator.remove(uids_to_remove)
                    for uid in uids_to_remove:
                        # It may have already been removed during
                        # generation
                        batch_results.pop(uid, None)

        if batch_generator is not None:
            batch_generator.close()

    def _serve_single(self, request):
        rqueue, request, args = request

        # Define the progress callback
        progress_started = False
        ctx = None

        def progress(tokens_processed, tokens_total):
            nonlocal progress_started
            if not progress_started and ctx is not None:
                ctx.prompt_cache_count = max(ctx.prompt_cache_count, tokens_processed)
                progress_started = True
            rqueue.put((tokens_processed, tokens_total))

        try:
            # Load the model and tokenizer
            model = self.model_provider.model
            tokenizer = self.model_provider.tokenizer
            draft_model = self.model_provider.draft_model

            # Prepare the prompt and state machine
            rendered_prompt = self._render_prompt_text(tokenizer, request, args)
            rendered_checkpoint = self._load_rendered_prompt_checkpoint(
                tokenizer,
                rendered_prompt,
            )
            if rendered_checkpoint is None:
                tokenized = self._tokenize(
                    tokenizer,
                    request,
                    args,
                    return_rendered=True,
                )
                prompt, segments, segment_types, initial_state, rendered_prompt = (
                    tokenized
                )
            else:
                prompt = rendered_checkpoint.prompt
                if rendered_checkpoint.suffix_tokens:
                    segments = [
                        rendered_checkpoint.prefix_tokens,
                        rendered_checkpoint.suffix_tokens,
                    ]
                    segment_types = ["system", "user"]
                else:
                    segments = [prompt]
                    segment_types = ["system"]
                initial_state = self._initial_state_from_prompt(tokenizer, prompt)
            sm, sequences = self._make_state_machine(
                self.model_provider.model_key,
                tokenizer,
                args.stop_words,
                initial_state=initial_state,
            )
            sm_state = sm.make_state()

            # Start the generation context
            ctx = GenerationContext(
                has_thinking=tokenizer.has_thinking,
                has_tool_calling=tokenizer.has_tool_calling,
                tool_parser=tokenizer.tool_parser,
                sequences=sequences,
                prompt=prompt,
            )
            rqueue.put(ctx)

            # Seed if requested
            if args.seed is not None:
                mx.random.seed(args.seed)

            # Make the sampler and logit processor
            sampler = _make_sampler(args, tokenizer)
            logits_processors = _make_logits_processors(args)

            # Load the KV cache
            self._log_cache_stats()
            ram_cache, ram_rest = self.prompt_cache.fetch_nearest_cache(
                self.model_provider.model_key, prompt
            )
            ram_cache_count = len(prompt) - len(ram_rest)
            if (
                rendered_checkpoint is not None
                and rendered_checkpoint.cached_tokens >= ram_cache_count
            ):
                cache = rendered_checkpoint.prompt_cache
                ctx.prompt_cache_count = rendered_checkpoint.cached_tokens
                rest = prompt[ctx.prompt_cache_count :]
            else:
                cache = ram_cache
                ctx.prompt_cache_count = ram_cache_count
                rest = ram_rest
            cache_key = prompt[:]
            if cache is None:
                cache = make_prompt_cache(self.model_provider.model)
                if self.model_provider.draft_model is not None:
                    cache += make_prompt_cache(self.model_provider.draft_model)

            checkpoint_prefix_lengths = _prompt_checkpoint_store_prefix_lengths(
                self.cli_args,
                prompt,
                segments,
                segment_types,
                ctx.prompt_cache_count,
            )
            checkpoint_frontier_min_tokens, checkpoint_frontier_stride_tokens = (
                _prompt_checkpoint_continued_frontier_args(self.cli_args)
            )
            prompt_token_count = len(prompt)
            prompt_checkpoint_save_exact = _prompt_checkpoint_save_exact_for_prompt(
                self.cli_args,
                prompt_token_count,
            )
            generated_text_parts = []
            finish_reason = None
            decode_progress_interval = _prompt_checkpoint_policy_int(
                self.cli_args,
                "decode_progress_interval_tokens",
                0,
            )
            logging.info(
                "generation request: prompt_tokens=%s max_tokens=%s "
                "stop_words=%s prompt_cached_tokens=%s",
                prompt_token_count,
                args.max_tokens,
                len(args.stop_words),
                ctx.prompt_cache_count,
            )

            # Process the prompt and generate tokens
            generation_started_at = time.perf_counter()
            first_generated_token_at = None
            draft_tokens = 0
            for gen in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=rest,
                max_tokens=args.max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=cache,
                draft_model=draft_model,
                num_draft_tokens=args.num_draft_tokens,
                prompt_progress_callback=progress,
                prefill_step_size=self.cli_args.prefill_step_size,
                prefill_max_qk_tokens=self.cli_args.prefill_max_qk_tokens,
                glm_dsa_adaptive_prefill_step_size=(
                    self.cli_args.glm_dsa_adaptive_prefill_step_size
                ),
                glm_dsa_adaptive_prefill_after_tokens=(
                    self.cli_args.glm_dsa_adaptive_prefill_after_tokens
                ),
                glm_dsa_adaptive_prefill_min_remaining_tokens=(
                    self.cli_args.glm_dsa_adaptive_prefill_min_remaining_tokens
                ),
                kv_bits=self.cli_args.kv_bits,
                kv_group_size=self.cli_args.kv_group_size,
                quantized_kv_start=self.cli_args.quantized_kv_start,
                prompt_checkpoint_full_prompt=prompt,
                prompt_checkpoint_initial_cached_tokens=ctx.prompt_cache_count,
                prompt_checkpoint_store_prefix_lengths=checkpoint_prefix_lengths,
                prompt_checkpoint_allow_existing_cache=True,
                prompt_checkpoint_save_exact=prompt_checkpoint_save_exact,
                prompt_checkpoint_frontier_min_tokens=checkpoint_frontier_min_tokens,
                prompt_checkpoint_frontier_stride_tokens=(
                    checkpoint_frontier_stride_tokens
                ),
                prompt_checkpoint_rendered_prompt=rendered_prompt,
            ):
                generated_token_at = time.perf_counter()
                if first_generated_token_at is None:
                    first_generated_token_at = generated_token_at
                if getattr(gen, "from_draft", False):
                    draft_tokens += 1
                finish_reason = gen.finish_reason
                sm_state, match_sequence, current_state = sm.match(sm_state, gen.token)
                if match_sequence is not None and current_state is None:
                    finish_reason = "stop"
                rqueue.put(
                    Response(
                        gen.text,
                        gen.token,
                        current_state,
                        match_sequence,
                        gen.logprobs[gen.token].item(),
                        finish_reason,
                        _format_top_logprobs(
                            gen.logprobs, args.top_logprobs, tokenizer
                        ),
                    )
                )
                cache_key.append(gen.token)
                if gen.text:
                    generated_text_parts.append(gen.text)

                generated_tokens_so_far = max(0, len(cache_key) - prompt_token_count)
                if generated_tokens_so_far == 1:
                    logging.info(
                        "decode first token: prompt_tokens=%s cache_tokens=%s "
                        "first_token_seconds=%.3f state=%s",
                        prompt_token_count,
                        len(cache_key),
                        generated_token_at - generation_started_at,
                        current_state,
                    )
                if (
                    decode_progress_interval > 0
                    and generated_tokens_so_far > 0
                    and generated_tokens_so_far % decode_progress_interval == 0
                ):
                    decode_elapsed = max(
                        generated_token_at - first_generated_token_at,
                        1e-9,
                    )
                    logging.info(
                        "generation progress: prompt_tokens=%s generated_tokens=%s "
                        "cache_tokens=%s decode_seconds=%.3f decode_tps=%.3f "
                        "state=%s draft_tokens=%s stopped_by_client=%s",
                        prompt_token_count,
                        generated_tokens_so_far,
                        len(cache_key),
                        decode_elapsed,
                        generated_tokens_so_far / decode_elapsed,
                        current_state,
                        draft_tokens,
                        ctx._should_stop,
                    )

                if ctx._should_stop:
                    if self._is_distributed:
                        raise NotImplementedError()
                    break

                if finish_reason is not None:
                    break

            rqueue.put(None)
            generated_tokens = max(0, len(cache_key) - prompt_token_count)
            generation_finished_at = time.perf_counter()
            generation_seconds = max(
                generation_finished_at - generation_started_at, 0.0
            )
            generation_tps = (
                generated_tokens / generation_seconds if generation_seconds > 0 else 0
            )
            decode_seconds = 0.0
            if first_generated_token_at is not None:
                decode_seconds = max(
                    generation_finished_at - first_generated_token_at,
                    0.0,
                )
            decode_tps = generated_tokens / decode_seconds if decode_seconds > 0 else 0
            logging.info(
                "generation complete: prompt_tokens=%s generated_tokens=%s "
                "finish_reason=%s stopped_by_client=%s cache_tokens=%s "
                "generation_seconds=%.3f generation_tps=%.3f "
                "decode_seconds=%.3f decode_tps=%.3f draft_tokens=%s",
                prompt_token_count,
                generated_tokens,
                finish_reason,
                ctx._should_stop,
                len(cache_key),
                generation_seconds,
                generation_tps,
                decode_seconds,
                decode_tps,
                draft_tokens,
            )

            rendered_continuation = None
            if rendered_prompt is not None and generated_text_parts:
                rendered_continuation = rendered_prompt + "".join(generated_text_parts)
            if (
                _prompt_checkpoint_post_response_save_mode(self.cli_args) == "async"
                and hasattr(self, "_checkpoint_save_queue")
            ):
                checkpoint_cache_key = list(cache_key)
                self._enqueue_checkpoint_save(
                    "continued",
                    lambda checkpoint_cache_key=checkpoint_cache_key: (
                        self._save_continued_prompt_checkpoint(
                            tokenizer,
                            cache,
                            checkpoint_cache_key,
                            prompt_token_count=prompt_token_count,
                            rendered_continuation=rendered_continuation,
                        )
                    ),
                )
                delta_cache_key = list(cache_key)
                self._enqueue_checkpoint_save(
                    "delta",
                    lambda delta_cache_key=delta_cache_key: (
                        self._save_delta_prompt_checkpoint(
                            tokenizer,
                            cache,
                            delta_cache_key,
                            base_checkpoint=rendered_checkpoint,
                            rendered_continuation=rendered_continuation,
                        )
                    ),
                )
            else:
                self._save_continued_prompt_checkpoint(
                    tokenizer,
                    cache,
                    cache_key,
                    prompt_token_count=prompt_token_count,
                    rendered_continuation=rendered_continuation,
                )
                self._save_delta_prompt_checkpoint(
                    tokenizer,
                    cache,
                    cache_key,
                    base_checkpoint=rendered_checkpoint,
                    rendered_continuation=rendered_continuation,
                )

            # Save the KV cache again
            self.prompt_cache.insert_cache(
                self.model_provider.model_key, cache_key, cache
            )

        except Exception as e:
            rqueue.put(e)

    def generate(
        self,
        request: CompletionRequest,
        generation_args: GenerationArguments,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        idle_callback: Optional[Callable[[], None]] = None,
    ):
        response_queue = Queue()
        self.requests.put((response_queue, request, generation_args))

        def _inner():
            while True:
                if idle_callback is None:
                    response = response_queue.get()
                else:
                    try:
                        response = response_queue.get(timeout=15)
                    except QueueEmpty:
                        idle_callback()
                        continue
                if response is None:
                    break
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, tuple):
                    if progress_callback is not None:
                        progress_callback(*response)
                    continue
                yield response

        ctx = response_queue.get()
        if isinstance(ctx, Exception):
            raise ctx

        return ctx, _process_control_tokens(ctx, _inner())

    @property
    def cli_args(self):
        return self.model_provider.cli_args


class APIHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        response_generator: ResponseGenerator,
        *args,
        system_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        """
        Create static request specific metadata
        """
        self.created = int(time.time())
        self.response_generator = response_generator
        self.system_fingerprint = system_fingerprint or get_system_fingerprint()
        super().__init__(*args, **kwargs)

    def _write_response_bytes(self, data: bytes, *, flush: bool = True) -> bool:
        try:
            self.wfile.write(data)
            if flush:
                self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            if not getattr(self, "_client_disconnected_logged", False):
                logging.info("Client disconnected while writing response: %s", e)
                self._client_disconnected_logged = True
            return False

    def _end_headers_safely(self) -> bool:
        try:
            self.end_headers()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            if not getattr(self, "_client_disconnected_logged", False):
                logging.info(
                    "Client disconnected while writing response headers: %s",
                    e,
                )
                self._client_disconnected_logged = True
            return False

    def _set_cors_headers(self):
        allowed_origins = self.response_generator.cli_args.allowed_origins
        origin = self.headers.get("Origin")
        if "*" in allowed_origins:
            self.send_header("Access-Control-Allow-Origin", "*")
        elif origin in allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _set_completion_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self._set_cors_headers()

    def _set_stream_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._set_cors_headers()

    def do_OPTIONS(self):
        self._set_completion_headers(204)
        self.end_headers()

    def do_POST(self):
        """
        Respond to a POST request from a client.
        """
        request_factories = {
            "/v1/completions": self.handle_text_completions,
            "/v1/chat/completions": self.handle_chat_completions,
            "/chat/completions": self.handle_chat_completions,
            "/v1/responses": self.handle_responses,
            "/responses": self.handle_responses,
        }

        if self.path not in request_factories:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Fetch and parse request body
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self._set_completion_headers(411)
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": "Content-Length header is required"}).encode()
            )
            return
        try:
            content_length = int(content_length)
        except ValueError:
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": "Invalid Content-Length header"}).encode()
            )
            return
        raw_body = self.rfile.read(content_length)
        try:
            self.body = json.loads(raw_body.decode())
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {e} - Raw body: {raw_body.decode()}")
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": f"Invalid JSON in request body: {e}"}).encode()
            )
            return

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            debug_body = json.dumps(self.body, indent="\t")
            logging.debug(f"Incoming Request Body: {debug_body}")
        if not isinstance(self.body, dict):
            debug_body = json.dumps(self.body, indent="\t")
            logging.error(f"Invalid Request Body: {debug_body}")
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": "Request should be a JSON dictionary"}).encode()
            )
            return

        # Extract request parameters from the body
        self.stream = self.body.get("stream", False)
        self.stream_options = self.body.get("stream_options", None)
        self.requested_model = self.body.get("model", "default_model")
        self.requested_draft_model = self.body.get("draft_model", "default_model")
        self.num_draft_tokens = self.body.get(
            "num_draft_tokens", self.response_generator.cli_args.num_draft_tokens
        )
        self.adapter = self.body.get("adapters", None)
        (
            self.max_tokens,
            self.max_tokens_source,
            self.requested_max_tokens,
            self.max_tokens_floor_applied,
        ) = _resolve_request_max_tokens(
            self.body,
            self.response_generator.cli_args,
        )
        self.temperature = self.body.get(
            "temperature", self.response_generator.cli_args.temp
        )
        self.top_p = self.body.get("top_p", self.response_generator.cli_args.top_p)
        self.top_k = self.body.get("top_k", self.response_generator.cli_args.top_k)
        self.min_p = self.body.get("min_p", self.response_generator.cli_args.min_p)
        self.repetition_penalty = self.body.get("repetition_penalty", 0.0)
        self.repetition_context_size = self.body.get("repetition_context_size", 20)
        self.presence_penalty = self.body.get("presence_penalty", 0.0)
        self.presence_context_size = self.body.get("presence_context_size", 20)
        self.frequency_penalty = self.body.get("frequency_penalty", 0.0)
        self.frequency_context_size = self.body.get("frequency_context_size", 20)
        self.xtc_probability = self.body.get("xtc_probability", 0.0)
        self.xtc_threshold = self.body.get("xtc_threshold", 0.0)
        self.logit_bias = self.body.get("logit_bias", None)
        self.logprobs = self.body.get("logprobs", False)
        self.top_logprobs = self.body.get("top_logprobs", -1)
        self.seed = self.body.get("seed", None)
        self.chat_template_kwargs = self.body.get("chat_template_kwargs")
        self.validate_model_parameters()
        logging.info(
            "request parameters: path=%s stream=%s model=%s "
            "max_tokens=%s requested_max_tokens=%s max_tokens_source=%s "
            "max_tokens_floor_applied=%s temperature=%s top_p=%s",
            self.path,
            self.stream,
            self.requested_model,
            self.max_tokens,
            self.requested_max_tokens,
            self.max_tokens_source,
            self.max_tokens_floor_applied,
            self.temperature,
            self.top_p,
        )

        # Get stop sequences
        stop_words = self.body.get("stop")
        stop_words = stop_words or []
        stop_words = [stop_words] if isinstance(stop_words, str) else stop_words

        # Create the completion request
        request = request_factories[self.path]()
        if self.object_type == "response":
            self.handle_responses_completion(request, stop_words)
        else:
            self.handle_completion(request, stop_words)

    def _validate(
        self,
        name,
        expected_type,
        min_val=None,
        max_val=None,
        optional=False,
        whitelist=None,
    ):
        value = getattr(self, name)
        if optional and value is None:
            return
        if not isinstance(value, expected_type):
            try:
                allowed = tuple(et.__name__ for et in expected_type)
            except TypeError:
                allowed = expected_type.__name__
            raise ValueError(f"{name} must be of type {allowed}")
        if whitelist is not None and value in whitelist:
            return
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be at least {min_val}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be at most {max_val}")

    def validate_model_parameters(self):
        """Validate that the passed model parameters have correct types and values."""
        self._validate("stream", bool)
        self._validate("max_tokens", int, min_val=0)
        self._validate("temperature", (float, int), min_val=0)
        self._validate("top_p", (float, int), min_val=0, max_val=1)
        self._validate("top_k", int, min_val=0)
        self._validate("min_p", (float, int), min_val=0, max_val=1)
        self._validate("num_draft_tokens", int, min_val=0)
        self._validate("repetition_penalty", (float, int), min_val=0)
        self._validate("repetition_context_size", int, min_val=0)
        self._validate("presence_penalty", (float, int))
        self._validate("presence_context_size", int, min_val=0)
        self._validate("frequency_penalty", (float, int))
        self._validate("frequency_context_size", int, min_val=0)
        self._validate("logprobs", bool)
        self._validate("top_logprobs", int, min_val=0, max_val=11, whitelist=[-1])
        self._validate("xtc_probability", float, min_val=0, max_val=1)
        self._validate("xtc_threshold", float, min_val=0, max_val=1)
        self._validate("requested_model", str)
        self._validate("adapter", str, optional=True)
        self._validate("seed", int, optional=True)
        self._validate("logit_bias", dict, optional=True)

        if self.logit_bias is not None:
            try:
                self.logit_bias = {int(k): float(v) for k, v in self.logit_bias.items()}
            except ValueError:
                raise ValueError("logit_bias must be a dict of int to float")

    def generate_response(
        self,
        text: str,
        finish_reason: Union[Literal["length", "stop"], None],
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        prompt_cache_count: Optional[int] = None,
        token_logprobs: Optional[List[float]] = None,
        top_tokens: Optional[List[Tuple[Dict[str, Any]]]] = None,
        tokens: Optional[List[int]] = None,
        tool_calls: Optional[List[str]] = None,
        reasoning_text: Optional[str] = None,
    ) -> dict:
        """
        Generate a single response packet based on response type (stream or
        not), completion type and parameters.

        Args:
            text (str): Text generated by model
            finish_reason (Union[Literal["length", "stop"], None]): The reason the
              response is being sent: "length", "stop" or `None`.
            prompt_token_count (Optional[int]): The number of tokens in the prompt,
              used to populate the "usage" field (not used when stream).
            completion_token_count (Optional[int]): The number of tokens in the
              response, used to populate the "usage" field (not used when stream).
            prompt_cache_count (Optional[int]): The portion of prompt_token_count
              that was found in the cache when servicing the request.
            token_logprobs (Optional[List[float]]): The log probabilities per token,
              in token order.
            top_tokens (Optional[List[Tuple[Dict[str, Any]]]]): List of outputs from
              _format_top_logprobs, giving info on the top N tokens at each token position.
            tokens (Optional[List[int]]): List of tokens to return with logprobs structure
            tool_calls (Optional[List[str]]): List of tool calls.
            reasoning_text (Optional[str]): The reasoning text generated by the model.

        Returns:
            dict: A dictionary containing the response, in the same format as
              OpenAI's API.
        """
        token_logprobs = token_logprobs or []
        top_logprobs = top_tokens or []
        tool_calls = tool_calls or []

        # Static response
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": self.object_type,
            "model": self.requested_model,
            "created": self.created,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                },
            ],
        }

        if top_logprobs:
            response["choices"][0]["logprobs"] = {
                "content": [
                    dict(i[0], top_logprobs=i) if i else {} for i in top_logprobs
                ]
            }
        elif token_logprobs:
            response["choices"][0]["logprobs"] = {
                "content": [
                    dict(id=i, logprob=g) for i, g in zip(tokens, token_logprobs)
                ]
            }

        if not self.stream:
            if not (
                isinstance(prompt_token_count, int)
                and isinstance(completion_token_count, int)
            ):
                raise ValueError(
                    "Response type is complete, but token counts not provided"
                )

            response["usage"] = {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            }
            if prompt_cache_count is not None and prompt_cache_count >= 0:
                response["usage"]["prompt_tokens_details"] = {
                    "cached_tokens": prompt_cache_count,
                }

        choice = response["choices"][0]

        # Add dynamic response
        if self.object_type.startswith("chat.completion"):
            key_name = "delta" if self.stream else "message"
            choice[key_name] = {"role": "assistant"}
            if text:
                choice[key_name]["content"] = text
            if reasoning_text:
                choice[key_name]["reasoning"] = reasoning_text
            if tool_calls:
                choice[key_name]["tool_calls"] = tool_calls
        elif self.object_type == "text_completion":
            choice.update(text=text)
        else:
            raise ValueError(f"Unsupported response type: {self.object_type}")

        return response

    def handle_completion(self, request: CompletionRequest, stop_words: List[str]):
        """
        Generate a response to a prompt and send it to the client in a single batch.

        Args:
            prompt (List[int]): The tokenized prompt.
            stop_words (List[str]): A list of stop words
        """
        args = GenerationArguments(
            model=ModelDescription(
                model=self.requested_model,
                draft=self.requested_draft_model,
                adapter=self.adapter,
            ),
            sampling=SamplingArguments(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                xtc_probability=self.xtc_probability,
                xtc_threshold=self.xtc_threshold,
            ),
            logits=LogitsProcessorArguments(
                logit_bias=self.logit_bias,
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
                presence_penalty=self.presence_penalty,
                presence_context_size=self.presence_context_size,
                frequency_penalty=self.frequency_penalty,
                frequency_context_size=self.frequency_context_size,
            ),
            stop_words=stop_words,
            max_tokens=self.max_tokens,
            num_draft_tokens=self.num_draft_tokens,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
            seed=self.seed,
            chat_template_kwargs=self.chat_template_kwargs,
        )

        client_connected = True
        ctx = None

        def stream_write(data: bytes) -> bool:
            nonlocal client_connected
            if not client_connected:
                return False
            if self._write_response_bytes(data):
                return True
            client_connected = False
            if ctx is not None:
                ctx.stop()
            return False

        prefill_progress_interval = _prompt_checkpoint_policy_int(
            self.response_generator.cli_args,
            "prefill_progress_interval_tokens",
            0,
        )
        prefill_started_at = time.perf_counter()
        prefill_initial_processed = None
        prefill_last_logged = None

        # Keep connection alive during long prompt processing (and also log
        # the progress).
        def keepalive_callback(processed, total):
            nonlocal prefill_initial_processed, prefill_last_logged
            if prefill_initial_processed is None:
                prefill_initial_processed = processed
            should_log = (
                prefill_progress_interval <= 0
                or prefill_last_logged is None
                or processed >= total
                or processed - prefill_last_logged >= prefill_progress_interval
            )
            if should_log:
                elapsed = max(time.perf_counter() - prefill_started_at, 1e-9)
                cached_tokens = prefill_initial_processed
                fresh_processed = max(processed - cached_tokens, 0)
                fresh_total = max(total - cached_tokens, 0)
                progress_pct = (processed / total * 100.0) if total > 0 else 100.0
                prefill_tps = fresh_processed / elapsed if fresh_processed else 0.0
                logging.info(
                    "Prompt processing progress: processed_tokens=%s "
                    "total_tokens=%s cached_tokens=%s fresh_processed_tokens=%s "
                    "fresh_total_tokens=%s progress_pct=%.2f "
                    "elapsed_seconds=%.3f prefill_tps=%.3f",
                    processed,
                    total,
                    cached_tokens,
                    fresh_processed,
                    fresh_total,
                    progress_pct,
                    elapsed,
                    prefill_tps,
                )
                prefill_last_logged = processed
            if self.stream:
                msg = f": keepalive {processed}/{total}\n\n".encode()
                stream_write(msg)

        def idle_callback():
            if self.stream:
                stream_write(b": keepalive decode\n\n")

        # Create the token generator
        try:
            ctx, response = self.response_generator.generate(
                request,
                args,
                progress_callback=keepalive_callback,
                idle_callback=idle_callback,
            )
        except Exception as e:
            self._set_completion_headers(404)
            if self._end_headers_safely():
                self._write_response_bytes(json.dumps({"error": str(e)}).encode())
            return

        # Prepare the headers
        if self.stream:
            self._set_stream_headers(200)
            if not self._end_headers_safely():
                client_connected = False
                ctx.stop()
                return
            logging.debug("Starting stream:")
        else:
            self._set_completion_headers(200)
            logging.debug("Starting completion:")

        # Tool call formatter
        tool_formatter = ToolCallFormatter(ctx.tool_parser, request.tools, self.stream)

        # Variables to save the generated text, tokens, logprobs, tools etc
        prev_state = None
        finish_reason = "stop"
        reasoning_text = ""
        made_tool_call = False
        tool_text = ""
        tool_calls = []
        text = ""
        tokens = []
        token_logprobs = []
        top_tokens = []
        decode_progress_interval = _prompt_checkpoint_policy_int(
            self.response_generator.cli_args,
            "decode_progress_interval_tokens",
            0,
        )
        tool_call_max_tokens = _prompt_checkpoint_policy_int(
            self.response_generator.cli_args,
            "tool_call_max_tokens",
            0,
        )
        decode_started_at = time.perf_counter()
        tool_state_tokens = 0
        tool_call_limit_reached = False
        loop_guard = TokenLoopGuard(
            self.response_generator.cli_args.loop_guard_ngram_size,
            self.response_generator.cli_args.loop_guard_repeats,
            self.response_generator.cli_args.loop_guard_min_tokens,
        )

        try:
            for gen in response:
                logging.debug(gen.text)

                # Collect the text according to our current state and state
                # transitions. Reasoning or tool or normal text.
                if gen.state == "reasoning":
                    reasoning_text += gen.text
                elif gen.state == "tool":
                    tool_text += gen.text
                elif gen.state == "normal":
                    if prev_state == "tool":
                        tool_calls.append(tool_text)
                        tool_text = ""
                        made_tool_call = True
                    text += gen.text

                # Add the tokens and logprobs to the vars.
                tokens.append(gen.token)
                if gen.state == "tool":
                    tool_state_tokens += 1
                else:
                    tool_state_tokens = 0
                if (
                    decode_progress_interval > 0
                    and len(tokens) % decode_progress_interval == 0
                ):
                    elapsed = max(time.perf_counter() - decode_started_at, 1e-9)
                    logging.info(
                        "Decode progress: generated_tokens=%s "
                        "elapsed_seconds=%.3f generation_tps=%.3f state=%s",
                        len(tokens),
                        elapsed,
                        len(tokens) / elapsed,
                        gen.state,
                    )
                if args.logprobs:
                    token_logprobs.append(gen.logprob)
                if args.top_logprobs > 0:
                    top_tokens.append(gen.top_tokens)
                if loop_guard.append(gen.token):
                    logging.warning(
                        "Stopping generation after detecting repeated token loop "
                        "(ngram_size=%s repeats=%s generated_tokens=%s)",
                        loop_guard.ngram_size,
                        loop_guard.repeats,
                        len(tokens),
                    )
                    finish_reason = "stop"
                    ctx.stop()
                    break
                if (
                    tool_call_max_tokens > 0
                    and tool_state_tokens > tool_call_max_tokens
                ):
                    logging.warning(
                        "Stopping generation after tool call exceeded token limit "
                        "(tool_call_max_tokens=%s generated_tokens=%s "
                        "tool_state_tokens=%s)",
                        tool_call_max_tokens,
                        len(tokens),
                        tool_state_tokens,
                    )
                    finish_reason = "length"
                    tool_call_limit_reached = True
                    ctx.stop()
                    break

                if (
                    self.stream
                    and gen.state != "tool"
                    and (text or tool_calls or reasoning_text)
                ):
                    resp = self.generate_response(
                        text,
                        None,
                        tool_calls=tool_formatter(tool_calls),
                        reasoning_text=reasoning_text,
                    )
                    if not stream_write(f"data: {json.dumps(resp)}\n\n".encode()):
                        break
                    reasoning_text = ""
                    text = ""
                    tool_calls = []

                if gen.finish_reason is not None:
                    finish_reason = gen.finish_reason

                prev_state = gen.state

            if prev_state == "tool" and tool_text and not tool_call_limit_reached:
                tool_calls.append(tool_text)
                made_tool_call = True

            if finish_reason == "stop" and made_tool_call:
                finish_reason = "tool_calls"

            logging.info(
                "completion finished: request_id=%s prompt_tokens=%s "
                "generated_tokens=%s finish_reason=%s stream=%s "
                "client_connected=%s max_tokens=%s max_tokens_source=%s "
                "requested_max_tokens=%s max_tokens_floor_applied=%s "
                "made_tool_call=%s tool_call_limit_reached=%s",
                self.request_id,
                len(ctx.prompt),
                len(tokens),
                finish_reason,
                self.stream,
                client_connected,
                self.max_tokens,
                self.max_tokens_source,
                self.requested_max_tokens,
                self.max_tokens_floor_applied,
                made_tool_call,
                tool_call_limit_reached,
            )

            if self.stream:
                if client_connected:
                    resp = self.generate_response(
                        text,
                        finish_reason,
                        tool_calls=tool_formatter(tool_calls),
                        reasoning_text=reasoning_text,
                    )
                    stream_write(f"data: {json.dumps(resp)}\n\n".encode())
                    if (
                        self.stream_options is not None
                        and self.stream_options["include_usage"]
                        and client_connected
                    ):
                        resp = self.completion_usage_response(
                            len(ctx.prompt),
                            len(tokens),
                            ctx.prompt_cache_count,
                        )
                        stream_write(f"data: {json.dumps(resp)}\n\n".encode())
                    if client_connected:
                        stream_write(b"data: [DONE]\n\n")
            else:
                resp = self.generate_response(
                    text,
                    finish_reason,
                    len(ctx.prompt),
                    len(tokens),
                    ctx.prompt_cache_count,
                    token_logprobs=token_logprobs,
                    top_tokens=top_tokens,
                    tokens=tokens,
                    reasoning_text=reasoning_text,
                    tool_calls=tool_formatter(tool_calls),
                )
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    response_debug = json.dumps(resp, indent="\t")
                    logging.debug(f"Outgoing Response: {response_debug}")

                response_json = json.dumps(resp).encode()
                self.send_header("Content-Length", str(len(response_json)))
                if self._end_headers_safely():
                    self._write_response_bytes(response_json)
        finally:
            ctx.stop()

    def completion_usage_response(
        self,
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        prompt_cache_count: Optional[int] = None,
    ):
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": "chat.completion",
            "model": self.requested_model,
            "created": self.created,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            },
        }
        if prompt_cache_count is not None and prompt_cache_count >= 0:
            response["usage"]["prompt_tokens_details"] = {
                "cached_tokens": prompt_cache_count,
            }
        return response

    def handle_responses(self) -> CompletionRequest:
        """
        Parse an OpenAI Responses API request and return a CompletionRequest.
        Translates the Responses API schema to the internal chat-completion
        representation so the existing generation pipeline can be reused.
        """
        body = self.body
        self.request_id = f"resp_{uuid.uuid4().hex}"
        self.object_type = "response"

        if "max_output_tokens" in body:
            self.max_tokens = body["max_output_tokens"]

        # Collect all system/developer messages first, then non-system messages.
        # Many models require system messages at the beginning only.
        system_parts = []
        non_system_messages = []

        instructions = body.get("instructions")
        if instructions:
            system_parts.append(instructions)

        inp = body.get("input", "")
        if isinstance(inp, str):
            non_system_messages.append({"role": "user", "content": inp})
        elif isinstance(inp, list):
            for item in inp:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "message":
                        content = item.get("content", "")
                        if isinstance(content, list):
                            parts = []
                            for c in content:
                                if isinstance(c, dict):
                                    text = c.get("text", "")
                                    if text:
                                        parts.append(text)
                                elif isinstance(c, str):
                                    parts.append(c)
                            content = "\n".join(parts) if parts else ""
                        role = item.get("role", "user")
                        if role in ("developer", "system"):
                            system_parts.append(content)
                            continue
                        non_system_messages.append({"role": role, "content": content})
                    elif item_type == "function_call":
                        non_system_messages.append({
                            "role": "assistant",
                            "tool_calls": [{
                                "id": item.get("call_id", ""),
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": item.get("arguments", "{}"),
                                },
                            }],
                        })
                    elif item_type == "function_call_output":
                        non_system_messages.append({
                            "role": "tool",
                            "tool_call_id": item.get("call_id", ""),
                            "content": item.get("output", ""),
                        })
                    elif "role" in item:
                        role = item.get("role", "user")
                        if role in ("developer", "system"):
                            system_parts.append(item.get("content", ""))
                        else:
                            non_system_messages.append(item)
                else:
                    non_system_messages.append({"role": "user", "content": str(item)})

        # Build final messages: single system message at start + rest
        messages = []
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        messages.extend(non_system_messages)

        # Convert Responses API tools to Chat Completions format and filter
        # unsupported types (web_search, image_generation, namespace, etc.)
        raw_tools = body.get("tools") or None
        if raw_tools:
            converted_tools = []
            for tool in raw_tools:
                tool_type = tool.get("type", "")
                if tool_type == "function" and "function" not in tool:
                    converted_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                            **({"strict": tool["strict"]} if "strict" in tool else {}),
                        },
                    })
                elif tool_type == "function" and "function" in tool:
                    converted_tools.append(tool)
                else:
                    logging.debug(
                        f"Skipping unsupported tool type: {tool_type} "
                        f"({tool.get('name', 'unnamed')})"
                    )
            raw_tools = converted_tools or None

        return CompletionRequest(
            "chat",
            "",
            messages,
            raw_tools,
            body.get("role_mapping"),
        )

    def handle_responses_completion(
        self, request: CompletionRequest, stop_words: List[str]
    ):
        """
        Generate and send a response in OpenAI Responses API format.
        Supports both text and tool-call outputs with streaming.
        """
        args = GenerationArguments(
            model=ModelDescription(
                model=self.requested_model,
                draft=self.requested_draft_model,
                adapter=self.adapter,
            ),
            sampling=SamplingArguments(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                xtc_probability=self.xtc_probability,
                xtc_threshold=self.xtc_threshold,
            ),
            logits=LogitsProcessorArguments(
                logit_bias=self.logit_bias,
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
                presence_penalty=self.presence_penalty,
                presence_context_size=self.presence_context_size,
                frequency_penalty=self.frequency_penalty,
                frequency_context_size=self.frequency_context_size,
            ),
            stop_words=stop_words,
            max_tokens=self.max_tokens,
            num_draft_tokens=self.num_draft_tokens,
            logprobs=False,
            top_logprobs=-1,
            seed=self.seed,
            chat_template_kwargs=self.chat_template_kwargs,
        )

        client_connected = True
        ctx = None

        def stream_write(data: bytes) -> bool:
            nonlocal client_connected
            if not client_connected:
                return False
            if self._write_response_bytes(data):
                return True
            client_connected = False
            if ctx is not None:
                ctx.stop()
            return False

        def keepalive_callback(processed, total):
            logging.info(f"Prompt processing progress: {processed}/{total}")
            if self.stream:
                stream_write(f": keepalive {processed}/{total}\n\n".encode())

        def idle_callback():
            if self.stream:
                stream_write(b": keepalive decode\n\n")

        try:
            ctx, response_gen = self.response_generator.generate(
                request,
                args,
                progress_callback=keepalive_callback,
                idle_callback=idle_callback,
            )
        except Exception as e:
            import traceback
            logging.error(f"handle_responses_completion error: {e}")
            logging.error(traceback.format_exc())
            self._set_completion_headers(500)
            if self._end_headers_safely():
                self._write_response_bytes(json.dumps({"error": str(e)}).encode())
            return

        tool_formatter = ToolCallFormatter(ctx.tool_parser, request.tools, False)

        msg_id = f"msg_{uuid.uuid4().hex}"
        full_text = ""
        reasoning_text = ""
        tokens = []
        finish_reason = "stop"
        prev_state = None
        tool_text = ""
        tool_calls_raw = []
        made_tool_call = False
        tool_call_max_tokens = _prompt_checkpoint_policy_int(
            self.response_generator.cli_args,
            "tool_call_max_tokens",
            0,
        )
        tool_state_tokens = 0
        tool_call_limit_reached = False
        loop_guard = TokenLoopGuard(
            self.response_generator.cli_args.loop_guard_ngram_size,
            self.response_generator.cli_args.loop_guard_repeats,
            self.response_generator.cli_args.loop_guard_min_tokens,
        )

        if self.stream:
            self._set_stream_headers(200)
            if not self._end_headers_safely():
                client_connected = False
                ctx.stop()
                return
            if not stream_write(self._sse_event("response.created", {
                "type": "response.created",
                "response": {
                    "id": self.request_id, "object": "response",
                    "status": "in_progress", "model": self.requested_model, "output": [],
                },
            })):
                ctx.stop()
                return
            if not stream_write(self._sse_event("response.output_item.added", {
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": msg_id, "type": "message", "role": "assistant",
                         "content": [], "status": "in_progress"},
            })):
                ctx.stop()
                return
            if not stream_write(self._sse_event("response.content_part.added", {
                "type": "response.content_part.added", "item_id": msg_id,
                "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            })):
                ctx.stop()
                return

        try:
            for gen in response_gen:
                tokens.append(gen.token)
                if gen.state == "tool":
                    tool_state_tokens += 1
                else:
                    tool_state_tokens = 0
                if loop_guard.append(gen.token):
                    logging.warning(
                        "Stopping generation after detecting repeated token loop "
                        "(ngram_size=%s repeats=%s generated_tokens=%s)",
                        loop_guard.ngram_size,
                        loop_guard.repeats,
                        len(tokens),
                    )
                    finish_reason = "stop"
                    ctx.stop()
                    break
                if (
                    tool_call_max_tokens > 0
                    and tool_state_tokens > tool_call_max_tokens
                ):
                    logging.warning(
                        "Stopping generation after tool call exceeded token limit "
                        "(tool_call_max_tokens=%s generated_tokens=%s "
                        "tool_state_tokens=%s)",
                        tool_call_max_tokens,
                        len(tokens),
                        tool_state_tokens,
                    )
                    finish_reason = "length"
                    tool_call_limit_reached = True
                    ctx.stop()
                    break
                if gen.state == "tool":
                    tool_text += gen.text
                elif gen.state == "reasoning":
                    if gen.text:
                        reasoning_text += gen.text
                        if self.stream:
                            if not stream_write(self._sse_event(
                                "response.reasoning_text.delta", {
                                    "type": "response.reasoning_text.delta",
                                    "item_id": msg_id, "output_index": 0,
                                    "content_index": 0, "delta": gen.text,
                                })):
                                break
                elif gen.state == "normal":
                    if prev_state == "tool":
                        tool_calls_raw.append(tool_text)
                        tool_text = ""
                        made_tool_call = True
                    if gen.text:
                        full_text += gen.text
                        if self.stream:
                            if not stream_write(self._sse_event(
                                "response.output_text.delta", {
                                    "type": "response.output_text.delta",
                                    "item_id": msg_id, "output_index": 0,
                                    "content_index": 0, "delta": gen.text,
                                })):
                                break
                if gen.finish_reason is not None:
                    finish_reason = gen.finish_reason
                prev_state = gen.state

            if prev_state == "tool" and tool_text and not tool_call_limit_reached:
                tool_calls_raw.append(tool_text)
                made_tool_call = True
            if finish_reason == "stop" and made_tool_call:
                finish_reason = "tool_calls"
            logging.info(
                "responses completion finished: request_id=%s prompt_tokens=%s "
                "generated_tokens=%s finish_reason=%s stream=%s "
                "client_connected=%s max_tokens=%s max_tokens_source=%s "
                "requested_max_tokens=%s max_tokens_floor_applied=%s "
                "made_tool_call=%s tool_call_limit_reached=%s",
                self.request_id,
                len(ctx.prompt),
                len(tokens),
                finish_reason,
                self.stream,
                client_connected,
                self.max_tokens,
                self.max_tokens_source,
                self.requested_max_tokens,
                self.max_tokens_floor_applied,
                made_tool_call,
                tool_call_limit_reached,
            )
        finally:
            ctx.stop()

        if not client_connected:
            return

        # Strip leaked special tokens
        import re as _re
        full_text = _re.sub(r'<\|im_end\|>\s*$', '', full_text).rstrip()

        # Format tool calls via the model's tool parser
        formatted_tool_calls = tool_formatter(tool_calls_raw)

        # Fallback: parse tool calls from plain text JSON
        if not formatted_tool_calls and request.tools and full_text:
            import re
            json_pattern = re.compile(
                r'(?:```(?:json)?\s*\n?)?\s*(\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\})\s*(?:\n?```)?',
                re.DOTALL
            )
            match = json_pattern.search(full_text)
            if match:
                try:
                    tc_data = json.loads(match.group(1))
                    if "name" in tc_data and "arguments" in tc_data:
                        tool_names = {
                            t["function"]["name"]
                            for t in (request.tools or [])
                            if isinstance(t, dict) and t.get("type") == "function" and "function" in t
                        }
                        if tc_data["name"] in tool_names:
                            args_str = tc_data["arguments"]
                            if isinstance(args_str, dict):
                                args_str = json.dumps(args_str, ensure_ascii=False)
                            formatted_tool_calls = [{
                                "id": f"call_{uuid.uuid4().hex}",
                                "type": "function",
                                "function": {"name": tc_data["name"], "arguments": args_str},
                            }]
                            full_text = ""
                            made_tool_call = True
                            finish_reason = "tool_calls"
                            logging.info(f"Fallback tool call parsed: {tc_data['name']}")
                except (json.JSONDecodeError, KeyError):
                    pass

        # Count reasoning tokens separately
        reasoning_token_count = len(reasoning_text.split()) if reasoning_text else 0

        usage = {
            "input_tokens": len(ctx.prompt),
            "output_tokens": len(tokens),
            "total_tokens": len(ctx.prompt) + len(tokens),
        }
        if ctx.prompt_cache_count is not None and ctx.prompt_cache_count >= 0:
            usage["input_tokens_details"] = {"cached_tokens": ctx.prompt_cache_count}
        if reasoning_token_count > 0:
            usage["output_tokens_details"] = {"reasoning_tokens": reasoning_token_count}

        output_items = []
        output_index = 0

        text_item = {
            "id": msg_id, "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": full_text, "annotations": []}],
        }
        if full_text or not formatted_tool_calls:
            output_items.append(text_item)
            output_index += 1

        for tc in formatted_tool_calls:
            func = tc.get("function", {})
            arguments = func.get("arguments", "{}")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            fc_item = {
                "id": f"fc_{uuid.uuid4().hex}", "type": "function_call",
                "call_id": tc.get("id", f"call_{uuid.uuid4().hex}"),
                "name": func.get("name", ""), "arguments": arguments, "status": "completed",
            }
            output_items.append(fc_item)

        if self.stream:
            stream_write(self._sse_event("response.output_text.done", {
                "type": "response.output_text.done", "item_id": msg_id,
                "output_index": 0, "content_index": 0, "text": full_text,
            }))
            stream_write(self._sse_event("response.content_part.done", {
                "type": "response.content_part.done", "item_id": msg_id,
                "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": full_text, "annotations": []},
            }))
            stream_write(self._sse_event("response.output_item.done", {
                "type": "response.output_item.done", "output_index": 0, "item": text_item,
            }))
            for i, fc_item in enumerate(
                item for item in output_items if item["type"] == "function_call"
            ):
                tc_idx = output_index + i
                stream_write(self._sse_event("response.output_item.added", {
                    "type": "response.output_item.added", "output_index": tc_idx,
                    "item": {**fc_item, "arguments": "", "status": "in_progress"},
                }))
                stream_write(self._sse_event("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": fc_item["id"], "output_index": tc_idx,
                    "delta": fc_item["arguments"],
                }))
                stream_write(self._sse_event("response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "item_id": fc_item["id"], "output_index": tc_idx,
                    "arguments": fc_item["arguments"],
                }))
                stream_write(self._sse_event("response.output_item.done", {
                    "type": "response.output_item.done", "output_index": tc_idx, "item": fc_item,
                }))
            stream_write(self._sse_event("response.completed", {
                "type": "response.completed",
                "response": {
                    "id": self.request_id, "object": "response", "status": "completed",
                    "model": self.requested_model, "output": output_items, "usage": usage,
                    "end_turn": finish_reason != "tool_calls",
                },
            }))
        else:
            resp = {
                "id": self.request_id, "object": "response", "created_at": self.created,
                "model": self.requested_model,
                "status": "completed" if finish_reason != "length" else "incomplete",
                "output": output_items, "usage": usage, "error": None,
                "incomplete_details": None, "instructions": self.body.get("instructions"),
                "metadata": {}, "parallel_tool_calls": True,
                "temperature": self.temperature, "tool_choice": "auto",
                "tools": [], "top_p": self.top_p, "truncation": "disabled",
            }
            self._set_completion_headers(200)
            resp_bytes = json.dumps(resp).encode()
            self.send_header("Content-Length", str(len(resp_bytes)))
            if self._end_headers_safely():
                self._write_response_bytes(resp_bytes)

    def _sse_event(self, event: str, data: dict) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    def handle_chat_completions(self) -> CompletionRequest:
        """
        Handle a chat completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        body = self.body
        assert "messages" in body, "Request did not contain messages"

        # Determine response type
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"

        return CompletionRequest(
            "chat",
            "",
            body["messages"],
            body.get("tools") or None,
            body.get("role_mapping"),
        )

    def handle_text_completions(self) -> CompletionRequest:
        """
        Handle a text completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        # Determine response type
        self.request_id = f"cmpl-{uuid.uuid4()}"
        self.object_type = "text_completion"
        assert "prompt" in self.body, "Request did not contain a prompt"
        return CompletionRequest(
            "text",
            self.body["prompt"],
            [],
            None,
            None,
        )

    def do_GET(self):
        """
        Respond to a GET request from a client.
        """
        if self.path.startswith("/v1/models"):
            self.handle_models_request()
        elif self.path == "/health":
            self.handle_health_check()
        else:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def handle_health_check(self):
        """
        Handle a GET request for the /health endpoint.
        """
        self._set_completion_headers(200)
        self.end_headers()

        self.wfile.write('{"status": "ok"}'.encode())
        self.wfile.flush()

    def handle_models_request(self):
        """
        Handle a GET request for the /v1/models endpoint.
        """
        self._set_completion_headers(200)
        self.end_headers()

        files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]

        parts = self.path.split("/")
        filter_repo_id = None
        if len(parts) > 3:
            filter_repo_id = "/".join(parts[3:])

        def probably_mlx_lm(repo):
            if repo.repo_type != "model":
                return False
            if "main" not in repo.refs:
                return False
            if filter_repo_id is not None and repo.repo_id != filter_repo_id:
                return False
            file_names = {f.file_path.name for f in repo.refs["main"].files}
            return all(f in file_names for f in files)

        # Scan the cache directory for downloaded mlx models
        hf_cache_info = scan_cache_dir()
        downloaded_models = [
            repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)
        ]

        # Create a list of available models
        models = [
            {
                "id": repo.repo_id,
                "object": "model",
                "created": self.created,
            }
            for repo in downloaded_models
        ]

        if self.response_generator.cli_args.model:
            model_path = Path(self.response_generator.cli_args.model)
            if model_path.exists():
                model_id = str(model_path.resolve())
                models.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": self.created,
                    }
                )

        response = {"object": "list", "data": models}

        response_json = json.dumps(response).encode()
        self.wfile.write(response_json)
        self.wfile.flush()


def _run_http_server(
    host: str,
    port: int,
    response_generator,
    server_class=ThreadingHTTPServer,
    handler_class=APIHandler,
):
    server_address = (host, port)
    infos = socket.getaddrinfo(
        *server_address, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )
    server_class.address_family, _, _, _, server_address = next(iter(infos))
    httpd = server_class(
        server_address,
        lambda *args, **kwargs: handler_class(
            response_generator,
            system_fingerprint=get_system_fingerprint(),
            *args,
            **kwargs,
        ),
    )
    logging.info(f"Starting httpd at {host} on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received; entering shutdown sequence.")
    finally:
        logging.info("HTTP server stopping...")
        response_generator.stop_and_join()
        httpd.server_close()
        logging.info("HTTP server stopped.")


def run(
    host: str,
    port: int,
    model_provider: ModelProvider,
    server_class=ThreadingHTTPServer,
    handler_class=APIHandler,
):
    group = mx.distributed.init()
    prompt_cache = LRUPromptCache(
        model_provider.cli_args.prompt_cache_size,
        max_bytes=model_provider.cli_args.prompt_cache_bytes or (1 << 63),
    )
    response_generator = ResponseGenerator(model_provider, prompt_cache)
    if group.rank() == 0:
        _run_http_server(host, port, response_generator)
    else:
        response_generator.join()


def configure_checkpoint_cache_dir(args):
    checkpoint_cache_dir = getattr(args, "checkpoint_cache_dir", None)
    if checkpoint_cache_dir is None:
        return None
    resolved = Path(checkpoint_cache_dir).expanduser().resolve()
    os.environ[PROMPT_CHECKPOINT_CACHE_DIR_ENV] = str(resolved)
    logging.info("Prompt checkpoint cache dir override: %s", resolved)
    return str(resolved)


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="MLX Http Server.")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--allowed-origins",
        type=lambda x: x.split(","),
        default="*",
        help="Allowed origins (default: *)",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding.",
        default=None,
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
        default=3,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
        required=False,
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling top-p (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k sampling (default: 0, disables top-k)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Default min-p sampling (default: 0.0, disables min-p)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--request-max-tokens-floor",
        type=int,
        default=0,
        help=(
            "Raise request max_tokens/max_completion_tokens below this value "
            "to the floor. Use 0 to honor client limits (default: 0)."
        ),
    )
    parser.add_argument(
        "--loop-guard-ngram-size",
        type=int,
        default=64,
        help=(
            "Stop generation when an exact repeated token n-gram loop is "
            "detected. Use 0 to disable (default: 64)."
        ),
    )
    parser.add_argument(
        "--decode-progress-interval-tokens",
        type=int,
        default=0,
        help=(
            "Log decode progress every N generated tokens. Use 0 to disable "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--tool-call-max-tokens",
        type=int,
        default=0,
        help=(
            "Stop generation if a single unclosed tool-call span exceeds this "
            "many generated tokens. Use 0 to disable (default: 0)."
        ),
    )
    parser.add_argument(
        "--prefill-progress-interval-tokens",
        type=int,
        default=0,
        help=(
            "Log prompt prefill progress every N newly processed tokens. "
            "Use 0 to log every prompt progress callback (default: 0)."
        ),
    )
    parser.add_argument(
        "--loop-guard-repeats",
        type=int,
        default=3,
        help="Number of consecutive repeated n-grams that trigger the loop guard.",
    )
    parser.add_argument(
        "--loop-guard-min-tokens",
        type=int,
        default=256,
        help="Minimum generated tokens before the loop guard can stop generation.",
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        help="""A JSON formatted string of arguments for the tokenizer's apply_chat_template, e.g. '{"enable_thinking":false}'""",
        default="{}",
    )
    parser.add_argument(
        "--decode-concurrency",
        type=int,
        default=32,
        help="When a request is batchable then decode that many requests in parallel",
    )
    parser.add_argument(
        "--prompt-concurrency",
        type=int,
        default=8,
        help="When a request is batchable then process that many prompts in parallel",
    )
    parser.add_argument(
        "--disable-batching",
        action="store_true",
        help=(
            "Disable continuous batching and serve requests sequentially. "
            "Useful for latency-focused long-context serving that relies on "
            "disk prompt checkpoint save/reuse."
        ),
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Step size for prefill processing (default: 2048)",
    )
    parser.add_argument(
        "--prefill-max-qk-tokens",
        type=int,
        default=DEFAULT_PREFILL_MAX_QK_TOKENS,
        help=(
            "Maximum chunk_tokens * effective_context_tokens for sequential "
            "prefill. Use 0 to disable context-aware step shrinking "
            f"(default: {DEFAULT_PREFILL_MAX_QK_TOKENS})."
        ),
    )
    parser.add_argument(
        "--glm-dsa-adaptive-prefill-step-size",
        type=int,
        default=0,
        help=(
            "Opt-in larger base prefill step for GLM DSA models before the "
            "context-aware QK cap is applied. Use 0 to disable (default: 0)."
        ),
    )
    parser.add_argument(
        "--glm-dsa-adaptive-prefill-after-tokens",
        type=int,
        default=0,
        help=(
            "Minimum processed prompt tokens before the GLM DSA adaptive "
            "prefill step can activate (default: 0)."
        ),
    )
    parser.add_argument(
        "--glm-dsa-adaptive-prefill-min-remaining-tokens",
        type=int,
        default=0,
        help=(
            "Minimum remaining prompt tokens required for the GLM DSA "
            "adaptive prefill step (default: 0)."
        ),
    )
    parser.add_argument(
        "--checkpoint-cache-dir",
        type=Path,
        help=(
            "Directory for prompt checkpoint files and manifest. Equivalent "
            f"to setting {PROMPT_CHECKPOINT_CACHE_DIR_ENV}; use an empty "
            "directory after model, tokenizer, quantization, or GLM runtime "
            "changes."
        ),
    )
    parser.add_argument(
        "--checkpoint-min-tokens",
        type=int,
        default=DEFAULT_PROMPT_CHECKPOINT_MIN_TOKENS,
        help=(
            "Minimum prompt length for ds4-style prompt checkpoint policy "
            f"(default: {DEFAULT_PROMPT_CHECKPOINT_MIN_TOKENS})."
        ),
    )
    parser.add_argument(
        "--checkpoint-cold-max-tokens",
        type=int,
        default=DEFAULT_PROMPT_CHECKPOINT_COLD_MAX_TOKENS,
        help=(
            "Maximum cold prompt length eligible for an aligned boundary "
            "checkpoint. Use 0 to disable the maximum "
            f"(default: {DEFAULT_PROMPT_CHECKPOINT_COLD_MAX_TOKENS})."
        ),
    )
    parser.add_argument(
        "--checkpoint-boundary-trim-tokens",
        type=int,
        default=DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_TRIM_TOKENS,
        help=(
            "Tokens to trim from the tail before aligning cold prompt "
            "checkpoint boundaries "
            f"(default: {DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_TRIM_TOKENS})."
        ),
    )
    parser.add_argument(
        "--checkpoint-boundary-align-tokens",
        type=int,
        default=DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_ALIGN_TOKENS,
        help=(
            "Token multiple used for stable prompt checkpoint boundaries. "
            "Use 0 to disable alignment "
            f"(default: {DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_ALIGN_TOKENS})."
        ),
    )
    parser.add_argument(
        "--checkpoint-continued-interval-tokens",
        type=int,
        default=DEFAULT_PROMPT_CHECKPOINT_CONTINUED_INTERVAL_TOKENS,
        help=(
            "Approximate token interval for automatic long-prompt frontier "
            "checkpoints. The effective interval is rounded up to "
            "--checkpoint-boundary-align-tokens. Use 0 to disable "
            f"(default: {DEFAULT_PROMPT_CHECKPOINT_CONTINUED_INTERVAL_TOKENS})."
        ),
    )
    parser.add_argument(
        "--checkpoint-max-age-seconds",
        type=int,
        default=DEFAULT_PROMPT_CHECKPOINT_MAX_AGE_SECONDS,
        help=(
            "Evict prompt checkpoint files that have not been hit or created "
            "within this many seconds. Use 0 to disable age eviction "
            f"(default: {DEFAULT_PROMPT_CHECKPOINT_MAX_AGE_SECONDS})."
        ),
    )
    parser.add_argument(
        "--checkpoint-save-exact",
        choices=["enabled", "disabled"],
        default="enabled",
        help=(
            "Save final exact full-prompt checkpoints after prefill. Disable for "
            "long-running server workloads that only need stable prefix/frontier "
            "and continued checkpoints (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-save-exact-checkpoint",
        dest="checkpoint_save_exact",
        action="store_const",
        const="disabled",
        help="Alias for --checkpoint-save-exact disabled.",
    )
    parser.add_argument(
        "--checkpoint-post-response-save-mode",
        choices=["async", "sync"],
        default="async",
        help=(
            "Run post-response continued/delta prompt checkpoint saves on a "
            "background worker or synchronously on the generation worker "
            "(default: async)."
        ),
    )
    parser.add_argument(
        "--checkpoint-async-save-shutdown-timeout",
        type=float,
        default=DEFAULT_PROMPT_CHECKPOINT_ASYNC_SHUTDOWN_TIMEOUT_SECONDS,
        help=(
            "Seconds to wait for async prompt checkpoint saves during server "
            "shutdown. Use 0 to continue shutdown immediately "
            f"(default: {DEFAULT_PROMPT_CHECKPOINT_ASYNC_SHUTDOWN_TIMEOUT_SECONDS})."
        ),
    )
    parser.add_argument(
        "--checkpoint-shutdown-save-limit",
        type=int,
        default=DEFAULT_PROMPT_CHECKPOINT_SHUTDOWN_SAVE_LIMIT,
        help=(
            "Maximum number of RAM prompt-cache entries to persist as continued "
            "checkpoints during server shutdown. Use 0 to disable "
            f"(default: {DEFAULT_PROMPT_CHECKPOINT_SHUTDOWN_SAVE_LIMIT})."
        ),
    )
    parser.add_argument(
        "--checkpoint-shutdown-max-tokens",
        type=int,
        default=DEFAULT_PROMPT_CHECKPOINT_SHUTDOWN_MAX_TOKENS,
        help=(
            "Maximum continued prompt checkpoint token length to save during "
            "server shutdown when --checkpoint-shutdown-save-limit is enabled. "
            "Use 0 to disable this cap "
            f"(default: {DEFAULT_PROMPT_CHECKPOINT_SHUTDOWN_MAX_TOKENS})."
        ),
    )
    parser.add_argument(
        "--generation-shutdown-timeout",
        type=float,
        default=DEFAULT_GENERATION_SHUTDOWN_TIMEOUT_SECONDS,
        help=(
            "Seconds to wait for the generation worker during server shutdown. "
            "Use 0 to continue shutdown immediately when model loading or "
            "generation is still active "
            f"(default: {DEFAULT_GENERATION_SHUTDOWN_TIMEOUT_SECONDS})."
        ),
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="Number of bits for KV cache quantization. GLM MLA KV "
        "quantization supports only --kv-bits 8.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=0,
        help="When --kv-bits is set, start quantizing the KV cache from "
        "this step (default: 0)",
    )
    parser.add_argument(
        "--prompt-cache-size",
        type=int,
        default=10,
        help="Maximum number of distinct KV caches to hold in the prompt cache",
    )
    parser.add_argument(
        "--prompt-cache-bytes",
        type=_parse_size,
        help="Maximum size in bytes of the KV caches",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    configure_checkpoint_cache_dir(args)
    run(args.host, args.port, ModelProvider(args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.server...` directly is deprecated."
        " Use `mlx_lm.server...` or `python -m mlx_lm server ...` instead."
    )
    main()
