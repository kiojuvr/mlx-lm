# Copyright © 2023-2024 Apple Inc.

import copy
import hashlib
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_reduce, tree_unflatten

from .base import create_causal_mask


PROMPT_CACHE_CHECKPOINT_FORMAT = "mlx_lm.prompt_cache_checkpoint"
PROMPT_CACHE_CHECKPOINT_VERSION = "1"
DEFAULT_PROMPT_CHECKPOINT_NAMESPACE = "glm52-local"
DEFAULT_PROMPT_CHECKPOINT_MODEL_ID = "default_model"
DEFAULT_PROMPT_CHECKPOINT_TOKENIZER_ID = "default_tokenizer"
GLM52_LOCAL_CACHE_ROOT = os.path.join(
    "~", ".cache", "mlx-lm", DEFAULT_PROMPT_CHECKPOINT_NAMESPACE
)
PROMPT_CHECKPOINTS_CACHE_DIR = "prompt-checkpoints"
KV_RUNTIME_CACHE_DIR = "kv"
EMPTY_ARRAYS_METADATA_KEY = "__mlx_lm_prompt_cache_empty_arrays_v1__"
PROMPT_CHECKPOINT_MANIFEST_NAME = "manifest.json"
PROMPT_CHECKPOINT_MANIFEST_VERSION = 1
PROMPT_CHECKPOINT_CACHE_DIR_ENV = "MLX_LM_PROMPT_CHECKPOINT_CACHE_DIR"
PROMPT_CHECKPOINT_MAX_FILES_ENV = "MLX_LM_PROMPT_CHECKPOINT_MAX_FILES"
PROMPT_CHECKPOINT_MAX_BYTES_ENV = "MLX_LM_PROMPT_CHECKPOINT_MAX_BYTES"
DEFAULT_PROMPT_CHECKPOINT_MAX_FILES = 256
DEFAULT_PROMPT_CHECKPOINT_MAX_BYTES = 128 * 1024**3

_CHECKPOINT_REQUIRED_METADATA_KEYS = (
    "checkpoint_format",
    "checkpoint_version",
    "checkpoint_namespace",
    "checkpoint_model_hint",
    "checkpoint_model_hint_hash",
    "checkpoint_tokenizer_hint",
    "checkpoint_tokenizer_hint_hash",
    "checkpoint_prefix_hash",
    "checkpoint_prefix_length",
    "checkpoint_cache_signature_hash",
    "checkpoint_cache_signature",
    "checkpoint_model_hint_metadata",
    "checkpoint_tokenizer_hint_metadata",
    "checkpoint_glm_dsa_metadata",
    "checkpoint_glm_mla_kv_quantization",
    "checkpoint_glm_mla_kv_settings",
)


class PromptCacheCheckpointError(ValueError):
    pass


# Trusted local GLM-5.2 runtime caches live under one root. When model
# weights, quantization, tokenizer, adapters, or GLM implementation details
# change, move or delete ~/.cache/mlx-lm/glm52-local/ to invalidate them.
def glm52_local_cache_root():
    return os.path.expanduser(GLM52_LOCAL_CACHE_ROOT)


def prompt_checkpoint_cache_dir_override():
    path = os.environ.get(PROMPT_CHECKPOINT_CACHE_DIR_ENV)
    if not path:
        return None
    return os.path.abspath(os.path.expanduser(path))


def glm52_prompt_checkpoints_dir():
    override = prompt_checkpoint_cache_dir_override()
    if override is not None:
        return override
    return os.path.join(glm52_local_cache_root(), PROMPT_CHECKPOINTS_CACHE_DIR)


def prompt_checkpoint_manifest_file():
    return os.path.join(glm52_prompt_checkpoints_dir(), PROMPT_CHECKPOINT_MANIFEST_NAME)


def glm52_kv_cache_dir():
    return os.path.join(glm52_local_cache_root(), KV_RUNTIME_CACHE_DIR)


def ensure_glm52_local_cache_dirs():
    os.makedirs(glm52_prompt_checkpoints_dir(), exist_ok=True)
    if prompt_checkpoint_cache_dir_override() is None:
        os.makedirs(glm52_kv_cache_dir(), exist_ok=True)


def _is_empty_array(value):
    return hasattr(value, "shape") and hasattr(value, "dtype") and 0 in value.shape


def make_prompt_cache(
    model: nn.Module,
    max_kv_size: Optional[int] = None,
) -> List[Any]:
    """
    Construct the model's cache for use in generation.

    This function will defer the cache construction to the model if it has a
    ``make_cache`` method, otherwise it will make a default KV cache.

    Args:
        model (nn.Module): The language model.
        max_kv_size (Optional[int]): If provided and the model does not have a
            ``make_cache`` method, a ``RotatingKVCache`` is used with a maximum
            size of ``max_kv_size``
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()

    num_layers = len(model.layers)
    if max_kv_size is not None:
        return [
            RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)
        ]
    else:
        return [KVCache() for _ in range(num_layers)]


def save_prompt_cache(
    file_name: str,
    cache: List[Any],
    metadata: Optional[Dict[str, str]] = None,
):
    """
    Save a pre-computed prompt cache to a file.

    Args:
        file_name (str): The ``.safetensors`` file name.
        cache (List[Any]): The model state.
        metadata (Dict[str, str]): Optional metadata to save along with model
            state.
    """
    metadata = dict(metadata or {})
    if EMPTY_ARRAYS_METADATA_KEY in metadata:
        raise ValueError(f"metadata uses reserved key: {EMPTY_ARRAYS_METADATA_KEY}")
    cache_data = [c.state for c in cache]
    cache_info = [c.meta_state for c in cache]
    cache_data = dict(tree_flatten(cache_data))
    empty_arrays = {}
    for key, value in list(cache_data.items()):
        if not _is_empty_array(value):
            continue
        empty_arrays[key] = {"shape": list(value.shape)}
        cache_data[key] = mx.zeros((1,), dtype=value.dtype)
    if empty_arrays:
        metadata[EMPTY_ARRAYS_METADATA_KEY] = json.dumps(
            empty_arrays,
            sort_keys=True,
            separators=(",", ":"),
        )
    cache_classes = [type(c).__name__ for c in cache]
    cache_metadata = [cache_info, metadata, cache_classes]
    cache_metadata = dict(tree_flatten(cache_metadata))
    mx.save_safetensors(file_name, cache_data, cache_metadata)


def load_prompt_cache(file_name, return_metadata=False):
    """
    Load a prompt cache from a file.

    Args:
        file_name (str): The ``.safetensors`` file name.
        return_metadata (bool): Whether or not to return metadata.
            Default: ``False``.

    Returns:
        List[Any] or Tuple[List[Any], Dict[str, str]]: The prompt cache and
            the metadata if requested.
    """
    arrays, cache_metadata = mx.load(file_name, return_metadata=True)
    cache_metadata = tree_unflatten(list(cache_metadata.items()))
    info, metadata, classes = cache_metadata
    empty_arrays = json.loads(metadata.pop(EMPTY_ARRAYS_METADATA_KEY, "{}"))
    for key, spec in empty_arrays.items():
        if key not in arrays:
            raise ValueError(f"Prompt cache is missing empty array placeholder {key}")
        arrays[key] = mx.zeros(spec["shape"], dtype=arrays[key].dtype)
    arrays = tree_unflatten(list(arrays.items()))
    cache = [
        globals()[c].from_state(state, meta_state)
        for c, state, meta_state in zip(classes, arrays, info)
    ]
    if return_metadata:
        return cache, metadata
    return cache


def _normalize_for_json(value):
    if is_dataclass(value):
        return _normalize_for_json(asdict(value))
    if isinstance(value, dict):
        return {
            str(k): _normalize_for_json(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    if hasattr(value, "tolist"):
        return _normalize_for_json(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _json_dumps(value):
    return json.dumps(
        _normalize_for_json(value),
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_hash(value):
    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()


def _token_list(tokens):
    tokens = _normalize_for_json(tokens)
    if isinstance(tokens, int) and not isinstance(tokens, bool):
        tokens = [tokens]
    if (
        isinstance(tokens, list)
        and len(tokens) == 1
        and isinstance(tokens[0], list)
    ):
        tokens = tokens[0]
    if not isinstance(tokens, list):
        raise TypeError("prefix_tokens must be a sequence of token ids")
    if any(isinstance(t, bool) or not isinstance(t, int) for t in tokens):
        raise TypeError("prefix_tokens must contain only integer token ids")
    return tokens


def prompt_prefix_hash(prefix_tokens):
    return _json_hash(_token_list(prefix_tokens))


def prompt_checkpoint_name(prefix_tokens):
    prefix = _token_list(prefix_tokens)
    return f"{_json_hash(prefix)}-{len(prefix)}.safetensors"


def prompt_checkpoint_file(prefix_tokens):
    return os.path.join(
        glm52_prompt_checkpoints_dir(),
        prompt_checkpoint_name(prefix_tokens),
    )


def _parse_prompt_checkpoint_name(name):
    suffix = ".safetensors"
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    hash_part, sep, length_part = stem.rpartition("-")
    if sep != "-" or not hash_part or not length_part:
        return None
    try:
        prefix_length = int(length_part)
    except ValueError:
        return None
    if prefix_length <= 0:
        return None
    return hash_part, prefix_length


_PROMPT_CHECKPOINT_MANIFEST_KINDS = {"exact", "prefix", "frontier", "unknown"}


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _safe_manifest_int(value, default=0):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _safe_manifest_float(value, default=None):
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _empty_prompt_checkpoint_manifest():
    return {"version": PROMPT_CHECKPOINT_MANIFEST_VERSION, "entries": {}}


def _manifest_metadata_identity(metadata):
    if not metadata:
        return {}
    identity = {}
    for key in (
        "checkpoint_namespace",
        "checkpoint_model_hint_hash",
        "checkpoint_tokenizer_hint_hash",
        "checkpoint_glm_mla_kv_quantization",
        "checkpoint_glm_mla_kv_settings",
    ):
        value = metadata.get(key)
        if value is None:
            continue
        identity[key] = str(value)
        identity[f"{key}_hash"] = hashlib.sha256(
            str(value).encode("utf-8")
        ).hexdigest()
    return identity


def _normalize_manifest_entry(filename, entry):
    if not isinstance(entry, dict):
        return None
    filename = str(entry.get("filename", filename))
    if filename != os.path.basename(filename):
        return None
    parsed = _parse_prompt_checkpoint_name(filename)
    if parsed is None:
        return None
    _, parsed_prefix_length = parsed
    prefix_length = _safe_manifest_int(
        entry.get("prefix_length"),
        parsed_prefix_length,
    )
    if prefix_length != parsed_prefix_length:
        return None
    kind = str(entry.get("kind", "unknown"))
    if kind not in _PROMPT_CHECKPOINT_MANIFEST_KINDS:
        kind = "unknown"
    created_at = _safe_manifest_float(entry.get("created_at"), None)
    if created_at is None:
        created_at = _safe_manifest_float(entry.get("mtime"), time.time())
    normalized = {
        "filename": filename,
        "prefix_length": prefix_length,
        "kind": kind,
        "created_at": created_at,
        "last_hit_at": _safe_manifest_float(entry.get("last_hit_at"), None),
        "hit_count": _safe_manifest_int(entry.get("hit_count"), 0),
        "size_bytes": _safe_manifest_int(entry.get("size_bytes"), 0),
    }
    for key in (
        "checkpoint_namespace",
        "checkpoint_model_hint_hash",
        "checkpoint_tokenizer_hint_hash",
        "checkpoint_glm_mla_kv_quantization",
        "checkpoint_glm_mla_kv_quantization_hash",
        "checkpoint_glm_mla_kv_settings",
        "checkpoint_glm_mla_kv_settings_hash",
    ):
        if key in entry and entry[key] is not None:
            normalized[key] = str(entry[key])
    return normalized


def load_prompt_checkpoint_manifest(return_stats=False):
    stats = {
        "loaded": False,
        "missing": False,
        "malformed": False,
        "entries": 0,
        "malformed_entries": 0,
    }
    try:
        with open(prompt_checkpoint_manifest_file(), "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        stats["missing"] = True
        manifest = _empty_prompt_checkpoint_manifest()
        return (manifest, stats) if return_stats else manifest
    except (OSError, json.JSONDecodeError, TypeError):
        stats["malformed"] = True
        manifest = _empty_prompt_checkpoint_manifest()
        return (manifest, stats) if return_stats else manifest

    if (
        not isinstance(raw, dict)
        or raw.get("version") != PROMPT_CHECKPOINT_MANIFEST_VERSION
        or not isinstance(raw.get("entries"), dict)
    ):
        stats["malformed"] = True
        manifest = _empty_prompt_checkpoint_manifest()
        return (manifest, stats) if return_stats else manifest

    manifest = _empty_prompt_checkpoint_manifest()
    for filename, entry in raw["entries"].items():
        normalized = _normalize_manifest_entry(filename, entry)
        if normalized is None:
            stats["malformed_entries"] += 1
            continue
        manifest["entries"][normalized["filename"]] = normalized
    stats["loaded"] = True
    stats["entries"] = len(manifest["entries"])
    return (manifest, stats) if return_stats else manifest


def save_prompt_checkpoint_manifest(manifest):
    ensure_glm52_local_cache_dirs()
    tmp_file = prompt_checkpoint_manifest_file() + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, sort_keys=True, separators=(",", ":"))
    os.replace(tmp_file, prompt_checkpoint_manifest_file())


def _manifest_entry_from_file(filename, *, kind="unknown", metadata=None):
    filename = os.path.basename(filename)
    parsed = _parse_prompt_checkpoint_name(filename)
    if parsed is None:
        return None
    _, prefix_length = parsed
    file_path = os.path.join(glm52_prompt_checkpoints_dir(), filename)
    try:
        stat = os.stat(file_path)
    except OSError:
        return None
    created_at = getattr(stat, "st_birthtime", stat.st_mtime)
    if kind not in _PROMPT_CHECKPOINT_MANIFEST_KINDS:
        kind = "unknown"
    entry = {
        "filename": filename,
        "prefix_length": prefix_length,
        "kind": kind,
        "created_at": created_at,
        "last_hit_at": None,
        "hit_count": 0,
        "size_bytes": int(stat.st_size),
    }
    entry.update(_manifest_metadata_identity(metadata))
    return entry


def _bootstrap_prompt_checkpoint_manifest():
    manifest = _empty_prompt_checkpoint_manifest()
    stats = {
        "filesystem_scanned": 0,
        "bootstrapped_entries": 0,
    }
    try:
        names = os.listdir(glm52_prompt_checkpoints_dir())
    except OSError:
        return manifest, stats
    for name in names:
        stats["filesystem_scanned"] += 1
        entry = _manifest_entry_from_file(name)
        if entry is None:
            continue
        manifest["entries"][entry["filename"]] = entry
    stats["bootstrapped_entries"] = len(manifest["entries"])
    return manifest, stats


def sync_prompt_checkpoint_manifest(*, bootstrap=False):
    manifest, stats = load_prompt_checkpoint_manifest(return_stats=True)
    stats.update(
        {
            "bootstrap": False,
            "filesystem_scanned": 0,
            "bootstrapped_entries": 0,
            "missing_entries_removed": 0,
            "size_updates": 0,
            "saved": False,
            "save_failed": False,
        }
    )
    changed = stats["malformed_entries"] > 0
    if bootstrap and (stats["missing"] or stats["malformed"]):
        manifest, bootstrap_stats = _bootstrap_prompt_checkpoint_manifest()
        stats["bootstrap"] = True
        stats["filesystem_scanned"] = bootstrap_stats["filesystem_scanned"]
        stats["bootstrapped_entries"] = bootstrap_stats["bootstrapped_entries"]
        changed = len(manifest["entries"]) > 0 or stats["malformed"]

    for filename in list(manifest["entries"]):
        file_path = os.path.join(glm52_prompt_checkpoints_dir(), filename)
        try:
            stat = os.stat(file_path)
        except OSError:
            del manifest["entries"][filename]
            stats["missing_entries_removed"] += 1
            changed = True
            continue
        size_bytes = int(stat.st_size)
        if manifest["entries"][filename].get("size_bytes") != size_bytes:
            manifest["entries"][filename]["size_bytes"] = size_bytes
            stats["size_updates"] += 1
            changed = True

    stats["entries"] = len(manifest["entries"])
    if changed:
        try:
            save_prompt_checkpoint_manifest(manifest)
            stats["saved"] = True
        except OSError:
            stats["save_failed"] = True
    return manifest, stats


def update_prompt_checkpoint_manifest(
    file_name,
    *,
    prefix_length,
    kind,
    metadata=None,
    hit=False,
):
    manifest, stats = load_prompt_checkpoint_manifest(return_stats=True)
    if stats["malformed"]:
        manifest = _empty_prompt_checkpoint_manifest()
    filename = os.path.basename(file_name)
    entry = _manifest_entry_from_file(filename, kind=kind, metadata=metadata)
    if entry is None:
        raise OSError("checkpoint file is missing or malformed")
    if int(prefix_length) != entry["prefix_length"]:
        raise OSError("checkpoint prefix length does not match manifest entry")

    now = time.time()
    old_entry = manifest["entries"].get(filename)
    if old_entry is not None:
        entry["created_at"] = old_entry.get("created_at", entry["created_at"])
        entry["hit_count"] = _safe_manifest_int(old_entry.get("hit_count"), 0)
        entry["last_hit_at"] = old_entry.get("last_hit_at")
    if hit:
        entry["hit_count"] += 1
        entry["last_hit_at"] = now
    manifest["entries"][filename] = entry
    save_prompt_checkpoint_manifest(manifest)
    return {
        "filename": filename,
        "entries": len(manifest["entries"]),
        "kind": entry["kind"],
        "size_bytes": entry["size_bytes"],
        "hit_count": entry["hit_count"],
        "manifest_was_malformed": stats["malformed"],
    }


def prompt_checkpoint_budget_from_env():
    return {
        "max_files": _env_int(
            PROMPT_CHECKPOINT_MAX_FILES_ENV,
            DEFAULT_PROMPT_CHECKPOINT_MAX_FILES,
        ),
        "max_bytes": _env_int(
            PROMPT_CHECKPOINT_MAX_BYTES_ENV,
            DEFAULT_PROMPT_CHECKPOINT_MAX_BYTES,
        ),
    }


def _prompt_checkpoint_prune_key(entry):
    kind_rank = {
        "exact": 0,
        "unknown": 1,
        "prefix": 2,
        "frontier": 3,
    }.get(entry.get("kind"), 1)
    hit_count = _safe_manifest_int(entry.get("hit_count"), 0)
    used_at = _safe_manifest_float(entry.get("last_hit_at"), None)
    if used_at is None:
        used_at = _safe_manifest_float(entry.get("created_at"), 0)
    prefix_length = _safe_manifest_int(entry.get("prefix_length"), 0)
    return (kind_rank, hit_count, used_at, prefix_length)


def prune_prompt_checkpoints(
    *,
    max_files=None,
    max_bytes=None,
    protected_files=None,
):
    if max_files is None or max_bytes is None:
        budget = prompt_checkpoint_budget_from_env()
        if max_files is None:
            max_files = budget["max_files"]
        if max_bytes is None:
            max_bytes = budget["max_bytes"]
    max_files = _safe_manifest_int(max_files, 0)
    max_bytes = _safe_manifest_int(max_bytes, 0)
    protected = {os.path.basename(f) for f in (protected_files or [])}

    manifest, sync_stats = sync_prompt_checkpoint_manifest(bootstrap=True)
    entries = manifest["entries"]
    total_bytes = sum(_safe_manifest_int(e.get("size_bytes"), 0) for e in entries.values())
    removed = []
    changed = False

    def over_budget():
        return (
            (max_files > 0 and len(entries) > max_files)
            or (max_bytes > 0 and total_bytes > max_bytes)
        )

    while over_budget():
        victims = [
            entry
            for entry in entries.values()
            if entry["filename"] not in protected
        ]
        if not victims:
            break
        victim = min(victims, key=_prompt_checkpoint_prune_key)
        filename = victim["filename"]
        file_path = os.path.join(glm52_prompt_checkpoints_dir(), filename)
        try:
            os.remove(file_path)
            removed_status = "removed"
        except FileNotFoundError:
            removed_status = "missing"
        except OSError:
            break
        removed.append(
            {
                "filename": filename,
                "kind": victim.get("kind", "unknown"),
                "prefix_length": victim.get("prefix_length", 0),
                "size_bytes": victim.get("size_bytes", 0),
                "status": removed_status,
            }
        )
        total_bytes -= _safe_manifest_int(victim.get("size_bytes"), 0)
        del entries[filename]
        changed = True

    if changed or sync_stats.get("saved"):
        save_prompt_checkpoint_manifest(manifest)

    return {
        "max_files": max_files,
        "max_bytes": max_bytes,
        "total_files": len(entries),
        "total_bytes": total_bytes,
        "removed": removed,
        "protected_files": sorted(protected),
        "sync": sync_stats,
    }


def find_prompt_checkpoint_prefix(
    prefix_tokens,
    *,
    min_prefix_length=2,
    return_stats=False,
):
    """
    Find checkpoint files whose token prefix is a prefix of ``prefix_tokens``.

    This is the token-prefix analogue of ds4.c's rendered-byte prefix lookup.
    It intentionally stays in the trusted single-model GLM-5.2 cache root and
    returns longest candidates first so callers can safely fall back to shorter
    checkpoints when a longer file is malformed or rejected.
    """
    tokens = _token_list(prefix_tokens)
    stats = {
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
    }
    manifest, manifest_stats = sync_prompt_checkpoint_manifest(bootstrap=True)
    stats["manifest_entries"] = len(manifest["entries"])
    stats["manifest_loaded"] = manifest_stats.get("loaded", False)
    stats["manifest_missing"] = manifest_stats.get("missing", False)
    stats["manifest_malformed"] = manifest_stats.get("malformed", False)
    stats["manifest_bootstrap"] = manifest_stats.get("bootstrap", False)
    stats["manifest_missing_entries_removed"] = manifest_stats.get(
        "missing_entries_removed",
        0,
    )

    by_length = {}
    for name in manifest["entries"]:
        stats["files_scanned"] += 1
        parsed = _parse_prompt_checkpoint_name(name)
        if parsed is None:
            continue
        hash_part, prefix_length = parsed
        if (
            prefix_length < min_prefix_length
            or prefix_length > len(tokens)
        ):
            continue
        stats["candidate_files_scanned"] += 1
        by_length.setdefault(prefix_length, {})[hash_part] = name

    candidates = []
    for prefix_length in sorted(by_length, reverse=True):
        stats["candidate_lengths_scanned"] += 1
        prefix = tokens[:prefix_length]
        prefix_hash = prompt_prefix_hash(prefix)
        stats["prefix_hashes_computed"] += 1
        name = by_length[prefix_length].get(prefix_hash)
        if name is None:
            continue
        candidates.append(
            (
                prefix_length,
                prefix,
                os.path.join(glm52_prompt_checkpoints_dir(), name),
            )
        )
    stats["matched_candidates"] = len(candidates)
    return (candidates, stats) if return_stats else candidates


def _model_config_dict(model):
    config = getattr(model, "args", None)
    if config is None:
        config = getattr(model, "config", None)
    if config is None and hasattr(model, "model"):
        config = getattr(model.model, "args", None)
    if config is None:
        return {}
    return _normalize_for_json(config)


def _model_metadata(model, model_id):
    config = _model_config_dict(model)
    model_type = config.get("model_type") if isinstance(config, dict) else None
    return {
        "id": model_id,
        "class": f"{type(model).__module__}.{type(model).__qualname__}"
        if model is not None
        else None,
        "model_type": model_type,
        "config": config,
    }


def _tokenizer_metadata(tokenizer, tokenizer_config, tokenizer_id):
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    chat_template = getattr(inner, "chat_template", None)
    eos_token_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_token_ids is not None:
        eos_token_ids = sorted(eos_token_ids)
    return {
        "id": tokenizer_id,
        "class": f"{type(inner).__module__}.{type(inner).__qualname__}"
        if inner is not None
        else None,
        "name_or_path": getattr(inner, "name_or_path", None),
        "vocab_size": getattr(inner, "vocab_size", None),
        "eos_token_ids": eos_token_ids,
        "chat_template_hash": hashlib.sha256(
            (chat_template or "").encode("utf-8")
        ).hexdigest(),
        "tokenizer_config": tokenizer_config or {},
    }


def _glm_dsa_metadata(model):
    config = _model_config_dict(model)
    model_type = config.get("model_type") if isinstance(config, dict) else None
    if model_type != "glm_moe_dsa" and "indexer_types" not in config:
        return {}

    layers = getattr(model, "layers", [])
    layer_cache_widths = []
    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        layer_cache_widths.append(
            1 if getattr(self_attn, "skip_topk", False) else 2
        )

    return {
        "model_type": model_type,
        "indexer_types": config.get("indexer_types"),
        "index_topk": config.get("index_topk"),
        "index_head_dim": config.get("index_head_dim"),
        "index_n_heads": config.get("index_n_heads"),
        "index_topk_pattern": config.get("index_topk_pattern"),
        "index_topk_freq": config.get("index_topk_freq"),
        "index_skip_topk_offset": config.get("index_skip_topk_offset"),
        "layer_cache_widths": layer_cache_widths,
    }


def _state_signature(value):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "kind": "array",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (list, tuple)):
        return [_state_signature(v) for v in value]
    if isinstance(value, dict):
        return {
            str(k): _state_signature(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {
        "kind": "object",
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
    }


def _cache_quantization_signature(cache):
    if hasattr(cache, "caches"):
        return [_cache_quantization_signature(c) for c in cache.caches]

    cache_type = type(cache).__name__
    if cache_type in ("QuantizedGlmMlaKVCache", "BatchQuantizedGlmMlaKVCache"):
        return {
            "scheme": "glm_mla_latent_int8",
            "group_size": cache.group_size,
            "bits": cache.bits,
        }
    if cache_type == "QuantizedKVCache":
        return {
            "scheme": "kv",
            "group_size": cache.group_size,
            "bits": cache.bits,
        }
    if cache_type in ("GlmMlaKVCache", "BatchGlmMlaKVCache"):
        return {"scheme": "glm_mla_latent_fp"}
    return None


def glm_mla_kv_quantization_metadata(cache: List[Any]):
    metadata = {
        "glm_mla_latent_fp_layers": 0,
        "glm_mla_latent_int8_layers": 0,
        "glm_mla_latent_int8_group_sizes": [],
        "glm_mla_latent_int8_bits": [],
    }
    group_sizes = set()
    bits = set()

    def visit(c):
        cache_type = type(c).__name__
        if cache_type in ("GlmMlaKVCache", "BatchGlmMlaKVCache"):
            metadata["glm_mla_latent_fp_layers"] += 1
        elif cache_type in ("QuantizedGlmMlaKVCache", "BatchQuantizedGlmMlaKVCache"):
            metadata["glm_mla_latent_int8_layers"] += 1
            group_sizes.add(c.group_size)
            bits.add(c.bits)
        elif hasattr(c, "caches"):
            for subcache in c.caches:
                visit(subcache)

    for c in cache:
        visit(c)

    metadata["glm_mla_latent_int8_group_sizes"] = sorted(group_sizes)
    metadata["glm_mla_latent_int8_bits"] = sorted(bits)
    return metadata


def expected_glm_mla_kv_quantization_metadata(
    model: Optional[nn.Module],
    *,
    cache_token_length: int,
    kv_bits: Optional[int],
    kv_group_size: int,
    quantized_kv_start: int,
):
    glm_dsa = _glm_dsa_metadata(model)
    layer_count = len(glm_dsa.get("layer_cache_widths", []))
    if layer_count == 0:
        return None
    if kv_bits == 8 and cache_token_length >= quantized_kv_start:
        return {
            "glm_mla_latent_fp_layers": 0,
            "glm_mla_latent_int8_layers": layer_count,
            "glm_mla_latent_int8_group_sizes": [kv_group_size],
            "glm_mla_latent_int8_bits": [8],
        }
    return {
        "glm_mla_latent_fp_layers": layer_count,
        "glm_mla_latent_int8_layers": 0,
        "glm_mla_latent_int8_group_sizes": [],
        "glm_mla_latent_int8_bits": [],
    }


def model_has_glm_mla_kv_cache(model: Optional[nn.Module]):
    return len(_glm_dsa_metadata(model).get("layer_cache_widths", [])) > 0


def glm_mla_kv_settings_metadata(
    *,
    kv_bits: Optional[int] = None,
    kv_group_size: Optional[int] = None,
    quantized_kv_start: Optional[int] = None,
):
    if kv_bits is None:
        return {
            "kv_bits": None,
            "kv_group_size": None,
            "quantized_kv_start": None,
        }
    return {
        "kv_bits": int(kv_bits),
        "kv_group_size": int(kv_group_size),
        "quantized_kv_start": int(quantized_kv_start),
    }


def expected_glm_mla_kv_settings_metadata(
    model: Optional[nn.Module],
    *,
    kv_bits: Optional[int] = None,
    kv_group_size: Optional[int] = None,
    quantized_kv_start: Optional[int] = None,
):
    if not model_has_glm_mla_kv_cache(model):
        return None
    return glm_mla_kv_settings_metadata(
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
    )


def prompt_cache_signature(cache: List[Any]):
    signature = []
    for c in cache:
        entry = {
            "class": type(c).__name__,
            "meta_state": _normalize_for_json(c.meta_state),
            "quantization": _cache_quantization_signature(c),
            "state": _state_signature(c.state),
        }
        try:
            entry["size"] = c.size()
        except Exception:
            entry["size"] = None
        signature.append(entry)
    return signature


def build_prompt_cache_checkpoint_metadata(
    cache: List[Any],
    *,
    prefix_tokens,
    checkpoint_namespace: str = DEFAULT_PROMPT_CHECKPOINT_NAMESPACE,
    model_id: str = DEFAULT_PROMPT_CHECKPOINT_MODEL_ID,
    tokenizer_id: str = DEFAULT_PROMPT_CHECKPOINT_TOKENIZER_ID,
    model: Optional[nn.Module] = None,
    tokenizer: Optional[Any] = None,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    kv_bits: Optional[int] = None,
    kv_group_size: Optional[int] = None,
    quantized_kv_start: Optional[int] = None,
    metadata: Optional[Dict[str, str]] = None,
):
    """
    Build metadata for trusted local prompt checkpoints.

    This checkpoint path is intended for trusted single-model local operation.
    It does not prove full model artifact identity. Users must clear checkpoint
    files or use a new checkpoint_namespace when changing model weights,
    quantization, tokenizer, adapters, or GLM-5.2 implementation details.
    """
    prefix = _token_list(prefix_tokens)
    model_info = _model_metadata(model, model_id)
    tokenizer_info = _tokenizer_metadata(tokenizer, tokenizer_config, tokenizer_id)
    glm_dsa_info = _glm_dsa_metadata(model)
    glm_mla_kv_quantization_info = glm_mla_kv_quantization_metadata(cache)
    glm_mla_kv_settings_info = glm_mla_kv_settings_metadata(
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
    )
    cache_signature = prompt_cache_signature(cache)

    checkpoint_metadata = {
        "checkpoint_format": PROMPT_CACHE_CHECKPOINT_FORMAT,
        "checkpoint_version": PROMPT_CACHE_CHECKPOINT_VERSION,
        "checkpoint_namespace": checkpoint_namespace,
        "checkpoint_model_hint": model_id,
        "checkpoint_model_hint_hash": _json_hash(model_info),
        "checkpoint_tokenizer_hint": tokenizer_id,
        "checkpoint_tokenizer_hint_hash": _json_hash(tokenizer_info),
        "checkpoint_prefix_hash": _json_hash(prefix),
        "checkpoint_prefix_length": str(len(prefix)),
        "checkpoint_cache_signature_hash": _json_hash(cache_signature),
        "checkpoint_cache_signature": _json_dumps(cache_signature),
        "checkpoint_model_hint_metadata": _json_dumps(model_info),
        "checkpoint_tokenizer_hint_metadata": _json_dumps(tokenizer_info),
        "checkpoint_glm_dsa_metadata": _json_dumps(glm_dsa_info),
        "checkpoint_glm_mla_kv_quantization": _json_dumps(
            glm_mla_kv_quantization_info
        ),
        "checkpoint_glm_mla_kv_settings": _json_dumps(
            glm_mla_kv_settings_info
        ),
        "model": model_id,
        "tokenizer_config": json.dumps(tokenizer_config or {}),
    }
    if metadata:
        extra_metadata = {str(k): str(v) for k, v in metadata.items()}
        reserved = set(checkpoint_metadata)
        conflicts = reserved.intersection(extra_metadata)
        if conflicts:
            keys = ", ".join(sorted(conflicts))
            raise ValueError(f"metadata overrides checkpoint keys: {keys}")
        checkpoint_metadata.update(extra_metadata)
    return checkpoint_metadata


def save_prompt_checkpoint(
    file_name: str,
    cache: List[Any],
    *,
    prefix_tokens,
    checkpoint_namespace: str = DEFAULT_PROMPT_CHECKPOINT_NAMESPACE,
    model_id: str = DEFAULT_PROMPT_CHECKPOINT_MODEL_ID,
    tokenizer_id: str = DEFAULT_PROMPT_CHECKPOINT_TOKENIZER_ID,
    model: Optional[nn.Module] = None,
    tokenizer: Optional[Any] = None,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    kv_bits: Optional[int] = None,
    kv_group_size: Optional[int] = None,
    quantized_kv_start: Optional[int] = None,
    metadata: Optional[Dict[str, str]] = None,
):
    checkpoint_metadata = build_prompt_cache_checkpoint_metadata(
        cache,
        prefix_tokens=prefix_tokens,
        checkpoint_namespace=checkpoint_namespace,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        model=model,
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
        metadata=metadata,
    )
    save_prompt_cache(file_name, cache, checkpoint_metadata)
    return checkpoint_metadata


def _require_metadata(metadata, key):
    value = metadata.get(key)
    if value is None:
        raise PromptCacheCheckpointError(f"checkpoint is missing {key}")
    return value


def _require_checkpoint_metadata(metadata):
    for key in _CHECKPOINT_REQUIRED_METADATA_KEYS:
        _require_metadata(metadata, key)


def _metadata_json(metadata, key):
    try:
        return json.loads(_require_metadata(metadata, key))
    except (TypeError, json.JSONDecodeError) as exc:
        raise PromptCacheCheckpointError(f"checkpoint has malformed {key}") from exc


def _metadata_int(metadata, key):
    try:
        return int(_require_metadata(metadata, key))
    except (TypeError, ValueError) as exc:
        raise PromptCacheCheckpointError(f"checkpoint has malformed {key}") from exc


def _validate_checkpoint_metadata(
    cache: List[Any],
    metadata: Dict[str, str],
    *,
    prefix_tokens,
    checkpoint_namespace: str,
    model_id: str,
    tokenizer_id: str,
    model: Optional[nn.Module],
    tokenizer: Optional[Any],
    tokenizer_config: Optional[Dict[str, Any]],
    expected_glm_mla_kv_quantization: Optional[Dict[str, Any]],
    expected_glm_mla_kv_settings: Optional[Dict[str, Any]],
):
    _require_checkpoint_metadata(metadata)
    if _require_metadata(metadata, "checkpoint_format") != PROMPT_CACHE_CHECKPOINT_FORMAT:
        raise PromptCacheCheckpointError("unsupported checkpoint format")
    if (
        _require_metadata(metadata, "checkpoint_version")
        != PROMPT_CACHE_CHECKPOINT_VERSION
    ):
        raise PromptCacheCheckpointError("unsupported checkpoint version")
    if _require_metadata(metadata, "checkpoint_namespace") != checkpoint_namespace:
        raise PromptCacheCheckpointError("checkpoint namespace does not match")
    if _require_metadata(metadata, "checkpoint_model_hint") != model_id:
        raise PromptCacheCheckpointError("checkpoint model hint does not match")
    if _require_metadata(metadata, "checkpoint_tokenizer_hint") != tokenizer_id:
        raise PromptCacheCheckpointError("checkpoint tokenizer hint does not match")

    prefix = _token_list(prefix_tokens)
    if _require_metadata(metadata, "checkpoint_prefix_hash") != _json_hash(prefix):
        raise PromptCacheCheckpointError("checkpoint prefix hash does not match")
    if _metadata_int(metadata, "checkpoint_prefix_length") != len(prefix):
        raise PromptCacheCheckpointError("checkpoint prefix length does not match")

    model_info = _model_metadata(model, model_id)
    saved_model_info = _metadata_json(metadata, "checkpoint_model_hint_metadata")
    if (
        saved_model_info != model_info
        or _require_metadata(metadata, "checkpoint_model_hint_hash")
        != _json_hash(model_info)
    ):
        raise PromptCacheCheckpointError("checkpoint model hint metadata does not match")

    tokenizer_info = _tokenizer_metadata(tokenizer, tokenizer_config, tokenizer_id)
    saved_tokenizer_info = _metadata_json(
        metadata, "checkpoint_tokenizer_hint_metadata"
    )
    if (
        saved_tokenizer_info != tokenizer_info
        or _require_metadata(metadata, "checkpoint_tokenizer_hint_hash")
        != _json_hash(tokenizer_info)
    ):
        raise PromptCacheCheckpointError(
            "checkpoint tokenizer hint metadata does not match"
        )

    cache_signature = prompt_cache_signature(cache)
    saved_cache_signature = _metadata_json(metadata, "checkpoint_cache_signature")
    if (
        saved_cache_signature != cache_signature
        or _require_metadata(metadata, "checkpoint_cache_signature_hash")
        != _json_hash(cache_signature)
    ):
        raise PromptCacheCheckpointError("checkpoint cache signature does not match")

    saved_glm_dsa = _metadata_json(metadata, "checkpoint_glm_dsa_metadata")
    current_glm_dsa = _glm_dsa_metadata(model)
    if saved_glm_dsa != current_glm_dsa:
        raise PromptCacheCheckpointError("checkpoint GLM DSA metadata does not match")

    saved_glm_mla_kv = _metadata_json(
        metadata, "checkpoint_glm_mla_kv_quantization"
    )
    current_glm_mla_kv = glm_mla_kv_quantization_metadata(cache)
    if saved_glm_mla_kv != current_glm_mla_kv:
        raise PromptCacheCheckpointError(
            "checkpoint GLM MLA KV quantization metadata does not match"
        )
    if (
        expected_glm_mla_kv_quantization is not None
        and saved_glm_mla_kv != expected_glm_mla_kv_quantization
    ):
        raise PromptCacheCheckpointError(
            "checkpoint GLM MLA KV quantization setting does not match"
        )

    saved_glm_mla_kv_settings = _metadata_json(
        metadata, "checkpoint_glm_mla_kv_settings"
    )
    if (
        expected_glm_mla_kv_settings is not None
        and saved_glm_mla_kv_settings != expected_glm_mla_kv_settings
    ):
        raise PromptCacheCheckpointError(
            "checkpoint GLM MLA KV settings do not match"
        )


def load_prompt_checkpoint(
    file_name: str,
    *,
    prefix_tokens,
    checkpoint_namespace: str = DEFAULT_PROMPT_CHECKPOINT_NAMESPACE,
    model_id: str = DEFAULT_PROMPT_CHECKPOINT_MODEL_ID,
    tokenizer_id: str = DEFAULT_PROMPT_CHECKPOINT_TOKENIZER_ID,
    model: Optional[nn.Module] = None,
    tokenizer: Optional[Any] = None,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    expected_glm_mla_kv_quantization: Optional[Dict[str, Any]] = None,
    expected_glm_mla_kv_settings: Optional[Dict[str, Any]] = None,
    return_metadata: bool = False,
):
    try:
        cache, metadata = load_prompt_cache(file_name, return_metadata=True)
    except Exception as exc:
        raise PromptCacheCheckpointError(f"failed to load prompt checkpoint: {exc}") from exc
    _validate_checkpoint_metadata(
        cache,
        metadata,
        prefix_tokens=prefix_tokens,
        checkpoint_namespace=checkpoint_namespace,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        model=model,
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        expected_glm_mla_kv_quantization=expected_glm_mla_kv_quantization,
        expected_glm_mla_kv_settings=expected_glm_mla_kv_settings,
    )
    if return_metadata:
        return cache, metadata
    return cache


def invalidate_prompt_checkpoint(file_name: str):
    """
    Clear a local prompt checkpoint by deleting its safetensors file.

    Using a new checkpoint_namespace is the non-destructive invalidation path
    when users want to keep older checkpoint files on disk.
    """
    try:
        import os

        os.remove(file_name)
    except FileNotFoundError:
        pass


def can_trim_prompt_cache(cache: List[Any]) -> bool:
    """
    Check if model's cache can be trimmed.
    """
    return all(c.is_trimmable() for c in cache)


def trim_prompt_cache(cache: List[Any], num_tokens: int) -> List[Any]:
    """
    Trim the model's cache by the given number of tokens.

    This function will trim the cache if possible (in-place) and return the
    number of tokens that were trimmed.

    Args:
        cache (List[Any]): The model's cache.
        num_tokens (int): The number of tokens to trim.

    Returns:
        (int): The number of tokens that were trimmed.
    """
    if not can_trim_prompt_cache(cache) or len(cache) == 0:
        return 0
    return [c.trim(num_tokens) for c in cache][0]


def create_attention_mask(
    N: int, offset: int, return_array: bool, window_size: Optional[int]
):
    if window_size is not None:
        return create_causal_mask(N, offset, window_size=window_size)
    elif N == 1:
        return None
    elif return_array:
        return create_causal_mask(N, offset, window_size=window_size)
    else:
        return "causal"


class _BaseCache:
    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no state but a state was set.")

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no meta_state but a meta_state was set.")

    def is_trimmable(self):
        return False

    def size(self):
        """
        Return the size (i.e. sequence length) of the cache.

        Not every cache is required to implement this, in which case the size
        will always be 0 (though the cache may not be empty).
        """
        return 0

    @property
    def nbytes(self):
        """Return the size of this cache in bytes"""
        raise NotImplementedError("Cache sub-class must implement nbytes")

    def empty(self):
        """
        Return if the cache is empty or not.
        """
        raise NotImplementedError("Cache sub-class must implement this.")

    @classmethod
    def from_state(cls, state, meta_state):
        # Create an instance of cls without calling __init__
        obj = cls.__new__(cls)
        obj.state = state
        obj.meta_state = meta_state
        return obj


class ConcatenateKVCache(_BaseCache):
    """ConcatenateKVCache the simplest KV cache implementation.

    Can be used as a mock KV cache or when large blocks are being processed at
    a time in which case KVCache isn't necessarily faster. Consider using the
    KVCache with a larger step size before using this cache.
    """

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=-2)
            self.values = mx.concatenate([self.values, values], axis=-2)
        self.offset = self.keys.shape[-2]

        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[-2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class QuantizedKVCache(_BaseCache):
    step = 256

    def __init__(self, group_size: int = 64, bits: int = 8):
        self.keys = None
        self.values = None
        self.offset = 0
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )

                self.keys, self.values = tree_map(
                    expand_quant, (self.keys, self.values)
                )
            else:
                self.keys, self.values = init_quant(k_head_dim), init_quant(v_head_dim)

        self.offset += num_steps

        keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.offset, :] = keys[i]
            self.values[i][..., prev : self.offset, :] = values[i]

        return tree_map(lambda x: x[..., : self.offset, :], (self.keys, self.values))

    @property
    def state(self):
        if self.offset == self.keys[0].shape[2]:
            return self.keys, self.values
        else:
            return tree_map(
                lambda x: x[..., : self.offset, :], (self.keys, self.values)
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.group_size, self.bits = map(int, v)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)


class QuantizedGlmMlaKVCache(_BaseCache):
    """GLM-5.2 MLA cache with only the latent KV tensor stored as int8.

    The associated RoPE side channel remains in its original floating dtype.
    This is intentionally narrow and is used only by ``glm_moe_dsa`` cache
    lists; DSA indexer caches remain regular ``KVCache`` instances.
    """

    step = 256
    quantize_with_cache_list = True

    def __init__(self, group_size: int = 64, bits: int = 8):
        if bits != 8:
            raise ValueError("GLM MLA latent KV cache only supports int8.")
        self.keys = None
        self.values = None
        self.offset = 0
        self.group_size = group_size
        self.bits = bits

    def _check_key_dim(self, key_dim: int):
        if key_dim % self.group_size != 0:
            raise ValueError(
                "GLM MLA latent KV cache dimension must be divisible by "
                f"kv_group_size ({key_dim} vs {self.group_size})."
            )

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        self._check_key_dim(k_head_dim)
        prev = self.offset

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            key_shape = (B, n_kv_heads, new_steps)
            value_shape = (B, n_kv_heads, new_steps, v_head_dim)

            def init_quant(dim):
                return (
                    mx.zeros((*key_shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*key_shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*key_shape, dim // self.group_size), dtype=keys.dtype),
                )

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = tree_map(lambda x: x[..., :prev, :], self.keys)
                    self.values = self.values[..., :prev, :]

                def expand_quant(x):
                    new_x = mx.zeros((*key_shape, x.shape[-1]), dtype=x.dtype)
                    return mx.concatenate([x, new_x], axis=-2)

                self.keys = tree_map(expand_quant, self.keys)
                self.values = mx.concatenate(
                    [self.values, mx.zeros(value_shape, values.dtype)], axis=-2
                )
            else:
                self.keys = init_quant(k_head_dim)
                self.values = mx.zeros(value_shape, values.dtype)

        self.offset += num_steps

        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.offset, :] = q_keys[i]
        self.values[..., prev : self.offset, :] = values

        return self.state

    def dequantize_keys(self, keys=None):
        keys = self.keys if keys is None else keys
        return mx.dequantize(*keys, group_size=self.group_size, bits=self.bits)

    @property
    def state(self):
        if self.offset == self.keys[0].shape[2]:
            return self.keys, self.values
        return (
            tree_map(lambda x: x[..., : self.offset, :], self.keys),
            self.values[..., : self.offset, :],
        )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys[0].shape[2]

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.group_size, self.bits = map(int, v)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def size(self):
        return self.offset

    def to_quantized(self, group_size: int = 64, bits: int = 8):
        if bits != 8:
            raise ValueError("GLM MLA latent KV cache only supports int8.")
        if self.group_size == group_size:
            return self
        quant_cache = QuantizedGlmMlaKVCache(group_size=group_size, bits=bits)
        quant_cache.offset = self.offset
        if self.keys is not None:
            latent = self.dequantize_keys()
            quant_cache.keys = mx.quantize(
                latent, group_size=group_size, bits=bits
            )
            quant_cache.values = self.values[..., : self.offset, :]
        return quant_cache

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, self.keys, 0) + self.values.nbytes

    @classmethod
    def merge(cls, caches):
        return BatchQuantizedGlmMlaKVCache.merge(caches)


class KVCache(_BaseCache):
    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def size(self):
        return self.offset

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        quant_cache = QuantizedKVCache(group_size=group_size, bits=bits)
        quant_cache.offset = self.offset
        if self.keys is not None:
            quant_cache.keys = mx.quantize(self.keys, group_size=group_size, bits=bits)
            quant_cache.values = mx.quantize(
                self.values, group_size=group_size, bits=bits
            )
        return quant_cache

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @classmethod
    def merge(_, caches):
        return BatchKVCache.merge(caches)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class GlmMlaKVCache(KVCache):
    """Unquantized GLM-5.2 MLA cache that can opt into latent-only int8."""

    quantize_with_cache_list = True

    def to_quantized(
        self, group_size: int = 64, bits: int = 8
    ) -> QuantizedGlmMlaKVCache:
        quant_cache = QuantizedGlmMlaKVCache(group_size=group_size, bits=bits)
        quant_cache.offset = self.offset
        if self.keys is not None:
            quant_cache._check_key_dim(self.keys.shape[-1])
            quant_cache.keys = mx.quantize(
                self.keys[..., : self.offset, :],
                group_size=group_size,
                bits=bits,
            )
            quant_cache.values = self.values[..., : self.offset, :]
        return quant_cache

    @classmethod
    def merge(_, caches):
        return BatchGlmMlaKVCache.merge(caches)


class RotatingKVCache(_BaseCache):
    step = 256

    def __init__(self, max_size, keep=0):
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self._idx = 0

    def _trim(self, trim_size, v, append=None):
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, v):
        """
        Rearrange the cache into temporal order, slicing off the end if unused.
        """
        if self._idx == v.shape[2]:
            return v
        elif self._idx < self.offset:
            return mx.concatenate(
                [
                    v[..., : self.keep, :],
                    v[..., self._idx :, :],
                    v[..., self.keep : self._idx, :],
                ],
                axis=2,
            )
        else:
            return v[..., : self._idx, :]

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self._idx = self.keys.shape[2]

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset += keys.shape[2]
        self._idx = self.keys.shape[2]
        return self.keys, self.values

    def _update_in_place(self, keys, values):
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self.offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        # Assign
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self.offset += S
        self._idx += S

        # If the buffer is not full, slice off the end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def size(self):
        return min(self.offset, self.max_size)

    @property
    def state(self):
        if self.offset < self.keys.shape[2]:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        else:
            return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.keep, self.max_size, self.offset, self._idx)))

    @meta_state.setter
    def meta_state(self, v):
        self.keep, self.max_size, self.offset, self._idx = map(
            int,
            v,
        )

    def is_trimmable(self):
        return self.offset < self.max_size

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self._idx -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        raise NotImplementedError("RotatingKVCache Quantization NYI")

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        if N > 1:
            window_size = window_size or self.max_size
            offset = min(self.max_size - 1, self.offset)
            if offset + N > window_size or return_array:
                return create_causal_mask(N, offset, window_size=window_size)
            else:
                return "causal"
        else:
            if window_size is None:
                return None
            # May need a mask for when window_size < max_size
            if self.offset >= window_size and self.max_size > window_size:
                idx = self._idx
                if idx >= self.max_size:
                    idx = 0
                if self.offset < self.max_size:
                    mask_size = self.offset + 1
                else:
                    mask_size = self.max_size
                mask = mx.arange(mask_size) >= (mask_size - window_size)
                mask = mx.roll(mask, shift=idx + 1)
                return mask

    @classmethod
    def merge(_, caches):
        return BatchRotatingKVCache.merge(caches)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class ArraysCache(_BaseCache):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.left_padding = None
        instance.lengths = None
        return instance

    def __init__(self, size, left_padding: Optional[List[int]] = None):
        self.cache = [None] * size
        if left_padding:
            self.left_padding = mx.array(left_padding)

    @property
    def batch_size(self):
        for c in self.cache:
            if c is not None:
                return c.shape[0]
        if self.left_padding is not None:
            return self.left_padding.size
        elif self.lengths is not None:
            return self.lengths.size
        else:
            return 1

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    def __getitem__(self, idx):
        return self.cache[idx]

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, v):
        self.cache = v

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        self.cache = [c[batch_indices] if c is not None else None for c in self.cache]
        if self.left_padding is not None:
            self.left_padding = self.left_padding[batch_indices]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """

        a_batch = self.batch_size
        b_batch = other.batch_size

        def cat(a, b):
            shape = dtype = None
            if a is not None:
                shape = a.shape
                dtype = a.dtype
            if b is not None:
                shape = b.shape
                dtype = b.dtype

            if shape is None:
                return None

            if a is None:
                a = mx.zeros((a_batch,) + shape[1:], dtype=dtype)
            if b is None:
                b = mx.zeros((b_batch,) + shape[1:], dtype=dtype)

            return mx.concatenate([a, b])

        self.cache = [cat(c, o) for c, o in zip(self.cache, other.cache)]
        self.left_padding = cat(self.left_padding, other.left_padding)
        self.lengths = cat(self.lengths, other.lengths)

    def extract(self, idx):
        cache = ArraysCache(len(self.cache))
        cache.cache = [c[idx : idx + 1] for c in self.cache]
        return cache

    def prepare(self, lengths=None, **kwargs):
        self.lengths = mx.array(lengths)

    def finalize(self):
        self.lengths = None
        self.left_padding = None

    def advance(self, N):
        if self.lengths is not None:
            self.lengths -= N
        if self.left_padding is not None:
            self.left_padding -= N

    def make_mask(self, N: int):
        if self.left_padding is not None:
            pos = mx.arange(N)
            return pos >= self.left_padding[:, None]
        elif self.lengths is not None:
            pos = mx.arange(N)
            return pos < self.lengths[:, None]
        else:
            return None

    @classmethod
    def merge(cls, caches):
        n_state = len(caches[0].cache)
        B = len(caches)
        cache = cls(n_state)

        # All caches are empty so return early
        if all(c.empty() for c in caches):
            cache.left_padding = mx.array([0] * B)
            return cache

        for e in range(n_state):
            c_init = next(iter(c[e] for c in caches if c[e] is not None))
            shape = list(c_init.shape)
            shape[0] = B
            cache[e] = mx.zeros(shape, c_init.dtype)
            for i in range(B):
                if caches[i][e] is None:
                    continue
                cache[e][i : i + 1] = caches[i][e]
        return cache

    def empty(self):
        return self.cache[0] is None

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.cache if c is not None)


class ChunkedKVCache(_BaseCache):
    step = 256

    def __init__(self, chunk_size):
        self.keys = None
        self.values = None
        self.offset = 0
        self.chunk_size = chunk_size
        self.start_position = 0

    def maybe_trim_front(self):
        # Maintain the cache below the chunk size
        if self.keys is not None and self.keys.shape[2] >= self.chunk_size:
            self.start_position += self.keys.shape[2] - self.chunk_size
            self.keys = self.keys[..., -self.chunk_size :, :]
            self.values = self.values[..., -self.chunk_size :, :]

    def update_and_fetch(self, keys, values):
        prev = self.offset - self.start_position
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        end = self.offset - self.start_position
        self.keys[..., prev:end, :] = keys
        self.values[..., prev:end, :] = values
        return self.keys[..., :end, :], self.values[..., :end, :]

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset - self.start_position, n)
        self.offset -= n
        return n

    @property
    def meta_state(self):
        return tuple(map(str, (self.chunk_size, self.start_position)))

    @meta_state.setter
    def meta_state(self, v):
        self.chunk_size, self.start_position = map(int, v)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class CacheList(_BaseCache):
    def __init__(self, *caches):
        self.caches = caches

    def __getitem__(self, idx):
        return self.caches[idx]

    @property
    def offset(self):
        offsets = []
        for c in self.caches:
            offset = getattr(c, "offset", 0)
            if hasattr(offset, "shape") and offset.shape != ():
                offset = c.size()
            offsets.append(offset)
        return max(offsets, default=0)

    def is_trimmable(self):
        return all(c.is_trimmable() for c in self.caches)

    def trim(self, n):
        for c in self.caches:
            m = c.trim(n)
        return m

    def to_quantized(self, group_size: int = 64, bits: int = 8):
        converted = []
        changed = False
        for c in self.caches:
            if getattr(c, "quantize_with_cache_list", False):
                q = c.to_quantized(group_size=group_size, bits=bits)
                converted.append(q)
                changed = changed or q is not c
            else:
                converted.append(c)
        if not changed:
            return self
        return CacheList(*converted)

    @property
    def state(self):
        return [c.state for c in self.caches]

    @state.setter
    def state(self, v):
        for c, s in zip(self.caches, v):
            c.state = s

    @property
    def meta_state(self):
        return (
            [type(c).__name__ for c in self.caches],
            [c.meta_state for c in self.caches],
        )

    @meta_state.setter
    def meta_state(self, v):
        for c, m in zip(self.caches, v[1]):
            c.meta_state = m

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        for c in self.caches:
            c.filter(batch_indices)

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        for c, o in zip(self.caches, other.caches):
            c.extend(o)

    @classmethod
    def merge(cls, caches):
        cache = cls()
        cache.caches = tuple(
            caches[0].caches[i].merge([c.caches[i] for c in caches])
            for i in range(len(caches[0].caches))
        )
        return cache

    def extract(self, idx):
        return CacheList(*(c.extract(idx) for c in self.caches))

    def prepare(self, **kwargs):
        for c in self.caches:
            c.prepare(**kwargs)

    def finalize(self):
        for c in self.caches:
            c.finalize()

    def size(self):
        return max(c.size() for c in self.caches)

    def empty(self):
        return self.caches[0].empty()

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.caches)

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.caches = [
            globals()[c].from_state(s, m) for s, c, m in zip(state, *meta_state)
        ]
        return obj


def dynamic_roll(x, shifts, axis):
    n = x.shape[axis]
    expand_shifts = (...,) + (None,) * (x.ndim - axis)
    expand_indices = expand_shifts[:-1]
    idx = (mx.arange(n)[expand_indices] - shifts[expand_shifts]) % n
    rolled = mx.take_along_axis(x, idx, axis=axis)
    return rolled


class BatchKVCache(_BaseCache):
    step = 256

    def __init__(self, left_padding: List[int]):
        """
        The BatchKV cache expects inputs to be left-padded.

        E.g. the following prompts:

            [1, 3, 5]
            [7]
            [2, 6, 8, 9]

        Should be padded like so:

            [0, 1, 3, 5]
            [0, 0, 0, 7]
            [2, 6, 8, 9]

        And ``left_padding`` specifies the amount of padding for each.
        In this case, ``left_padding = [1, 3, 0]``.
        """
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0

        self._right_padding = None

    def update_and_fetch(self, keys, values):
        prev = self._idx
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self._idx += keys.shape[2]
        self.keys[..., prev : self._idx, :] = keys
        self.values[..., prev : self._idx, :] = values
        return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            self.keys = dynamic_roll(self.keys, padding[:, None], axis=2)
            self.values = dynamic_roll(self.values, padding[:, None], axis=2)
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    @property
    def state(self):
        k, v = self.keys, self.values
        if self._idx < k.shape[2]:
            k = k[..., : self._idx, :]
            v = v[..., : self._idx, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v
        self._idx = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        # Shift left to reduce padding
        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = self.keys[..., min_left_pad:, :]
                self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            B, H, L1, D = self.keys.shape
            M = self.values.shape[3]
        if other.keys is not None:
            B, H, L2, D = other.keys.shape
            M = other.values.shape[3]
        max_size = max(L1, L2)

        # Pad the keys and values so they are right-justified
        # with the index and the same size
        def pad(c):
            k, v = c.keys, c.values
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.array([]).reshape(Bc, H, 0, D)
                v = mx.array([]).reshape(Bc, H, 0, M)
            left = max_idx - c._idx
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx

    def extract(self, idx):
        cache = KVCache()
        padding = self.left_padding[idx].item()
        cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding : self._idx])
        cache.values = mx.contiguous(self.values[idx : idx + 1, :, padding : self._idx])
        cache.offset = cache.keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        # No cache has content so make an empty one
        if max_length == 0:
            return cls([0] * len(caches))

        padding = [max_length - l for l in lengths]
        B = len(caches)
        H = max(c.keys.shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values.shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + c.offset] = c.keys[..., : c.offset, :]
            values[i : i + 1, :, p : p + c.offset] = c.values[..., : c.offset, :]

        cache = cls(padding)
        cache.keys = keys
        cache.values = values
        cache.offset += keys.shape[2]
        cache._idx = keys.shape[2]

        return cache

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class BatchGlmMlaKVCache(BatchKVCache):
    """Batch-aware GLM-5.2 MLA cache that can opt into latent-only int8."""

    quantize_with_cache_list = True

    def extend(self, other):
        if isinstance(other, BatchQuantizedGlmMlaKVCache):
            raise ValueError(
                "Cannot extend BatchGlmMlaKVCache with "
                "BatchQuantizedGlmMlaKVCache."
            )
        super().extend(other)

    def to_quantized(
        self, group_size: int = 64, bits: int = 8
    ) -> "BatchQuantizedGlmMlaKVCache":
        quant_cache = BatchQuantizedGlmMlaKVCache(
            self.left_padding, group_size=group_size, bits=bits
        )
        quant_cache.offset = self.offset
        quant_cache._idx = self._idx
        quant_cache._right_padding = self._right_padding
        if self.keys is not None:
            keys = self.keys[..., : self._idx, :]
            quant_cache._check_key_dim(keys.shape[-1])
            quant_cache.keys = mx.quantize(
                keys,
                group_size=group_size,
                bits=bits,
            )
            quant_cache.values = self.values[..., : self._idx, :]
        return quant_cache


class BatchQuantizedGlmMlaKVCache(_BaseCache):
    """Batch-aware GLM-5.2 MLA cache with only latent KV stored as int8."""

    step = 256
    quantize_with_cache_list = True

    def __init__(self, left_padding: List[int], group_size: int = 64, bits: int = 8):
        if bits != 8:
            raise ValueError("GLM MLA latent KV cache only supports int8.")
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = -self.left_padding
        self._idx = 0
        self._right_padding = None
        self.group_size = group_size
        self.bits = bits

    def _check_key_dim(self, key_dim: int):
        if key_dim % self.group_size != 0:
            raise ValueError(
                "GLM MLA latent KV cache dimension must be divisible by "
                f"kv_group_size ({key_dim} vs {self.group_size})."
            )

    def _trimmed_data(self):
        if self.keys is None:
            return self.keys, self.values
        if self._idx == self.keys[0].shape[2]:
            return self.keys, self.values
        return (
            tree_map(lambda x: x[..., : self._idx, :], self.keys),
            self.values[..., : self._idx, :],
        )

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        self._check_key_dim(k_head_dim)
        prev = self._idx

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            n_steps = (self.step + num_steps - 1) // self.step
            key_shape = (B, n_kv_heads, n_steps * self.step)
            value_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)

            def init_quant(dim):
                return (
                    mx.zeros((*key_shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*key_shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*key_shape, dim // self.group_size), dtype=keys.dtype),
                )

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = tree_map(lambda x: x[..., :prev, :], self.keys)
                    self.values = self.values[..., :prev, :]

                def expand_quant(x):
                    new_x = mx.zeros((*key_shape, x.shape[-1]), dtype=x.dtype)
                    return mx.concatenate([x, new_x], axis=2)

                self.keys = tree_map(expand_quant, self.keys)
                self.values = mx.concatenate(
                    [self.values, mx.zeros(value_shape, values.dtype)], axis=2
                )
            else:
                self.keys = init_quant(k_head_dim)
                self.values = mx.zeros(value_shape, values.dtype)

        self.offset += num_steps
        self._idx += num_steps

        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self._idx, :] = q_keys[i]
        self.values[..., prev : self._idx, :] = values

        return self._trimmed_data()

    def dequantize_keys(self, keys=None):
        if keys is None:
            keys = self._trimmed_data()[0]
        return mx.dequantize(*keys, group_size=self.group_size, bits=self.bits)

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty "
                    "BatchQuantizedGlmMlaKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            self.keys = tree_map(
                lambda x: dynamic_roll(x, padding[:, None], axis=2),
                self.keys,
            )
            self.values = dynamic_roll(self.values, padding[:, None], axis=2)
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    @property
    def state(self):
        keys, values = self._trimmed_data()
        return keys, values, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v
        self._idx = 0 if self.keys is None else self.keys[0].shape[2]
        self._right_padding = None

    @property
    def meta_state(self):
        return tuple(map(str, (self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self.group_size, self.bits = map(int, v)
        self._right_padding = None

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            self.keys = tree_map(lambda x: x[batch_indices], self.keys)
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        if self._right_padding is not None:
            self._right_padding = self._right_padding[batch_indices]

        # Shift left to reduce padding.
        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = tree_map(
                    lambda x: x[..., min_left_pad:, :],
                    self.keys,
                )
                self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def _empty_key_tuple(self, batch_size, template):
        return tuple(
            mx.array([], dtype=t.dtype).reshape(batch_size, t.shape[1], 0, t.shape[3])
            for t in template
        )

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        if not isinstance(other, BatchQuantizedGlmMlaKVCache):
            raise ValueError(
                "Cannot extend BatchQuantizedGlmMlaKVCache with "
                f"{type(other).__name__}."
            )
        if self.group_size != other.group_size or self.bits != other.bits:
            raise ValueError(
                "BatchQuantizedGlmMlaKVCache can only extend caches with the same "
                "group size and bit width."
            )

        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        key_template = self.keys if self.keys is not None else other.keys
        if self.keys is not None:
            L1 = self.keys[0].shape[2]
            value_template = self.values
        if other.keys is not None:
            L2 = other.keys[0].shape[2]
            value_template = other.values
        max_size = max(L1, L2)

        def pad(c):
            k, v = c.keys, c.values
            if k is None:
                batch_size = c.offset.shape[0]
                k = self._empty_key_tuple(batch_size, key_template)
                v = mx.array([], dtype=value_template.dtype).reshape(
                    batch_size,
                    value_template.shape[1],
                    0,
                    value_template.shape[3],
                )
            left = max_idx - c._idx
            right = max_size - k[0].shape[2] - left
            if right < 0:
                k = tree_map(lambda x: x[..., :right, :], k)
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad_width = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = tree_map(lambda x: mx.pad(x, pad_width), k)
                v = mx.pad(v, pad_width)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        (self_keys, self_values, self_offset, self_left_padding), (
            other_keys,
            other_values,
            other_offset,
            other_left_padding,
        ) = pad(self), pad(other)
        self.keys = tuple(
            mx.concatenate([a, b]) for a, b in zip(self_keys, other_keys)
        )
        self.values = mx.concatenate([self_values, other_values])
        self.offset = mx.concatenate([self_offset, other_offset])
        self.left_padding = mx.concatenate([self_left_padding, other_left_padding])
        self._idx = max_idx

    def extract(self, idx):
        cache = QuantizedGlmMlaKVCache(group_size=self.group_size, bits=self.bits)
        if self.keys is None:
            return cache
        padding = self.left_padding[idx].item()
        cache.keys = tuple(
            mx.contiguous(k[idx : idx + 1, :, padding : self._idx])
            for k in self.keys
        )
        cache.values = mx.contiguous(
            self.values[idx : idx + 1, :, padding : self._idx]
        )
        cache.offset = cache.keys[0].shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        if not caches:
            return cls([])
        if not all(c.group_size == caches[0].group_size for c in caches):
            raise ValueError(
                "BatchQuantizedGlmMlaKVCache can only merge caches with the same "
                "group size"
            )
        if not all(c.bits == caches[0].bits for c in caches):
            raise ValueError(
                "BatchQuantizedGlmMlaKVCache can only merge caches with the same "
                "bit width"
            )

        lengths = [c.size() for c in caches]
        max_length = max(lengths, default=0)
        group_size = caches[0].group_size
        bits = caches[0].bits

        if max_length == 0:
            return cls([0] * len(caches), group_size=group_size, bits=bits)

        padding = [max_length - l for l in lengths]
        B = len(caches)
        key_template = next(c.keys for c in caches if c.keys is not None)
        value_template = next(c.values for c in caches if c.values is not None)
        keys = tuple(
            mx.zeros(
                (B, t.shape[1], max_length, t.shape[3]),
                dtype=t.dtype,
            )
            for t in key_template
        )
        values = mx.zeros(
            (B, value_template.shape[1], max_length, value_template.shape[3]),
            dtype=value_template.dtype,
        )
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            for e in range(len(keys)):
                keys[e][i : i + 1, :, p : p + c.offset] = c.keys[e][
                    ..., : c.offset, :
                ]
            values[i : i + 1, :, p : p + c.offset] = c.values[..., : c.offset, :]

        cache = cls(padding, group_size=group_size, bits=bits)
        cache.keys = keys
        cache.values = values
        cache.offset += max_length
        cache._idx = max_length

        return cache

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    def to_quantized(self, group_size: int = 64, bits: int = 8):
        if bits != 8:
            raise ValueError("GLM MLA latent KV cache only supports int8.")
        if self.group_size == group_size:
            return self
        quant_cache = BatchQuantizedGlmMlaKVCache(
            self.left_padding,
            group_size=group_size,
            bits=bits,
        )
        quant_cache.offset = self.offset
        quant_cache._idx = self._idx
        quant_cache._right_padding = self._right_padding
        if self.keys is not None:
            latent = self.dequantize_keys()
            quant_cache.keys = mx.quantize(latent, group_size=group_size, bits=bits)
            quant_cache.values = self.values[..., : self._idx, :]
        return quant_cache

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, self.keys, 0) + self.values.nbytes


class BatchRotatingKVCache(_BaseCache):
    step = 256

    def __init__(self, max_size, left_padding: List[int]):
        self.keys = None
        self.values = None

        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])

        self.max_size = max_size
        self._idx = 0
        self._offset = 0
        self.rotated = False

        # Lengths for right_padded inputs to make sure that padding tokens do
        # not evict valid tokens.
        self._lengths = None

    def _trim(self, trim_size, v, append=None):
        if trim_size > 0:
            v = v[..., trim_size:, :]
        if append is not None:
            return mx.concatenate([v, append], axis=2)
        return v

    def _temporal_order(self):
        """
        Rearrange the cache into temporal order.
        """
        if self.rotated:
            self.keys = mx.roll(self.keys, -self._idx, axis=2)
            self.values = mx.roll(self.values, -self._idx, axis=2)
            self._idx = self.keys.shape[2]
            self.rotated = False

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self._temporal_order()

            # Slice off the end if needed
            if self.keys.shape[2] > self._idx:
                self.keys = self.keys[..., : self._idx, :]
                self.values = self.values[..., : self._idx, :]

            # Roll right sequences that are padded to make sure that we don't
            # trim valid cache entries
            if self._lengths is not None:
                roll = mx.maximum(0, self.offset - self._lengths)
                self.keys = dynamic_roll(self.keys, roll[:, None], axis=2)
                self.values = dynamic_roll(self.values, roll[:, None], axis=2)
                self.left_padding += roll
                self.offset -= roll

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size + 1
            if trim_size > 0:
                self.left_padding -= trim_size
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset += keys.shape[2]
        self._offset += keys.shape[2]
        self._idx = self.keys.shape[2]

        # Make sure left_padding and offset are evaluated
        self.keys = mx.depends(self.keys, (self.left_padding, self.offset))

        return self.keys, self.values

    def _update_in_place(self, keys, values):
        if self._lengths is not None:
            raise RuntimeError(
                "finalize() should be called before deocoding with BatchRotatingKVCache"
            )

        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self._offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size
            self.left_padding -= trim_size

        # Rotate
        if self._idx == self.max_size:
            self.rotated = True
            self._idx = 0
        if self.rotated:
            self.left_padding -= S

        # Assign
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self._offset += S
        self.offset += S
        self._idx += S

        # Make sure left_padding and offset are evaluated
        self.keys = mx.depends(self.keys, (self.left_padding, self.offset))

        # If the buffer is not full, slice off the end
        if self._offset < self.max_size:
            return (
                self.keys[..., : self._offset, :],
                self.values[..., : self._offset, :],
            )
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchRotatingKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._lengths = mx.array(lengths) + self.offset

    def finalize(self):
        if self._lengths is not None:
            roll = mx.maximum(0, self.offset - self._lengths)
            self.keys = dynamic_roll(self.keys, roll[:, None], axis=2)
            self.values = dynamic_roll(self.values, roll[:, None], axis=2)
            self.left_padding += roll
            self.offset -= roll
            self._lengths = None

    @property
    def state(self):
        k, v = self.keys, self.values
        if self._offset < k.shape[2]:
            k, v = k[..., : self._offset, :], v[..., : self._offset, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.max_size, self._offset, self._idx, self.rotated)))

    @meta_state.setter
    def meta_state(self, v):
        self.max_size, self._offset, self._idx = map(
            int,
            v[:3],
        )
        self.rotated = bool(v[3])

    def is_trimmable(self):
        return self._offset < self.max_size

    def trim(self, n):
        n = min(self._offset, n)
        self._offset -= n
        self._idx -= n
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        raise NotImplementedError("BatchRotatingKVCache Quantization NYI")

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        left_padding = self.left_padding
        window_size = window_size or self.max_size
        offset = min(self.max_size - 1, self._offset)
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        mask &= linds < rinds + window_size
        if (trim_size := self._idx - self.max_size + int(N > 1)) > 0:
            left_padding = left_padding - trim_size

        rotated = N == 1 and (self.rotated or self._idx >= self.max_size)
        if rotated:
            left_padding = left_padding - 1

        mask = mask & (rinds >= mx.expand_dims(left_padding, (1, 2, 3)))

        if rotated:
            idx = self._idx
            if idx >= self.max_size:
                idx = 0
            mask = mx.roll(mask, shift=idx + 1, axis=-1)

        return mask

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        if (self.rotated != other.rotated) or self._idx != other._idx:
            self._temporal_order()
            other._temporal_order()

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            B, H, L1, D = self.keys.shape
            M = self.values.shape[3]
        if other.keys is not None:
            B, H, L2, D = other.keys.shape
            M = other.values.shape[3]
        max_size = max(L1, L2)

        def pad(c):
            left = max_idx - c._idx
            k, v = c.keys, c.values
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.array([]).reshape(Bc, H, 0, D)
                v = mx.array([]).reshape(Bc, H, 0, M)
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx
        self._offset = max(self._offset, other._offset)

    def extract(self, idx):
        mx.eval(self.left_padding, self.offset)
        cache = RotatingKVCache(self.max_size)
        padding = max(0, self.left_padding.tolist()[idx])
        offset = self.offset.tolist()[idx]
        cache.keys = self.keys[idx : idx + 1]
        cache.values = self.values[idx : idx + 1]
        cache._idx = self._idx
        if self.rotated:
            cache.keys = mx.roll(cache.keys, -self._idx, axis=2)
            cache.values = mx.roll(cache.values, -self._idx, axis=2)
            cache._idx = self.max_size
        cache.keys = mx.contiguous(cache.keys[:, :, padding : cache._idx])
        cache.values = mx.contiguous(cache.values[:, :, padding : cache._idx])
        cache.offset = offset
        cache._idx = cache.keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        if not all(c.max_size == caches[0].max_size for c in caches):
            raise ValueError(
                "BatchRotatingKVCache can only merge caches with the same maximum size"
            )

        offsets = [c.offset for c in caches]
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        # No cache has content so make an empty one
        if max_length == 0:
            return cls(caches[0].max_size, [0] * len(caches))

        padding = [max_length - l for l in lengths]
        B = len(caches)
        H = max(c.keys.shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values.shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, l, c) in enumerate(zip(padding, lengths, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + l] = c._temporal_order(c.keys)[..., -l:, :]
            values[i : i + 1, :, p : p + l] = c._temporal_order(c.values)[..., -l:, :]

        cache = cls(caches[0].max_size, padding)
        cache.keys = keys
        cache.values = values
        cache.offset = mx.array(offsets)
        cache._idx = keys.shape[2]
        cache._offset = keys.shape[2]

        return cache

    def size(self):
        return min(self._offset, self.max_size)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class TokenBuffer:
    """A simple token buffer that can be efficiently appended to in a similar
    fashion to the KVCache.

    Perhaps these could share some logic in the future.
    """

    step = 256

    def __init__(self, tokens=[]):
        self._buffer = mx.array(tokens, dtype=mx.int32)
        self._size = len(tokens)

    def update_and_fetch(self, tokens):
        start = self._size
        end = start + len(tokens)

        new_size = ((end + self.step - 1) // self.step) * self.step
        if new_size > self._buffer.size:
            self._buffer = mx.concatenate(
                [self._buffer, mx.zeros(new_size - self._buffer.size, dtype=mx.int32)]
            )
        self._buffer[start:end] = tokens
        self._size = end

        return self._buffer[:end]

    @property
    def state(self):
        return self._buffer

    @property
    def tokens(self):
        return self._buffer[: self._size]


@dataclass
class PromptTrieResult:
    model: Any
    exact: Optional[List[int]]  # Exact match found
    shorter: Optional[List[int]]  # Longest prefix with a value
    longer: Optional[List[int]]  # Shortest value that extends beyond tokens
    common_prefix: int  # Length of common prefix with any path


class PromptTrie:
    def __init__(self):
        self._trie = {}

    def add(self, model: Any, tokens: List[int], value: Any):
        if model not in self._trie:
            self._trie[model] = {}

        current = self._trie[model]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]
        prev = current.get("__value__", None)
        current["__value__"] = value
        return prev

    def get(self, model: Any, tokens: List[int]):
        current = self._trie[model]
        for tok in tokens:
            current = current[tok]
        return current["__value__"]

    def pop(self, model: Any, tokens: List[int]):
        path = [self._trie[model]]
        for tok in tokens:
            path.append(path[-1][tok])
        value = path[-1].pop("__value__")
        for i in range(len(tokens), 0, -1):
            node = path[i]
            parent = path[i - 1]
            tok = tokens[i - 1]
            if len(node) > 0:
                break
            del parent[tok]
        return value

    def pop_prefixes(self, model: Any, tokens: List[int]):
        values = []
        current = self._trie[model]
        for i, tok in enumerate(tokens):
            if "__value__" in current:
                values.append((i, current.pop("__value__")))
            current = current[tok]
        return values

    def search(self, model: Any, tokens: List[int]) -> PromptTrieResult:
        if model not in self._trie:
            return PromptTrieResult(model, None, None, None, 0)

        current = self._trie[model]

        if not tokens and "__value__" in current:
            return PromptTrieResult(model, [], None, None, 0)

        # Walk the tokens as far as we can
        last_index = -1
        index = 0
        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "__value__" in current:
                last_index = index
            index += 1

        # Got an exact match
        if last_index == len(tokens) - 1 >= 0:
            return PromptTrieResult(model, tokens, None, None, 0)

        # Check if we found a prefix at any point
        shorter = None
        if last_index > 0:
            shorter = tokens[: last_index + 1]

        # Check for sequences that are longer
        longer = None
        common_prefix = index
        if index > 0:
            best = None
            stack = [(current, [])]
            while stack:
                current, extra = stack.pop()
                if "__value__" in current:
                    if best is None or len(extra) < len(best):
                        best = extra
                elif best is None or len(extra) < len(best):
                    for tok in current:
                        stack.append((current[tok], extra + [tok]))
            longer = tokens[:index] + best
        return PromptTrieResult(model, None, shorter, longer, common_prefix)


class LRUPromptCache:
    @dataclass
    class CacheEntry:
        prompt_cache: List[Any]
        nbytes: int
        cache_type: str

    class CacheOrder:
        def __init__(self, ordering: List[str] = ["assistant", "user", "system"]):
            self._ordering = ordering
            self._lrus = {k: deque() for k in ordering}

        def __len__(self):
            return sum(len(lru) for lru in self._lrus.values())

        def push(self, model: Any, tokens: List[Any], cache_type: str = "assistant"):
            self._lrus[cache_type].append((model, tokens))

        def remove(self, model: Any, tokens: List[Any]):
            for cache_type in self._ordering:
                try:
                    self._lrus[cache_type].remove((model, tokens))
                    break
                except ValueError:
                    pass

        def pop(self):
            i = 0
            while i + 1 < len(self._ordering):
                lru_a = self._lrus[self._ordering[i]]
                lru_b = self._lrus[self._ordering[i + 1]]
                if lru_a and len(lru_a) >= len(lru_b):
                    return lru_a.popleft()
                i += 1
            return lru_b.popleft()

    def __init__(self, max_size: int = 10, max_bytes: int = 1 << 63):
        self.max_size = max_size
        self.max_bytes = max_bytes
        self._trie = PromptTrie()
        self._lru = LRUPromptCache.CacheOrder()
        self._n_bytes = 0
        self._n_bytes_by_type = {k: 0 for k in self._lru._ordering}

    def __len__(self):
        return len(self._lru)

    @property
    def nbytes(self):
        return self._n_bytes

    def fetch_nearest_cache(self, model: Any, tokens: List[int]):
        result = self._trie.search(model, tokens)
        if result.exact is not None:
            cache_entry = self._trie.get(result.model, result.exact)
            return copy.deepcopy(cache_entry.prompt_cache), []

        short_length = len(result.shorter) if result.shorter is not None else 0
        if result.longer is not None and result.common_prefix > short_length:
            cache_entry = self._trie.get(result.model, result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                cache = copy.deepcopy(cache_entry.prompt_cache)
                prefix = min(len(tokens) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                trim_prompt_cache(cache, num_to_trim)
                return cache, tokens[prefix:]

        if short_length > 0:
            cache_entry = self._trie.get(result.model, result.shorter)
            return copy.deepcopy(cache_entry.prompt_cache), tokens[short_length:]

        return None, tokens

    def insert_cache(
        self,
        model: Any,
        tokens: List[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
    ):
        # Make the cache entry
        entry = LRUPromptCache.CacheEntry(
            prompt_cache, sum(c.nbytes for c in prompt_cache), cache_type
        )

        # Insert into the trie and update the byte counter and lru position
        self._n_bytes += entry.nbytes
        self._n_bytes_by_type[cache_type] += entry.nbytes
        prev = self._trie.add(model, tokens, entry)
        if prev is not None:
            self._n_bytes -= prev.nbytes
            self._n_bytes_by_type[prev.cache_type] -= prev.nbytes
            self._lru.remove(model, tokens)
        self._lru.push(model, tokens, cache_type)

        # If it is a trimmable cache remove all prefixes cause they just take
        # space
        if can_trim_prompt_cache(prompt_cache):
            for prefix_len, entry in self._trie.pop_prefixes(model, tokens):
                self._n_bytes -= entry.nbytes
                self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
                self._lru.remove(model, tokens[:prefix_len])

        # Ensure we match the constraints
        if len(self._lru) > self.max_size:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
        while self._n_bytes > self.max_bytes:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes

    def trim_to(
        self, *, n_sequences: Optional[int] = None, n_bytes: Optional[int] = None
    ):
        n_sequences = max(0, n_sequences) if n_sequences is not None else 1 << 63
        n_bytes = max(0, n_bytes) if n_bytes is not None else 1 << 63

        while len(self._lru) > n_sequences:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
        while self._n_bytes > n_bytes:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes

    def stats_by_type(self):
        result = {}
        for cache_type in self._lru._ordering:
            result[cache_type] = {
                "n_sequences": len(self._lru._lrus[cache_type]),
                "n_bytes": self._n_bytes_by_type[cache_type],
            }
        return result
