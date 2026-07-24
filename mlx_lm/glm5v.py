# Copyright © 2026 Apple Inc.

"""Initial GLM-5.2 Vision support.

This module combines the frozen MoonViT-3d tower from Kimi-K2.6 with the
trained GLM-5.2 PatchMerger projector.  It intentionally remains an adapter:
the GLM-5.2 language model is loaded normally by :func:`mlx_lm.load`, while
MoonViT and the projector are loaded from their much smaller, separate files.
"""

import base64
import copy
import io
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union
from urllib.parse import unquote, unquote_to_bytes, urlsplit

import mlx.core as mx
import mlx.nn as nn
import numpy as np


DEFAULT_IMAGE_TOKEN_ID = 154854
DEFAULT_MOONVIT_REPO = "moonshotai/Kimi-K2.6"
DEFAULT_MOONVIT_SHARD = "model-00064-of-000064.safetensors"
IMAGE_PLACEHOLDER = "<|begin_of_image|><|image|><|end_of_image|>"
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000
DEFAULT_MAX_IMAGES = 8
DEFAULT_MAX_TOTAL_IMAGE_PIXELS = 64_000_000


def normalize_vision_messages(messages):
    """Replace structured image parts with GLM image tokens.

    The returned messages are a deep copy, so the HTTP request body remains
    untouched. Image sources are returned in the same order as their inserted
    placeholders.
    """
    prepared = copy.deepcopy(messages)
    image_sources = []
    for message in prepared:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        normalized = []
        for part in content:
            if not isinstance(part, dict):
                normalized.append(part)
                continue
            part_type = part.get("type")
            if part_type in ("input_text", "output_text"):
                text = part.get("text")
                if not isinstance(text, str):
                    raise ValueError(
                        f"Vision content type {part_type!r} requires text."
                    )
                normalized.append({"type": "text", "text": text})
                continue
            if part_type not in ("image", "image_url", "input_image"):
                normalized.append(part)
                continue

            source = part.get("image_url")
            if isinstance(source, dict):
                source = source.get("url")
            if source is None and part_type == "image":
                source = part.get("image") or part.get("url")
            if not isinstance(source, str) or not source:
                raise ValueError(
                    f"Vision content type {part_type!r} requires an image URL."
                )
            image_sources.append(source)
            normalized.append({"type": "text", "text": IMAGE_PLACEHOLDER})
        message["content"] = normalized
    return prepared, image_sources


def _checked_image_bytes(data, max_bytes):
    if len(data) > max_bytes:
        raise ValueError(
            f"Image contains {len(data)} bytes; limit is {max_bytes} bytes."
        )
    return data


def load_image_source(
    source,
    *,
    max_bytes=DEFAULT_MAX_IMAGE_BYTES,
    max_pixels=DEFAULT_MAX_IMAGE_PIXELS,
    allow_local=True,
):
    """Load a PIL RGB image from a data URL, local path, or PIL image.

    HTTP(S) fetching is deliberately not performed by the server: clients can
    download remote images and send a bounded data URL instead.
    """
    from PIL import Image

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive.")
    if isinstance(source, Image.Image):
        if source.width * source.height > max_pixels:
            raise ValueError(
                f"Decoded image exceeds the {max_pixels}-pixel limit."
            )
        return source.convert("RGB")
    if isinstance(source, Path):
        source = str(source)
    if isinstance(source, str):
        parsed = urlsplit(source)
        if parsed.scheme == "data":
            header, separator, payload = source.partition(",")
            if not separator or not header.lower().startswith("data:image/"):
                raise ValueError("Only data:image/... URLs are supported.")
            if ";base64" in header.lower():
                if len(payload) > 4 * ((max_bytes + 2) // 3) + 4:
                    raise ValueError(f"Image exceeds the {max_bytes}-byte limit.")
                try:
                    data = base64.b64decode(payload, validate=True)
                except (ValueError, base64.binascii.Error) as error:
                    raise ValueError("Malformed base64 image data URL.") from error
            else:
                data = unquote_to_bytes(payload)
            stream = io.BytesIO(_checked_image_bytes(data, max_bytes))
        elif parsed.scheme in ("http", "https"):
            raise ValueError(
                "Remote HTTP(S) image fetching is not supported; download the "
                "image client-side and send a data:image URL."
            )
        elif parsed.scheme == "file":
            if not allow_local:
                raise ValueError(
                    "Local image paths are disabled for HTTP requests; use a data "
                    "URL or start the server with --vision-allow-local-images."
                )
            if parsed.netloc not in ("", "localhost"):
                raise ValueError("Only local file:// image URLs are supported.")
            path = Path(unquote(parsed.path)).expanduser()
            if path.stat().st_size > max_bytes:
                raise ValueError(f"Local image exceeds the {max_bytes}-byte limit.")
            stream = path
        else:
            if parsed.scheme:
                raise ValueError(
                    f"Unsupported image URL scheme: {parsed.scheme!r}."
                )
            if not allow_local:
                raise ValueError(
                    "Local image paths are disabled for HTTP requests; use a data "
                    "URL or start the server with --vision-allow-local-images."
                )
            path = Path(source).expanduser()
            if path.stat().st_size > max_bytes:
                raise ValueError(f"Local image exceeds the {max_bytes}-byte limit.")
            stream = path
    else:
        stream = source

    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(stream) as image:
            if image.width * image.height > max_pixels:
                raise ValueError(
                    f"Decoded image exceeds the {max_pixels}-pixel limit."
                )
            image.load()
            return image.convert("RGB")


def load_image_sources(
    sources,
    *,
    max_bytes=DEFAULT_MAX_IMAGE_BYTES,
    max_pixels=DEFAULT_MAX_IMAGE_PIXELS,
    max_images=DEFAULT_MAX_IMAGES,
    max_total_pixels=DEFAULT_MAX_TOTAL_IMAGE_PIXELS,
    allow_local=True,
):
    """Load and bound all images in one request."""
    sources = list(sources)
    if max_images <= 0:
        raise ValueError("max_images must be positive.")
    if max_total_pixels <= 0:
        raise ValueError("max_total_pixels must be positive.")
    if len(sources) > max_images:
        raise ValueError(
            f"Vision request has {len(sources)} images; limit is {max_images}."
        )

    images = []
    total_pixels = 0
    for source in sources:
        image = load_image_source(
            source,
            max_bytes=max_bytes,
            max_pixels=max_pixels,
            allow_local=allow_local,
        )
        total_pixels += image.width * image.height
        if total_pixels > max_total_pixels:
            raise ValueError(
                "Vision request decoded pixels exceed the "
                f"{max_total_pixels}-pixel aggregate limit."
            )
        images.append(image)
    return images


@dataclass
class MoonViTConfig:
    patch_size: int = 14
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    init_pos_emb_time: int = 4
    pos_emb_type: str = "divided_fixed"
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    merge_kernel_size: tuple[int, int] = (2, 2)
    merge_type: str = "sd2_tpool"
    video_attn_type: str = "spatial_temporal"
    text_hidden_size: int = 6144
    projector_ln_eps: float = 1e-5

    @classmethod
    def from_dict(cls, config):
        """Accept both the GLM5V and upstream Kimi vision-config spellings."""
        aliases = {
            "num_attention_heads": ("num_attention_heads", "vt_num_attention_heads"),
            "num_hidden_layers": ("num_hidden_layers", "vt_num_hidden_layers"),
            "hidden_size": ("hidden_size", "vt_hidden_size", "mm_hidden_size"),
            "intermediate_size": ("intermediate_size", "vt_intermediate_size"),
        }
        values = {}
        for field in cls.__dataclass_fields__:
            if field in aliases:
                for key in aliases[field]:
                    if key in config:
                        values[field] = config[key]
                        break
            elif field in config:
                values[field] = config[field]
        if "merge_kernel_size" in values:
            merge = values["merge_kernel_size"]
            if isinstance(merge, int):
                merge = (merge, merge)
            values["merge_kernel_size"] = tuple(merge)
        return cls(**values)


def _cubic_weight(t):
    """PyTorch-compatible cubic convolution weights (a=-0.75)."""
    a = -0.75
    at = mx.abs(t)
    at2 = at * at
    at3 = at2 * at
    near = (a + 2.0) * at3 - (a + 3.0) * at2 + 1.0
    far = a * at3 - 5.0 * a * at2 + 8.0 * a * at - 4.0 * a
    return mx.where(at <= 1.0, near, mx.where(at < 2.0, far, mx.zeros_like(t)))


def _bicubic_interpolate(x: mx.array, size: tuple[int, int]) -> mx.array:
    """Bicubic NCHW interpolation matching torch's non-antialiased default."""
    batch, channels, in_h, in_w = x.shape
    out_h, out_w = size
    input_dtype = x.dtype
    x = x.astype(mx.float32)

    y = (mx.arange(out_h, dtype=mx.float32) + 0.5) / out_h * in_h - 0.5
    z = (mx.arange(out_w, dtype=mx.float32) + 0.5) / out_w * in_w - 0.5

    def weights(coords, input_size):
        start = mx.floor(coords - 2.0).astype(mx.int32) + 1
        pixels = start[:, None] + mx.arange(5, dtype=mx.int32)[None, :]
        weight = _cubic_weight(coords[:, None] - pixels.astype(mx.float32))
        # PyTorch's align_corners=False bicubic path clamps border indices.
        # Repeated clamped pixels therefore accumulate their original cubic
        # weights; masking and renormalizing the out-of-range taps diverges.
        return mx.clip(pixels, 0, input_size - 1), weight

    pix_y, wy = weights(y, in_h)
    pix_x, wx = weights(z, in_w)
    gathered_y = x[:, :, pix_y.reshape(-1), :].reshape(
        batch, channels, out_h, pix_y.shape[1], in_w
    )
    tmp = mx.sum(gathered_y * wy[None, None, :, :, None], axis=3)
    gathered_x = tmp[:, :, :, pix_x.reshape(-1)].reshape(
        batch, channels, out_h, out_w, pix_x.shape[1]
    )
    result = mx.sum(gathered_x * wx[None, None, None, :, :], axis=4)
    return result.astype(input_dtype)


def _sincos_time_embedding(num_frames: int, dim: int) -> mx.array:
    half = dim // 2
    omega = mx.arange(half, dtype=mx.float32) / half
    omega = 1.0 / (10000.0**omega)
    positions = mx.arange(num_frames, dtype=mx.float32)
    values = mx.outer(positions, omega)
    return mx.concatenate([mx.sin(values), mx.cos(values)], axis=1)


class Learnable2DPositionEmbedding(nn.Module):
    def __init__(self, height: int, width: int, num_frames: int, dim: int):
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.weight = mx.ones((height, width, dim))

    def __call__(self, x: mx.array, grid_thws: mx.array) -> mx.array:
        embeddings = []
        time_weight = _sincos_time_embedding(self.num_frames, self.dim)
        for t, height, width in grid_thws.tolist():
            if t > self.num_frames:
                raise ValueError(
                    f"MoonViT supports at most {self.num_frames} frames per chunk, "
                    f"got {t}."
                )
            if (height, width) == (self.height, self.width):
                spatial = self.weight.reshape(-1, self.dim)
            else:
                spatial = (
                    _bicubic_interpolate(
                        self.weight.transpose(2, 0, 1)[None],
                        (height, width),
                    )[0]
                    .transpose(1, 2, 0)
                    .reshape(-1, self.dim)
                )
            if t > 1:
                spatial = (
                    mx.tile(spatial[None], (t, 1, 1))
                    + time_weight[:t, None].astype(spatial.dtype)
                ).reshape(-1, self.dim)
            embeddings.append(spatial)
        return x + mx.concatenate(embeddings, axis=0).astype(x.dtype)


class MoonViTPatchEmbed(nn.Module):
    def __init__(self, config: MoonViTConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            3,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.pos_emb = Learnable2DPositionEmbedding(
            config.init_pos_emb_height,
            config.init_pos_emb_width,
            config.init_pos_emb_time,
            config.hidden_size,
        )

    def __call__(self, pixels: mx.array, grid_thws: mx.array) -> mx.array:
        hidden = self.proj(pixels).reshape(pixels.shape[0], -1)
        return self.pos_emb(hidden, grid_thws)


def _view_as_complex(x: mx.array) -> mx.array:
    x = x.reshape(*x.shape[:-1], -1, 2)
    return x[..., 0] + 1j * x[..., 1]


def _apply_rope(q: mx.array, k: mx.array, frequencies: mx.array):
    frequencies = frequencies[:, None]

    def rotate(x):
        dtype = x.dtype
        complex_x = _view_as_complex(x.astype(mx.float32))
        result = complex_x * frequencies
        result = mx.stack([mx.real(result), mx.imag(result)], axis=-1)
        return result.reshape(*x.shape).astype(dtype)

    return rotate(q), rotate(k)


class Rope2D(nn.Module):
    def __init__(self, dim: int, max_height: int = 512, max_width: int = 512):
        super().__init__()
        if dim % 4:
            raise ValueError("MoonViT head dimension must be divisible by four.")
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self._frequencies = None

    def _precompute(self):
        count = self.max_height * self.max_width
        flat = mx.arange(count, dtype=mx.float32)
        x_position = flat % self.max_width
        y_position = flat // self.max_width
        dimensions = mx.arange(0, self.dim, 4, dtype=mx.float32)
        frequency = 1.0 / (10000.0 ** (dimensions / self.dim))
        x_angle = mx.outer(x_position, frequency)
        y_angle = mx.outer(y_position, frequency)
        x_cis = mx.cos(x_angle) + 1j * mx.sin(x_angle)
        y_cis = mx.cos(y_angle) + 1j * mx.sin(y_angle)
        return mx.stack([x_cis, y_cis], axis=-1).reshape(
            self.max_height, self.max_width, self.dim // 2
        )

    def __call__(self, grid_thws: mx.array):
        if self._frequencies is None:
            self._frequencies = self._precompute()
        outputs = []
        for t, height, width in grid_thws.tolist():
            if not (1 <= height <= self.max_height and 1 <= width <= self.max_width):
                raise ValueError(
                    f"MoonViT grid {(t, height, width)} exceeds "
                    f"{self.max_height}x{self.max_width}."
                )
            spatial = self._frequencies[:height, :width].reshape(-1, self.dim // 2)
            outputs.append(mx.tile(spatial, (t, 1)) if t > 1 else spatial)
        return mx.concatenate(outputs, axis=0)


class MoonViTMLP(nn.Module):
    def __init__(self, config: MoonViTConfig):
        super().__init__()
        self.fc0 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc1 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU(approx="tanh")

    def __call__(self, x):
        return self.fc1(self.activation(self.fc0(x)))


class MoonViTBlock(nn.Module):
    def __init__(self, config: MoonViTConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.norm0 = nn.LayerNorm(config.hidden_size)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.wqkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlp = MoonViTMLP(config)

    def __call__(self, hidden, cu_seqlens, rope):
        residual = hidden
        qkv = self.wqkv(self.norm0(hidden)).reshape(
            hidden.shape[0], 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k = _apply_rope(q, k, rope)

        outputs = []
        bounds = cu_seqlens.tolist()
        for start, end in zip(bounds[:-1], bounds[1:]):
            qs = q[start:end].transpose(1, 0, 2)
            ks = k[start:end].transpose(1, 0, 2)
            vs = v[start:end].transpose(1, 0, 2)
            output = mx.fast.scaled_dot_product_attention(
                qs[None], ks[None], vs[None], scale=self.scale
            )[0]
            outputs.append(output.transpose(1, 0, 2).reshape(end - start, -1))
        hidden = residual + self.wo(mx.concatenate(outputs, axis=0))
        return hidden + self.mlp(self.norm1(hidden))


class MoonViTEncoder(nn.Module):
    def __init__(self, config: MoonViTConfig):
        super().__init__()
        if config.video_attn_type != "spatial_temporal":
            raise ValueError(
                f"Unsupported MoonViT video attention: {config.video_attn_type}"
            )
        self.rope_2d = Rope2D(config.hidden_size // config.num_attention_heads)
        self.blocks = [MoonViTBlock(config) for _ in range(config.num_hidden_layers)]
        self.final_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(self, hidden, grid_thws):
        lengths = mx.prod(grid_thws, axis=1)
        cu_seqlens = mx.cumsum(
            mx.concatenate([mx.zeros((1,), dtype=mx.int32), lengths.astype(mx.int32)])
        )
        rope = self.rope_2d(grid_thws)
        for block in self.blocks:
            hidden = block(hidden, cu_seqlens, rope)
        return self.final_layernorm(hidden)


def patch_merge(
    hidden: mx.array,
    grid_thws: mx.array,
    merge_kernel_size: tuple[int, int] = (2, 2),
):
    """Temporal-average and concatenate spatial patches in official order."""
    outputs = []
    offset = 0
    merge_h, merge_w = merge_kernel_size
    dim = hidden.shape[-1]
    for t, height, width in grid_thws.tolist():
        length = t * height * width
        item = hidden[offset : offset + length]
        if height % merge_h or width % merge_w:
            raise ValueError(
                f"Grid {(t, height, width)} is not divisible by "
                f"merge kernel {merge_kernel_size}."
            )
        item = item.reshape(
            t,
            height // merge_h,
            merge_h,
            width // merge_w,
            merge_w,
            dim,
        )
        item = item.transpose(0, 1, 3, 2, 4, 5).mean(axis=0)
        outputs.append(
            item.reshape(
                (height // merge_h) * (width // merge_w), merge_h * merge_w, dim
            )
        )
        offset += length
    return outputs


class MoonViT(nn.Module):
    def __init__(self, config: MoonViTConfig):
        super().__init__()
        if config.pos_emb_type != "divided_fixed":
            raise ValueError(
                f"Unsupported MoonViT position embedding: {config.pos_emb_type}"
            )
        if config.merge_type != "sd2_tpool":
            raise ValueError(f"Unsupported MoonViT merge type: {config.merge_type}")
        self.config = config
        self.patch_embed = MoonViTPatchEmbed(config)
        self.encoder = MoonViTEncoder(config)

    def __call__(self, pixel_values: mx.array, grid_thws: mx.array):
        if grid_thws.ndim != 2 or grid_thws.shape[1] != 3:
            raise ValueError(f"grid_thws must have shape (N, 3), got {grid_thws.shape}")
        hidden = self.patch_embed(pixel_values, grid_thws)
        hidden = self.encoder(hidden, grid_thws)
        return patch_merge(hidden, grid_thws, self.config.merge_kernel_size)


class PatchMergerProjector(nn.Module):
    """The exact 49,558,272-parameter GLM5V projector."""

    def __init__(self, config: MoonViTConfig):
        super().__init__()
        merge_tokens = math.prod(config.merge_kernel_size)
        merged_size = config.hidden_size * merge_tokens
        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=config.projector_ln_eps)
        self.linear_1 = nn.Linear(merged_size, merged_size)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(merged_size, config.text_hidden_size)

    def __call__(self, features):
        return [
            self.linear_2(
                self.activation(
                    self.linear_1(self.pre_norm(item).reshape(item.shape[0], -1))
                )
            )
            for item in features
        ]


class MoonViTImageProcessor:
    """Kimi-K2.6's NaViT image-only preprocessing path."""

    def __init__(
        self,
        patch_size=14,
        merge_kernel_size=(2, 2),
        in_patch_limit=16384,
        patch_limit_on_one_side=512,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        fixed_output_tokens=None,
    ):
        self.patch_size = patch_size
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.in_patch_limit = in_patch_limit
        self.patch_limit_on_one_side = patch_limit_on_one_side
        self.image_mean = np.asarray(image_mean, dtype=np.float32)
        self.image_std = np.asarray(image_std, dtype=np.float32)
        self.fixed_output_tokens = fixed_output_tokens

    @classmethod
    def from_preprocessor_config(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            media = json.load(handle).get("media_proc_cfg", {})
        merge = media.get("merge_kernel_size", 2)
        if isinstance(merge, int):
            merge = (merge, merge)
        return cls(
            patch_size=media.get("patch_size", 14),
            merge_kernel_size=merge,
            in_patch_limit=media.get("in_patch_limit", 16384),
            patch_limit_on_one_side=media.get("patch_limit_on_one_side", 512),
            image_mean=media.get("image_mean", (0.5, 0.5, 0.5)),
            image_std=media.get("image_std", (0.5, 0.5, 0.5)),
            fixed_output_tokens=media.get("fixed_output_tokens"),
        )

    def resize_config(self, width, height):
        patch = self.patch_size
        merge_h, merge_w = self.merge_kernel_size
        scale_tokens = math.sqrt(
            self.in_patch_limit / (max(1.0, width // patch) * max(1.0, height // patch))
        )
        scale_w = self.patch_limit_on_one_side * patch / width
        scale_h = self.patch_limit_on_one_side * patch / height
        scale = min(1.0, scale_tokens, scale_w, scale_h)
        new_width = min(
            max(1, int(width * scale)),
            self.patch_limit_on_one_side * patch,
        )
        new_height = min(
            max(1, int(height * scale)),
            self.patch_limit_on_one_side * patch,
        )
        factor_w = merge_w * patch
        factor_h = merge_h * patch
        pad_width = (-new_width) % factor_w
        pad_height = (-new_height) % factor_h
        tokens = self.fixed_output_tokens
        if tokens is None:
            tokens = (
                (new_width + pad_width)
                // factor_w
                * ((new_height + pad_height) // factor_h)
            )
        return {
            "new_width": new_width,
            "new_height": new_height,
            "pad_width": pad_width,
            "pad_height": pad_height,
            "num_tokens": tokens,
        }

    @staticmethod
    def _load_image(image):
        return load_image_source(image)

    def __call__(self, images):
        from PIL import Image

        if not isinstance(images, (list, tuple)):
            images = [images]
        all_patches = []
        grids = []
        token_counts = []
        for source in images:
            image = self._load_image(source)
            cfg = self.resize_config(*image.size)
            image = image.resize(
                (cfg["new_width"], cfg["new_height"]),
                Image.Resampling.BICUBIC,
            )
            pixels = np.asarray(image, dtype=np.uint8)
            pixels = np.pad(
                pixels,
                (
                    (0, cfg["pad_height"]),
                    (0, cfg["pad_width"]),
                    (0, 0),
                ),
                constant_values=0,
            )
            pixels = pixels.astype(np.float32) / 255.0
            pixels = (pixels - self.image_mean) / self.image_std
            height, width, channels = pixels.shape
            patch = self.patch_size
            patches = pixels.reshape(
                1,
                height // patch,
                patch,
                width // patch,
                patch,
                channels,
            ).transpose(0, 1, 3, 5, 2, 4)
            patches = patches.reshape(-1, channels, patch, patch)
            all_patches.append(patches)
            grids.append((1, height // patch, width // patch))
            token_counts.append(cfg["num_tokens"])
        return {
            "pixel_values": mx.array(np.concatenate(all_patches)),
            "grid_thws": mx.array(grids, dtype=mx.int32),
            "image_token_counts": token_counts,
        }


def expand_image_tokens(
    input_ids: Union[Sequence[int], mx.array],
    image_token_counts: Sequence[int],
    image_token_id: int = DEFAULT_IMAGE_TOKEN_ID,
) -> mx.array:
    """Expand one ``<|image|>`` token per image to one token per feature."""
    ids = input_ids.tolist() if isinstance(input_ids, mx.array) else list(input_ids)
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Initial GLM5V support accepts batch size one.")
        ids = ids[0]
    image_token_counts = [int(count) for count in image_token_counts]
    runs = []
    for token in ids:
        if token == image_token_id:
            if runs and runs[-1][1]:
                runs[-1][0] += 1
            else:
                runs.append([1, True])
        elif runs:
            runs[-1][1] = False
    run_lengths = [length for length, _ in runs]
    expected = len(image_token_counts)
    if run_lengths == image_token_counts:
        return mx.array(ids, dtype=mx.int32)
    if run_lengths != [1] * expected:
        raise ValueError(
            f"Expected image-token runs {[1] * expected} before expansion or "
            f"{image_token_counts} after expansion, found {run_lengths}."
        )
    output = []
    image_index = 0
    for token in ids:
        if token == image_token_id:
            output.extend([token] * image_token_counts[image_index])
            image_index += 1
        else:
            output.append(token)
    return mx.array(output, dtype=mx.int32)


def insert_image_embeddings(
    input_ids: mx.array,
    text_embeddings: mx.array,
    image_embeddings: mx.array,
    image_token_id: int = DEFAULT_IMAGE_TOKEN_ID,
) -> mx.array:
    """Replace expanded GLM image placeholders without mutating the input."""
    if input_ids.ndim == 1:
        input_ids = input_ids[None]
    if text_embeddings.ndim == 2:
        text_embeddings = text_embeddings[None]
    if input_ids.shape[0] != 1:
        raise ValueError("Initial GLM5V support accepts batch size one.")
    mask = input_ids == image_token_id
    count = int(mx.sum(mask).item())
    if count != image_embeddings.shape[0]:
        raise ValueError(
            f"Image placeholder/features mismatch: {count} tokens and "
            f"{image_embeddings.shape[0]} features."
        )
    feature_index = mx.cumsum(mask.astype(mx.int32), axis=1) - 1
    feature_index = mx.clip(feature_index, 0, max(count - 1, 0))
    selected = image_embeddings[feature_index]
    return mx.where(mask[..., None], selected, text_embeddings)


class GLM5VisionAdapter(nn.Module):
    """Frozen MoonViT plus the trained GLM-5.2 projector."""

    def __init__(
        self,
        config: MoonViTConfig,
        image_token_id: int = DEFAULT_IMAGE_TOKEN_ID,
        image_processor: Optional[MoonViTImageProcessor] = None,
    ):
        super().__init__()
        self.config = config
        self.image_token_id = image_token_id
        self.vision_tower = MoonViT(config)
        self.mm_projector = PatchMergerProjector(config)
        self.image_processor = image_processor or MoonViTImageProcessor(
            patch_size=config.patch_size,
            merge_kernel_size=config.merge_kernel_size,
        )

    @staticmethod
    def _sanitize_weights(weights):
        output = {}
        for key, value in weights.items():
            if key.startswith("mm_projector.proj.0."):
                key = key.replace("mm_projector.proj.0.", "mm_projector.linear_1.", 1)
            elif key.startswith("mm_projector.proj.2."):
                key = key.replace("mm_projector.proj.2.", "mm_projector.linear_2.", 1)
            if key == "vision_tower.patch_embed.proj.weight":
                # Torch OIHW -> MLX OHWI.
                if value.ndim == 4 and value.shape[1] == 3:
                    value = value.transpose(0, 2, 3, 1)
            output[key] = value
        return output

    @classmethod
    def from_pretrained(
        cls,
        projector_path,
        moonvit_weights_path,
        *,
        config_path=None,
        preprocessor_config_path=None,
        lazy=False,
    ):
        projector_path = Path(projector_path)
        if projector_path.is_dir():
            config_path = config_path or projector_path / "config.json"
            projector_path = projector_path / "mm_projector.safetensors"
        if config_path is None:
            raise ValueError("config_path is required when projector_path is a file.")
        with open(config_path, "r", encoding="utf-8") as handle:
            full_config = json.load(handle)
        vision_config = MoonViTConfig.from_dict(full_config["vision_config"])
        vision_config.text_hidden_size = full_config.get("text_config", {}).get(
            "hidden_size", vision_config.text_hidden_size
        )
        processor = (
            MoonViTImageProcessor.from_preprocessor_config(preprocessor_config_path)
            if preprocessor_config_path is not None
            else None
        )
        model = cls(
            vision_config,
            image_token_id=full_config.get(
                "media_placeholder_token_id", DEFAULT_IMAGE_TOKEN_ID
            ),
            image_processor=processor,
        )
        weights = {}
        weights.update(
            {
                key: value
                for key, value in mx.load(str(moonvit_weights_path)).items()
                if key.startswith("vision_tower.")
            }
        )
        weights.update(mx.load(str(projector_path)))
        weights = model._sanitize_weights(weights)
        model.load_weights(list(weights.items()), strict=True)
        if not lazy:
            mx.eval(model.parameters())
        return model

    def encode_images(self, pixel_values, grid_thws):
        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixels = pixel_values.transpose(0, 2, 3, 1).astype(target_dtype)
        features = self.vision_tower(pixels, grid_thws)
        projected = self.mm_projector(features)
        return mx.concatenate(projected, axis=0)

    def validate_model(self, language_model, tokenizer=None):
        embedding = language_model.model.embed_tokens
        embedding_width = getattr(
            getattr(language_model, "args", None),
            "hidden_size",
            getattr(embedding, "dims", embedding.weight.shape[-1]),
        )
        if embedding_width != self.config.text_hidden_size:
            raise ValueError(
                f"GLM5V projector output width is {self.config.text_hidden_size}, "
                f"but the language model embedding width is {embedding_width}."
            )
        if tokenizer is not None:
            token_id = tokenizer.convert_tokens_to_ids("<|image|>")
            if token_id != self.image_token_id:
                raise ValueError(
                    f"GLM5V expects <|image|> ID {self.image_token_id}, "
                    f"but the tokenizer uses {token_id}."
                )

    def prepare(
        self,
        language_model,
        input_ids,
        images,
    ):
        """Return prompt IDs and embeddings ready for ``mlx_lm.generate``."""
        self.validate_model(language_model)
        processed = self.image_processor(images)
        input_ids = expand_image_tokens(
            input_ids,
            processed["image_token_counts"],
            self.image_token_id,
        )
        image_embeddings = self.encode_images(
            processed["pixel_values"], processed["grid_thws"]
        )
        # Keep the 27-layer MoonViT graph out of the first language-model
        # prefill graph.  MLX arrays are lazy, so returning this unevaluated
        # would otherwise fuse image encoding with a very large GLM prefill.
        mx.eval(image_embeddings)
        text_embeddings = language_model.model.embed_tokens(input_ids[None])
        embeddings = insert_image_embeddings(
            input_ids,
            text_embeddings,
            image_embeddings,
            self.image_token_id,
        )
        # Materialize the quantized text lookup and replacement as a compact
        # (sequence, hidden) input buffer before generation starts.
        mx.eval(embeddings)
        return input_ids, embeddings[0]
