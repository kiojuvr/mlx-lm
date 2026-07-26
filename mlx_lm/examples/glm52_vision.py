"""Run one GLM-5.2 Vision image request.

Only the MoonViT shard and projector are additional to an existing MLX
GLM-5.2 model.  This script never downloads the full Baseten checkpoint.
"""

import argparse
import sys
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.glm5v import (
    DEFAULT_MOONVIT_SHARD,
    GLM5VisionAdapter,
    IMAGE_PLACEHOLDER,
    load_image_source,
)


DEFAULT_ALIS_MODEL = (
    Path.home()
    / ".lmstudio/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-4.5bpw"
)
DEFAULT_PROJECTOR = Path.home() / "models/glm52-vision-projector"


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _limit_processor_for_image(processor, image_size, max_image_tokens):
    """Reduce official MoonViT resize budgets to cap one image's tokens.

    Kimi's ``fixed_output_tokens`` only changes placeholder bookkeeping, so it
    cannot safely impose a feature limit. Instead, retain the official resize
    algorithm and find the largest ``in_patch_limit`` that produces no more
    than the requested number of merged tokens. The one-side patch limit is
    reduced only for an extreme aspect ratio that cannot meet the cap at the
    minimum input-patch budget.
    """
    if max_image_tokens <= 0:
        raise ValueError("max_image_tokens must be positive.")
    if processor.fixed_output_tokens is not None:
        if processor.fixed_output_tokens > max_image_tokens:
            raise ValueError(
                "Cannot cap a processor with fixed_output_tokens above the "
                "requested image-token limit."
            )
        return processor.resize_config(*image_size)

    original_patch_limit = max(1, int(processor.in_patch_limit))
    original_side_limit = max(1, int(processor.patch_limit_on_one_side))

    def config_for(patch_limit):
        processor.in_patch_limit = patch_limit
        return processor.resize_config(*image_size)

    if config_for(original_patch_limit)["num_tokens"] <= max_image_tokens:
        return config_for(original_patch_limit)

    # Very thin images can still have too many merged tokens at a one-patch
    # area budget because Kimi clamps each dimension's patch estimate to one.
    # Its independent one-side budget can safely constrain that case.
    if config_for(1)["num_tokens"] > max_image_tokens:
        low, high = 1, original_side_limit
        while low < high:
            midpoint = (low + high + 1) // 2
            processor.patch_limit_on_one_side = midpoint
            if config_for(1)["num_tokens"] <= max_image_tokens:
                low = midpoint
            else:
                high = midpoint - 1
        processor.patch_limit_on_one_side = low

    low, high = 1, original_patch_limit
    while low < high:
        midpoint = (low + high + 1) // 2
        if config_for(midpoint)["num_tokens"] <= max_image_tokens:
            low = midpoint
        else:
            high = midpoint - 1
    config = config_for(low)
    if config["num_tokens"] > max_image_tokens:
        raise ValueError(
            "The requested image-token limit cannot represent this image."
        )
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_ALIS_MODEL,
        type=Path,
        help="Local MLX GLM-5.2 directory",
    )
    parser.add_argument(
        "--projector",
        default=DEFAULT_PROJECTOR,
        type=Path,
        help="Projector directory",
    )
    parser.add_argument(
        "--moonvit",
        type=Path,
        help="Kimi-K2.6 vision shard (inferred below the projector by default)",
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--max-tokens", type=_positive_int, default=256)
    parser.add_argument(
        "--max-image-tokens",
        type=_positive_int,
        help=(
            "Reduce the official MoonViT resize budget until this image uses "
            "at most this many merged tokens. Recommended: 128 for a first "
            "full-model smoke test."
        ),
    )
    parser.add_argument(
        "--prefill-step-size",
        type=_positive_int,
        default=16,
        help="Language-model prefill chunk size (default: 16).",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable the Alis reasoning preamble for a short direct answer.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_path = args.model.expanduser()
    projector_dir = args.projector.expanduser()
    moonvit = (
        args.moonvit.expanduser()
        if args.moonvit
        else projector_dir / "moonvit" / DEFAULT_MOONVIT_SHARD
    )
    print(f"[glm5v] Loading language model: {model_path}", file=sys.stderr, flush=True)
    model, tokenizer = load(str(model_path))
    preprocessor = moonvit.parent / "preprocessor_config.json"
    print(
        f"[glm5v] Loading MoonViT and projector: {projector_dir}",
        file=sys.stderr,
        flush=True,
    )
    adapter = GLM5VisionAdapter.from_pretrained(
        projector_dir,
        moonvit,
        preprocessor_config_path=preprocessor if preprocessor.exists() else None,
    )
    adapter.validate_model(model, tokenizer)
    image = load_image_source(str(Path(args.image).expanduser()))
    resize_config = adapter.image_processor.resize_config(*image.size)
    if args.max_image_tokens is not None:
        resize_config = _limit_processor_for_image(
            adapter.image_processor,
            image.size,
            args.max_image_tokens,
        )
    print(
        "[glm5v] Image "
        f"{image.width}x{image.height} -> "
        f"{resize_config['new_width']}x{resize_config['new_height']}, "
        f"{resize_config['num_tokens']} merged tokens",
        file=sys.stderr,
        flush=True,
    )

    content = IMAGE_PLACEHOLDER + args.prompt
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        **({"enable_thinking": False} if args.no_thinking else {}),
    )
    input_ids = tokenizer.encode(rendered)
    prompt, embeddings = adapter.prepare(
        model,
        input_ids,
        image,
    )
    print(
        f"[glm5v] Vision encoded; language prompt has {len(prompt)} tokens",
        file=sys.stderr,
        flush=True,
    )

    last_progress = [-1]

    def report_prefill(processed, total):
        if processed == last_progress[0]:
            return
        last_progress[0] = processed
        print(
            f"[glm5v] Prefill {processed}/{total} tokens",
            file=sys.stderr,
            flush=True,
        )

    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        input_embeddings=embeddings,
        max_tokens=args.max_tokens,
        prefill_step_size=args.prefill_step_size,
        prompt_progress_callback=report_prefill,
        prompt_checkpoint=False,
        verbose=args.verbose,
    )
    if not args.verbose:
        print(output)


if __name__ == "__main__":
    main()
