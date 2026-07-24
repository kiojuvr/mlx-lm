# Copyright © 2026 Apple Inc.

import base64
import io
import os
import types
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image

from mlx_lm.examples.glm52_vision import _limit_processor_for_image
from mlx_lm.generate import generate_step
from mlx_lm.glm5v import (
    _bicubic_interpolate,
    DEFAULT_IMAGE_TOKEN_ID,
    GLM5VisionAdapter,
    IMAGE_PLACEHOLDER,
    MoonViTConfig,
    MoonViTImageProcessor,
    PatchMergerProjector,
    expand_image_tokens,
    insert_image_embeddings,
    load_image_source,
    load_image_sources,
    normalize_vision_messages,
    patch_merge,
)


def tiny_glm_model():
    from mlx_lm.models import glm_moe_dsa

    args = glm_moe_dsa.ModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=1024,
        hidden_size=128,
        index_head_dim=16,
        index_n_heads=4,
        index_topk=4,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=2.5,
        kv_lora_rank=64,
        q_lora_rank=24,
        qk_rope_head_dim=16,
        v_head_dim=32,
        qk_nope_head_dim=16,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0},
        attention_bias=False,
        index_topk_pattern="FS",
    )
    return glm_moe_dsa.Model(args)


class TestGLM5Vision(unittest.TestCase):
    def test_example_image_token_limit_uses_official_resize_budgets(self):
        processor = MoonViTImageProcessor()
        original = processor.resize_config(1600, 800)
        limited = _limit_processor_for_image(processor, (1600, 800), 128)

        self.assertEqual(original["num_tokens"], 1682)
        self.assertLessEqual(limited["num_tokens"], 128)
        self.assertEqual(limited["num_tokens"], 128)
        self.assertEqual(
            limited,
            processor.resize_config(1600, 800),
        )
        self.assertLess(processor.in_patch_limit, 16384)
        self.assertEqual(processor.patch_limit_on_one_side, 512)

    def test_example_image_token_limit_handles_extreme_aspect_ratio(self):
        processor = MoonViTImageProcessor()
        limited = _limit_processor_for_image(processor, (10, 10_000), 2)

        self.assertLessEqual(limited["num_tokens"], 2)
        self.assertLess(processor.patch_limit_on_one_side, 512)
        with self.assertRaisesRegex(ValueError, "positive"):
            _limit_processor_for_image(processor, (10, 10_000), 0)

    @unittest.skipUnless(
        os.environ.get("GLM5V_TEST_ALIS_MODEL"),
        "set GLM5V_TEST_ALIS_MODEL to run the local Alis tokenizer fixture",
    )
    def test_local_alis_template_preserves_image_and_responses_text(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            os.environ["GLM5V_TEST_ALIS_MODEL"],
            local_files_only=True,
        )
        self.assertEqual(
            tokenizer.encode(IMAGE_PLACEHOLDER, add_special_tokens=False),
            [154830, 154854, 154831],
        )
        messages, sources = normalize_vision_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,example",
                        },
                        {"type": "input_text", "text": "after"},
                    ],
                }
            ]
        )
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertEqual(sources, ["data:image/png;base64,example"])
        self.assertIn(IMAGE_PLACEHOLDER, rendered)
        self.assertIn("after", rendered)
        self.assertNotIn("unable to process", rendered)

    @unittest.skipUnless(
        os.environ.get("GLM5V_TEST_PROJECTOR_DIR"),
        "set GLM5V_TEST_PROJECTOR_DIR to run the real-weight parity fixture",
    )
    def test_real_weights_match_official_pytorch_golden(self):
        """Compare a real-weight MLX result to an offline Transformers oracle."""
        root = Path(os.environ["GLM5V_TEST_PROJECTOR_DIR"]).expanduser()
        adapter = GLM5VisionAdapter.from_pretrained(
            root,
            root / "moonvit" / "model-00064-of-000064.safetensors",
            preprocessor_config_path=root / "moonvit" / "preprocessor_config.json",
        )
        adapter.set_dtype(mx.float32)
        pixels = (
            (np.arange(28 * 28 * 3, dtype=np.uint32) * 37 + 11) % 256
        ).astype(np.uint8)
        processed = adapter.image_processor(
            Image.fromarray(pixels.reshape(28, 28, 3))
        )
        projected = adapter.encode_images(
            processed["pixel_values"],
            processed["grid_thws"],
        )
        mx.eval(projected)

        # Transformers 5.14.1 Kimi_K25VisionModel + native conversion map,
        # evaluated in FP32 with the same real MoonViT/projector weights.
        expected = mx.array(
            [
                -0.0507267490029335,
                0.3582862913608551,
                0.1883070468902588,
                0.9183377623558044,
                0.37444373965263367,
                0.4423174262046814,
                0.4910241961479187,
                -0.6691057085990906,
                -0.12120501697063446,
                -0.4308021664619446,
                0.43481630086898804,
                0.16306021809577942,
                0.8697333931922913,
                -0.12727336585521698,
                -0.7894905805587769,
                0.08204254508018494,
            ]
        )
        self.assertEqual(projected.shape, (1, 6144))
        self.assertTrue(
            mx.allclose(
                projected[0, : expected.size],
                expected,
                rtol=1e-5,
                atol=2e-5,
            )
        )

    def test_bicubic_border_matches_pytorch_align_corners_false(self):
        values = mx.array([[[[0.0, 1.0]]]])
        resized = _bicubic_interpolate(values, (1, 4))
        expected = mx.array(
            [[[[-0.10546875, 0.2265625, 0.7734375, 1.10546875]]]]
        )
        self.assertTrue(mx.allclose(resized, expected, atol=1e-7))

    def test_structured_images_become_ordered_text_placeholders(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,first"},
                    },
                    {"type": "input_text", "text": "middle"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,second",
                    },
                    {"type": "text", "text": "after"},
                ],
            }
        ]
        normalized, sources = normalize_vision_messages(messages)

        self.assertEqual(
            sources,
            [
                "data:image/png;base64,first",
                "data:image/png;base64,second",
            ],
        )
        self.assertEqual(
            [part["text"] for part in normalized[0]["content"]],
            ["before", IMAGE_PLACEHOLDER, "middle", IMAGE_PLACEHOLDER, "after"],
        )
        self.assertEqual(
            [part["type"] for part in normalized[0]["content"]],
            ["text", "text", "text", "text", "text"],
        )
        self.assertEqual(messages[0]["content"][1]["type"], "image_url")

    def test_data_url_image_loading_and_byte_limit(self):
        image = Image.new("RGB", (2, 3), (12, 34, 56))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        source = f"data:image/png;base64,{encoded}"

        loaded = load_image_source(source, max_bytes=len(buffer.getvalue()))
        self.assertEqual(loaded.size, (2, 3))
        self.assertEqual(loaded.getpixel((0, 0)), (12, 34, 56))
        with self.assertRaisesRegex(ValueError, "limit"):
            load_image_source(source, max_bytes=len(buffer.getvalue()) - 1)
        with self.assertRaisesRegex(ValueError, "pixel limit"):
            load_image_source(source, max_pixels=5)
        with self.assertRaisesRegex(ValueError, "Malformed"):
            load_image_source("data:image/png;base64,not-valid!")
        with self.assertRaisesRegex(ValueError, "not supported"):
            load_image_source("https://example.com/image.png")

    def test_request_image_count_and_aggregate_pixel_limits(self):
        images = [
            Image.new("RGB", (2, 3), (12, 34, 56)),
            Image.new("RGB", (2, 3), (65, 43, 21)),
        ]

        with self.assertRaisesRegex(ValueError, "2 images"):
            load_image_sources(images, max_images=1)
        with self.assertRaisesRegex(ValueError, "aggregate limit"):
            load_image_sources(images, max_total_pixels=11)
        loaded = load_image_sources(
            images,
            max_images=2,
            max_pixels=6,
            max_total_pixels=12,
        )
        self.assertEqual([image.size for image in loaded], [(2, 3), (2, 3)])

    def test_model_validation_uses_logical_quantized_embedding_width(self):
        config = MoonViTConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_hidden_layers=1,
            intermediate_size=16,
            text_hidden_size=128,
        )
        adapter = GLM5VisionAdapter(config)
        packed_embedding = types.SimpleNamespace(
            dims=128,
            weight=mx.zeros((1024, 24), dtype=mx.uint32),
        )
        language_model = types.SimpleNamespace(
            args=types.SimpleNamespace(hidden_size=128),
            model=types.SimpleNamespace(embed_tokens=packed_embedding),
        )

        adapter.validate_model(language_model)
        language_model.args.hidden_size = 64
        with self.assertRaisesRegex(ValueError, "embedding width is 64"):
            adapter.validate_model(language_model)

    def test_projector_architecture_and_parameter_count(self):
        config = MoonViTConfig()
        projector = PatchMergerProjector(config)
        parameters = dict(projector.parameters())

        self.assertEqual(parameters["pre_norm"]["weight"].shape, (1152,))
        self.assertEqual(parameters["linear_1"]["weight"].shape, (4608, 4608))
        self.assertEqual(parameters["linear_2"]["weight"].shape, (6144, 4608))
        count = sum(value.size for _, value in tree_flatten(projector.parameters()))
        self.assertEqual(count, 49_558_272)

    def test_patch_merge_preserves_official_spatial_order(self):
        hidden = mx.arange(16, dtype=mx.float32).reshape(16, 1)
        merged = patch_merge(hidden, mx.array([[1, 4, 4]]), (2, 2))[0]
        expected = mx.array(
            [
                [[0], [1], [4], [5]],
                [[2], [3], [6], [7]],
                [[8], [9], [12], [13]],
                [[10], [11], [14], [15]],
            ],
            dtype=mx.float32,
        )
        self.assertTrue(mx.array_equal(merged, expected))

    def test_image_processor_matches_navit_token_math(self):
        processor = MoonViTImageProcessor(
            patch_size=14,
            merge_kernel_size=(2, 2),
        )
        image = Image.fromarray(np.zeros((57, 43, 3), dtype=np.uint8))
        output = processor(image)

        self.assertEqual(output["grid_thws"].tolist(), [[1, 6, 4]])
        self.assertEqual(output["pixel_values"].shape, (24, 3, 14, 14))
        self.assertEqual(output["image_token_counts"], [6])

    def test_placeholder_expansion_and_embedding_insertion(self):
        image_token = DEFAULT_IMAGE_TOKEN_ID
        ids = expand_image_tokens([10, image_token, 11], [2], image_token)
        self.assertEqual(ids.tolist(), [10, image_token, image_token, 11])

        text = mx.zeros((1, 4, 3))
        image = mx.array([[1, 2, 3], [4, 5, 6]])
        merged = insert_image_embeddings(ids, text, image, image_token)
        self.assertEqual(
            merged.tolist(),
            [[[0, 0, 0], [1, 2, 3], [4, 5, 6], [0, 0, 0]]],
        )

        with self.assertRaisesRegex(ValueError, "mismatch"):
            insert_image_embeddings(ids, text, image[:1], image_token)
        self.assertEqual(
            expand_image_tokens(
                [image_token, 9, image_token, image_token, image_token],
                [1, 3],
                image_token,
            ).tolist(),
            [image_token, 9, image_token, image_token, image_token],
        )
        with self.assertRaisesRegex(ValueError, r"found \[2, 2\]"):
            expand_image_tokens(
                [
                    image_token,
                    image_token,
                    9,
                    image_token,
                    image_token,
                ],
                [1, 3],
                image_token,
            )

    def test_weight_key_mapping(self):
        weights = {
            "mm_projector.proj.0.weight": mx.zeros((4, 4)),
            "mm_projector.proj.2.bias": mx.zeros((8,)),
            "vision_tower.patch_embed.proj.weight": mx.zeros((8, 3, 2, 2)),
        }
        mapped = GLM5VisionAdapter._sanitize_weights(weights)
        self.assertIn("mm_projector.linear_1.weight", mapped)
        self.assertIn("mm_projector.linear_2.bias", mapped)
        self.assertEqual(
            mapped["vision_tower.patch_embed.proj.weight"].shape,
            (8, 2, 2, 3),
        )

    def test_single_image_request_reaches_glm_decode(self):
        vision_config = MoonViTConfig(
            patch_size=2,
            init_pos_emb_height=2,
            init_pos_emb_width=2,
            init_pos_emb_time=1,
            num_attention_heads=2,
            num_hidden_layers=1,
            hidden_size=8,
            intermediate_size=16,
            merge_kernel_size=(2, 2),
            text_hidden_size=128,
        )
        image_token_id = 7
        adapter = GLM5VisionAdapter(
            vision_config,
            image_token_id=image_token_id,
        )
        language_model = tiny_glm_model()
        mx.eval(adapter.parameters(), language_model.parameters())

        image = Image.fromarray(np.full((4, 4, 3), 127, dtype=np.uint8))
        prompt, embeddings = adapter.prepare(
            language_model,
            [1, image_token_id, 2],
            image,
        )
        self.assertEqual(prompt.shape, (3,))
        self.assertEqual(embeddings.shape, (3, 128))

        generator = generate_step(
            prompt,
            language_model,
            input_embeddings=embeddings,
            max_tokens=1,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            prompt_checkpoint=False,
        )
        token, _ = next(generator)
        self.assertIsInstance(token, int)
        self.assertGreaterEqual(token, 0)

    def test_prepare_materializes_bfloat16_quantized_embeddings(self):
        from unittest import mock

        vision_config = MoonViTConfig(
            patch_size=2,
            init_pos_emb_height=2,
            init_pos_emb_width=2,
            init_pos_emb_time=1,
            num_attention_heads=2,
            num_hidden_layers=1,
            hidden_size=8,
            intermediate_size=16,
            merge_kernel_size=(2, 2),
            text_hidden_size=128,
        )
        image_token_id = 7
        adapter = GLM5VisionAdapter(
            vision_config,
            image_token_id=image_token_id,
        )
        adapter.set_dtype(mx.bfloat16)
        language_model = tiny_glm_model()
        quantized_embedding = nn.QuantizedEmbedding.from_embedding(
            language_model.model.embed_tokens,
            group_size=64,
            bits=6,
        )
        quantized_embedding.scales = quantized_embedding.scales.astype(mx.bfloat16)
        quantized_embedding.biases = quantized_embedding.biases.astype(mx.bfloat16)
        language_model.model.embed_tokens = quantized_embedding
        mx.eval(adapter.parameters(), quantized_embedding.parameters())

        image = Image.fromarray(np.full((4, 4, 3), 127, dtype=np.uint8))
        with mock.patch("mlx_lm.glm5v.mx.eval", wraps=mx.eval) as eval_mock:
            prompt, embeddings = adapter.prepare(
                language_model,
                [1, image_token_id, 2],
                image,
            )

        evaluated_shapes = [
            call.args[0].shape for call in eval_mock.call_args_list if call.args
        ]
        self.assertEqual(evaluated_shapes, [(1, 128), (1, 3, 128)])
        self.assertEqual(prompt.shape, (3,))
        self.assertEqual(embeddings.shape, (3, 128))
        self.assertEqual(embeddings.dtype, mx.bfloat16)
        self.assertTrue(bool(mx.all(mx.isfinite(embeddings)).item()))


if __name__ == "__main__":
    unittest.main()
