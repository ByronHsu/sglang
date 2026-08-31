"""Focused tests for the Transformers-5.8 GLM-5.3 image processor."""

import unittest

import torch

from sglang.srt.configs.glm5_next_processor import (
    _LEGACY_MEDIA_TEMPLATE,
    Glm5NextImageProcessorCompat,
    Glm5NextProcessorCompat,
    enable_image_chat_template,
    smart_resize,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGlm5NextProcessor(unittest.TestCase):
    def test_minimum_canvas_and_patch_layout(self):
        processor = Glm5NextImageProcessorCompat()
        output = processor(torch.zeros(3, 32, 32), return_tensors="pt")

        self.assertEqual(output["image_grid_thw"].tolist(), [[1, 8, 8]])
        self.assertEqual(tuple(output["pixel_values"].shape), (64, 1176))

    def test_uint8_rescaling_uses_explicit_processor_policy(self):
        processor = Glm5NextImageProcessorCompat(
            min_image_tokens=1,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
        )
        image = torch.ones(3, 28, 28, dtype=torch.uint8)

        rescaled = processor(image, return_tensors="pt")["pixel_values"]
        unscaled = processor(image, return_tensors="pt", do_rescale=False)[
            "pixel_values"
        ]

        torch.testing.assert_close(rescaled, torch.full_like(rescaled, 1 / 255))
        torch.testing.assert_close(unscaled, torch.ones_like(unscaled))

    def test_resize_preserves_aligned_aspect_ratio(self):
        height, width = smart_resize(400, 800)

        self.assertEqual(height % 28, 0)
        self.assertEqual(width % 28, 0)
        self.assertGreater(width, height)

    def test_image_placeholder_expansion(self):
        expanded = Glm5NextProcessorCompat._expand_image_tokens(
            "before<|image|>after", "<|image|>", [3]
        )

        self.assertEqual(expanded, "before<|image|><|image|><|image|>after")

    def test_legacy_chat_template_is_upgraded_for_images_only(self):
        tokenizer = type("Tokenizer", (), {"chat_template": _LEGACY_MEDIA_TEMPLATE})()

        enable_image_chat_template(tokenizer)

        self.assertIn(
            "<|begin_of_image|><|image|><|end_of_image|>", tokenizer.chat_template
        )
        self.assertIn("unable to process this", tokenizer.chat_template)
        self.assertIn("['video', 'video_url', 'audio'", tokenizer.chat_template)

    def test_replaced_processor_tokenizer_is_also_upgraded(self):
        processor = Glm5NextProcessorCompat.__new__(Glm5NextProcessorCompat)
        tokenizer = type("Tokenizer", (), {"chat_template": _LEGACY_MEDIA_TEMPLATE})()

        processor.tokenizer = tokenizer

        self.assertIn("<|begin_of_image|>", processor.tokenizer.chat_template)
        self.assertIn("<|begin_of_image|>", processor.chat_template)


if __name__ == "__main__":
    unittest.main()
