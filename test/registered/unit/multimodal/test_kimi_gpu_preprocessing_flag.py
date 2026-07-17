import unittest
from unittest.mock import patch

import torch
from PIL import Image

from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin
from sglang.srt.multimodal.processors.kimi_k25 import KimiGPUProcessorWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeMediaProcessor:
    @staticmethod
    def media_tokens_calculator(media):
        assert isinstance(media["image"], Image.Image)
        return 7


class _FakeHFProcessor:
    image_processor = object()
    tokenizer = object()
    media_processor = _FakeMediaProcessor()

    def __call__(self, text, **kwargs):
        assert isinstance(kwargs["medias"][0]["image"], Image.Image)
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "grid_thws": torch.tensor([[1, 2, 3]]),
        }


class _KimiTokenCountResolver(KimiGridMMDataMixin):
    def __init__(self, processor):
        self._processor = processor


class TestKimiGPUPreprocessingFlag(unittest.TestCase):
    def test_cpu_fallback_preserves_token_id_expansion_path(self):
        wrapper = KimiGPUProcessorWrapper(
            _FakeHFProcessor(),
            image_token="<|media_pad|>",
            patch_size=14,
            merge_kernel_size=2,
            in_patch_limit=4096,
            patch_limit_on_one_side=256,
            fixed_output_tokens=None,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
        image = Image.new("RGB", (32, 32))

        with patch(
            "sglang.srt.multimodal.processors.kimi_k25._ENABLE_GPU_IMAGE_PREPROCESSING",
            False,
        ), patch.object(torch.cuda, "is_available", return_value=True):
            output = wrapper(text="look <|media_pad|>", images=[image])

        self.assertEqual(output["image_grid_thw"].tolist(), [[1, 2, 3]])
        resolver = _KimiTokenCountResolver(wrapper)
        self.assertEqual(resolver.resolve_image_token_counts([image]), [7])


if __name__ == "__main__":
    unittest.main()
