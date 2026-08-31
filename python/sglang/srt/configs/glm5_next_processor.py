"""Transformers-5.8-compatible image processor for GLM-5.3."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers.utils.hub import cached_file

_LEGACY_MEDIA_TEMPLATE = """{%- elif item is mapping and item.type in ['image', 'image_url', 'video', 'video_url', 'audio', 'audio_url', 'input_audio'] -%}
                {%- set media_type = item.type | replace('_url', '') | replace('input_', '') -%}
                {{- "<reminder>You are unable to process this " ~ media_type ~ " because you don't have multi-modal input ability. Try different methods.</reminder>" }}"""

_IMAGE_CAPABLE_MEDIA_TEMPLATE = """{%- elif item is mapping and item.type in ['image', 'image_url'] -%}
                {{- "<|begin_of_image|><|image|><|end_of_image|>" }}
            {%- elif item is mapping and item.type in ['video', 'video_url', 'audio', 'audio_url', 'input_audio'] -%}
                {%- set media_type = item.type | replace('_url', '') | replace('input_', '') -%}
                {{- "<reminder>You are unable to process this " ~ media_type ~ " because you don't have multi-modal input ability. Try different methods.</reminder>" }}"""


def enable_image_chat_template(tokenizer) -> None:
    """Upgrade the legacy text-only media branch shipped by early checkpoints."""
    template = getattr(tokenizer, "chat_template", None)
    if not template or "<|begin_of_image|><|image|><|end_of_image|>" in template:
        return
    tokenizer.chat_template = template.replace(
        _LEGACY_MEDIA_TEMPLATE, _IMAGE_CAPABLE_MEDIA_TEMPLATE
    )


def smart_resize(
    height: int,
    width: int,
    *,
    num_frames: int = 2,
    temporal_factor: int = 2,
    factor: int = 28,
    min_image_tokens: int = 16,
    max_image_tokens: int = 8000,
) -> tuple[int, int]:
    pixels_per_token = temporal_factor * factor**2
    min_pixels = min_image_tokens * pixels_per_token
    max_pixels = max_image_tokens * pixels_per_token
    aligned_frames = max(
        temporal_factor, round(num_frames / temporal_factor) * temporal_factor
    )

    def align(value: int) -> int:
        return math.ceil(value / factor) * factor

    def fit_within_budget() -> tuple[int, int]:
        low, high = 1, height
        best_height = best_width = factor
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate_height = align(content_height)
            candidate_width = align(content_width)
            if aligned_frames * candidate_height * candidate_width <= max_pixels:
                best_height, best_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
        return best_height, best_width

    target_height, target_width = align(height), align(width)
    pixel_budget = aligned_frames * target_height * target_width
    if pixel_budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        target_height = align(max(1, math.ceil(height * scale)))
        target_width = align(max(1, math.ceil(width * scale)))
        pixel_budget = aligned_frames * target_height * target_width
    if pixel_budget > max_pixels:
        target_height, target_width = fit_within_budget()
    return target_height, target_width


class Glm5NextImageProcessorCompat(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        patch_expand_factor: int = 1,
        min_image_tokens: int = 16,
        max_image_tokens: int = 8000,
        image_mean=None,
        image_std=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.patch_expand_factor = patch_expand_factor
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.image_mean = image_mean or OPENAI_CLIP_MEAN
        self.image_std = image_std or OPENAI_CLIP_STD

    @staticmethod
    def _to_chw(image) -> torch.Tensor:
        from PIL import Image
        from torchvision.transforms import functional as TF

        if isinstance(image, Image.Image):
            tensor = TF.pil_to_tensor(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            tensor = image
        else:
            raise TypeError(f"Unsupported GLM-5.3 image type: {type(image).__name__}")
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        if tensor.ndim != 3:
            raise ValueError(
                f"Expected a 3D image tensor, got shape {tuple(tensor.shape)}"
            )
        tensor = tensor[:3].to(torch.float32)
        if tensor.numel() and tensor.max() > 1:
            tensor = tensor / 255.0
        return tensor

    def _preprocess_one(self, image) -> tuple[torch.Tensor, list[int]]:
        from torchvision.transforms import functional as TF

        image = self._to_chw(image)
        _, height, width = image.shape
        factor = self.patch_size * self.merge_size * self.patch_expand_factor
        target_height, target_width = smart_resize(
            height,
            width,
            num_frames=self.temporal_patch_size,
            temporal_factor=self.temporal_patch_size,
            factor=factor,
            min_image_tokens=self.min_image_tokens,
            max_image_tokens=self.max_image_tokens,
        )
        pixels_per_token = self.temporal_patch_size * factor**2
        scale = min(target_height / height, target_width / width)
        if self.temporal_patch_size * height * width >= (
            pixels_per_token * self.min_image_tokens
        ):
            scale = min(1.0, scale)
        content_height = max(1, min(target_height, math.floor(height * scale)))
        content_width = max(1, min(target_width, math.floor(width * scale)))
        if (content_height, content_width) != (height, width):
            image = TF.resize(
                image,
                [content_height, content_width],
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True,
            )
        image = torch.nn.functional.pad(
            image,
            (0, target_width - content_width, 0, target_height - content_height),
        )
        mean = image.new_tensor(self.image_mean)[:, None, None]
        std = image.new_tensor(self.image_std)[:, None, None]
        image = (image - mean) / std

        channels, resized_height, resized_width = image.shape
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size
        patches = image.reshape(
            1,
            channels,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        ).permute(0, 2, 5, 3, 6, 1, 4, 7)
        patches = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, self.temporal_patch_size, -1, -1)
            .reshape(
                grid_h * grid_w,
                channels * self.temporal_patch_size * self.patch_size**2,
            )
        )
        return patches, [1, grid_h, grid_w]

    def preprocess(self, images, return_tensors=None, **kwargs) -> BatchFeature:
        if not isinstance(images, (list, tuple)):
            images = [images]
        processed = [self._preprocess_one(image) for image in images]
        return BatchFeature(
            {
                "pixel_values": torch.cat([item[0] for item in processed]),
                "image_grid_thw": torch.tensor(
                    [item[1] for item in processed], dtype=torch.long
                ),
            },
            tensor_type=return_tensors,
        )

    __call__ = preprocess


class Glm5NextProcessorCompat(ProcessorMixin):
    """Minimal GLM-5.3 image/text processor for Transformers versions before 5.16."""

    attributes = ["image_processor", "tokenizer"]

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        enable_image_chat_template(tokenizer)
        self._tokenizer = tokenizer
        self.chat_template = tokenizer.chat_template

    def __init__(self, image_processor, tokenizer, chat_template=None):
        self.image_token = getattr(tokenizer, "image_token", "<|image|>")
        self.video_token = getattr(tokenizer, "video_token", "<|video|>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        self.chat_template = self.tokenizer.chat_template

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        revision = kwargs.pop("revision", None)
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        use_fast = kwargs.pop("use_fast", True)
        config_path = cached_file(
            model_path, "processor_config.json", revision=revision
        )
        with Path(config_path).open() as file:
            processor_config = json.load(file)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )
        return cls(
            image_processor=Glm5NextImageProcessorCompat(
                **processor_config.get("image_processor", {})
            ),
            tokenizer=tokenizer,
            chat_template=getattr(tokenizer, "chat_template", None),
        )

    @staticmethod
    def _expand_image_tokens(text: str, token: str, token_counts: list[int]) -> str:
        parts = text.split(token)
        if len(parts) - 1 != len(token_counts):
            raise ValueError(
                f"Prompt contains {len(parts) - 1} image placeholders for "
                f"{len(token_counts)} images"
            )
        output = parts[0]
        for count, suffix in zip(token_counts, parts[1:]):
            output += token * count + suffix
        return output

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        return_tensors=None,
        padding=False,
        images_kwargs=None,
        **kwargs,
    ) -> BatchFeature:
        if videos is not None:
            raise NotImplementedError(
                "GLM-5.3 video preprocessing requires a newer Transformers stack"
            )
        image_inputs = None
        if images is not None:
            image_inputs = self.image_processor(
                images, return_tensors=return_tensors, **(images_kwargs or {})
            )
        texts = [text] if isinstance(text, str) else list(text or [])
        if image_inputs is not None:
            if len(texts) != 1:
                raise ValueError("GLM-5.3 compatibility processing expects one prompt")
            merge_length = self.image_processor.merge_size**2
            counts = [
                int(grid.prod().item() // merge_length)
                for grid in image_inputs["image_grid_thw"]
            ]
            texts[0] = self._expand_image_tokens(texts[0], self.image_token, counts)
        outputs = self.tokenizer(
            texts,
            padding=padding,
            return_tensors=return_tensors,
            return_token_type_ids=False,
        )
        if image_inputs is not None:
            outputs.update(image_inputs)
        return BatchFeature(outputs)
