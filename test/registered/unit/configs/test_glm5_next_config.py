"""Unit tests for GLM-5 Next configuration compatibility."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from sglang.srt.configs.glm5_next_processor import Glm5NextProcessorCompat
from sglang.srt.configs.model_config import ModelConfig, is_multimodal_model
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.multimodal.customized_mm_processor_utils import (
    _CUSTOMIZED_MM_PROCESSOR,
)
from sglang.srt.server_args import ServerArgs, auto_choose_speculative_params
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGlm5NextTextConfig(CustomTestCase):
    def test_vision_config_and_processor_are_registered_without_transformers_bump(self):
        config = Glm5NextConfig(
            architectures=["Glm5NextForConditionalGeneration"],
            vision_config={
                "hidden_size": 1024,
                "num_heads": 16,
                "projection_intermediate_size": 10240,
                "swiglu_limit": 10.0,
            },
        )

        self.assertIsInstance(config.vision_config, Glm5NextVisionConfig)
        self.assertEqual(config.vision_config.projection_intermediate_size, 10240)
        self.assertEqual(config.vision_config.swiglu_limit, 10.0)
        self.assertEqual(
            config.vision_config.to_dict()["model_type"], "glm5_next_vision"
        )
        self.assertIs(_CUSTOMIZED_MM_PROCESSOR["glm5_next"], Glm5NextProcessorCompat)
        self.assertTrue(is_multimodal_model(["Glm5NextForConditionalGeneration"]))

    def test_outer_config_uses_registered_text_config(self):
        config = Glm5NextConfig(
            architectures=["Glm5NextForConditionalGeneration"],
            text_config={
                "num_hidden_layers": 4,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "deepseek_sparse_attention",
                    "linear_attention",
                ],
            },
        )

        self.assertIsInstance(config.get_text_config(), Glm5NextTextConfig)
        self.assertEqual(config.text_config.linear_layer_ids, [0, 1, 3])
        self.assertEqual(config.text_config.full_attention_layer_ids, [2])
        self.assertEqual(config.text_config.nextn_layer_ids, [4])

    def test_model_runner_classifies_glm5_as_mambaish(self):
        config = Glm5NextConfig(
            architectures=["Glm5NextForConditionalGeneration"],
            text_config={"num_hidden_layers": 2},
        )
        runner = ModelRunner.__new__(ModelRunner)
        runner.model_config = SimpleNamespace(hf_config=config)
        runner.is_draft_worker = False

        self.assertIs(runner.mambaish_config, config.text_config)

    def test_draft_config_maps_to_native_nextn_architecture(self):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.is_draft_model = True
        model_config.hf_config = SimpleNamespace(
            architectures=["Glm5NextForConditionalGeneration"]
        )
        model_config.hf_text_config = SimpleNamespace(
            architectures=["Glm5NextForConditionalGeneration"],
            linear_attn_config={"kda_layers": [0]},
        )

        model_config._config_draft_model()

        expected = ["Glm5NextForConditionalGenerationNextN"]
        self.assertEqual(model_config.hf_config.architectures, expected)
        self.assertEqual(model_config.hf_text_config.architectures, expected)
        self.assertEqual(model_config.hf_text_config.num_nextn_predict_layers, 1)
        self.assertIsNone(model_config.hf_text_config.linear_attn_config)

    def test_native_nextn_defaults_are_tuned_for_glm5(self):
        args = SimpleNamespace(
            speculative_algorithm="NEXTN",
            get_model_config=lambda: SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["Glm5NextForConditionalGeneration"]
                )
            ),
        )

        self.assertEqual(auto_choose_speculative_params(args), (5, 1, 6))

    def test_auto_mamba_strategy_uses_extra_buffer(self):
        args = ServerArgs(model_path="dummy", mamba_scheduler_strategy="auto")
        args._handle_missing_default_values()

        args._resolve_glm5_next_mamba_scheduler_strategy(
            "Glm5NextForConditionalGeneration"
        )

        self.assertEqual(args.mamba_scheduler_strategy, "extra_buffer")

    def test_explicit_mamba_strategy_is_preserved(self):
        args = ServerArgs(model_path="dummy", mamba_scheduler_strategy="no_buffer")
        args._handle_missing_default_values()

        args._resolve_glm5_next_mamba_scheduler_strategy(
            "Glm5NextForConditionalGeneration"
        )

        self.assertEqual(args.mamba_scheduler_strategy, "no_buffer")

    def test_transformers_format_builds_legacy_linear_attention_config(self):
        config = Glm5NextTextConfig(
            num_hidden_layers=4,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "deepseek_sparse_attention",
                "linear_attention",
            ],
            linear_head_dim=96,
            linear_num_heads=32,
            linear_conv_kernel_dim=3,
            gate_lower_bound=-5.0,
        )

        self.assertEqual(
            config.linear_attn_config,
            {
                "full_attn_layers": [2],
                "head_dim": 96,
                "kda_layers": [0, 1, 3],
                "num_heads": 32,
                "short_conv_kernel_size": 3,
                "gate_lower_bound": -5.0,
            },
        )
        self.assertEqual(config.linear_layer_ids, [0, 1, 3])
        self.assertEqual(config.full_attention_layer_ids, [2])

    def test_transformers_format_without_lower_bound(self):
        config = Glm5NextTextConfig(
            num_hidden_layers=2,
            layer_types=["linear_attention", "deepseek_sparse_attention"],
        )

        self.assertIsNone(config.linear_attn_config["gate_lower_bound"])

    def test_legacy_linear_lower_bound_is_normalized(self):
        config = Glm5NextTextConfig(
            num_hidden_layers=2,
            layer_types=["linear_attention", "deepseek_sparse_attention"],
            linear_lower_bound=-4.0,
        )

        self.assertEqual(config.linear_attn_config["gate_lower_bound"], -4.0)

    def test_gate_lower_bound_takes_precedence(self):
        config = Glm5NextTextConfig(
            num_hidden_layers=2,
            layer_types=["linear_attention", "deepseek_sparse_attention"],
            linear_lower_bound=-4.0,
            gate_lower_bound=-5.0,
        )

        self.assertEqual(config.linear_attn_config["gate_lower_bound"], -5.0)

    def test_legacy_linear_attention_config_takes_precedence(self):
        linear_attn_config = {
            "full_attn_layers": [1],
            "head_dim": 64,
            "kda_layers": [0],
            "num_heads": 16,
            "short_conv_kernel_size": 2,
            "gate_lower_bound": -4.0,
        }

        config = Glm5NextTextConfig(
            num_hidden_layers=2,
            linear_attn_config=linear_attn_config,
            layer_types=["deepseek_sparse_attention", "linear_attention"],
            linear_head_dim=96,
            linear_num_heads=32,
            linear_conv_kernel_dim=3,
            gate_lower_bound=-5.0,
        )

        self.assertIs(config.linear_attn_config, linear_attn_config)


if __name__ == "__main__":
    unittest.main()
