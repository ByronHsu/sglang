import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_loader.weight_utils import sharded_weight_loader
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, loaded_weight):
        self.loaded = loaded_weight


class TestGlm5NextWeightLoading(unittest.TestCase):
    def test_linear_attention_loader_uses_explicit_shard_rank(self):
        param = torch.empty(2)
        loaded = torch.arange(6)

        sharded_weight_loader(0, lambda: 2)(param, loaded)

        torch.testing.assert_close(param, torch.tensor([4.0, 5.0]))

    @patch("sglang.srt.models.glm5_next.DeepseekV2WeightLoaderMixin.post_load_weights")
    def test_quark_block_fp8_weight_scale_loads_scale_inv(self, post_load):
        scale_param = _FakeParam()
        model = SimpleNamespace(
            config=SimpleNamespace(
                n_routed_experts=0,
                num_hidden_layers=45,
                num_nextn_predict_layers=1,
            ),
            num_fused_shared_experts=0,
            quant_config=None,
            named_parameters=lambda: iter(
                [("model.layers.0.mlp.down_proj.weight_scale_inv", scale_param)]
            ),
        )
        loaded_scale = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        Glm5NextForConditionalGeneration.load_weights(
            model,
            [
                (
                    "model.language_model.layers.0.mlp.down_proj.weight_scale",
                    loaded_scale,
                )
            ],
        )

        self.assertIs(scale_param.loaded, loaded_scale)
        post_load.assert_called_once()

    def test_shared_expert_fusion_is_disabled_for_ep8(self):
        config = SimpleNamespace(n_shared_experts=1)
        backend = SimpleNamespace(is_deepep=lambda: False)
        with (
            patch("sglang.srt.models.glm5_next._is_cuda", True),
            patch("sglang.srt.models.glm5_next._device_sm", 90),
            patch(
                "sglang.srt.models.glm5_next.get_moe_ep_group",
                return_value=SimpleNamespace(world_size=8),
            ),
            patch(
                "sglang.srt.models.glm5_next.get_moe_a2a_backend",
                return_value=backend,
            ),
        ):
            reason = (
                Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason(
                    config, None
                )
            )

        self.assertIn("expert parallelism", reason)


if __name__ == "__main__":
    unittest.main()
