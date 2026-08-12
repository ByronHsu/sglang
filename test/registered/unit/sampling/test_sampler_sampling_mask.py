"""Unit tests for faithful, bounded sampling-support capture."""

import math
import unittest

import torch

from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    SamplingMaskStatus,
)
from sglang.srt.layers.sampler import (
    Sampler,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_logprob_dict_from_result,
    get_logprob_from_pp_outputs,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class TestSamplingMaskCapture(unittest.TestCase):
    def setUp(self):
        self.sampler = Sampler.__new__(Sampler)
        torch.nn.Module.__init__(self.sampler)
        self.sampler.sampling_mask_max_tokens = 3
        self.sampler.tp_sync_group = None
        self.sampler.cp_sync_group = None

    def _capture(self, weights, sampled_tokens, token_ids=None, batch_indices=None):
        if batch_indices is None:
            batch_indices = torch.arange(weights.shape[0])
        return self.sampler._build_sampling_mask_output(
            torch.tensor(sampled_tokens),
            (batch_indices, weights, token_ids),
        )

    def test_returns_complete_support_and_selected_logprob(self):
        output = self._capture(torch.tensor([[0.4, 0.3, 0.2, 0.0]]), sampled_tokens=[1])

        length = int(output.lengths[0])
        self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.OK])
        self.assertEqual(set(output.token_ids[0, :length].tolist()), {0, 1, 2})
        self.assertAlmostEqual(
            float(output.selected_logprobs[0]), math.log(0.3 / 0.9), places=6
        )

    def test_support_above_cap_is_explicit_overflow(self):
        output = self._capture(torch.tensor([[0.4, 0.3, 0.2, 0.1]]), sampled_tokens=[0])

        self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.OVERFLOW])
        self.assertEqual(output.token_ids.tolist(), [[0, 0, 0]])

    def test_sampled_token_is_not_added_to_support(self):
        output = self._capture(torch.tensor([[0.6, 0.4, 0.0]]), sampled_tokens=[2])

        self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.INVALID])
        self.assertEqual(output.token_ids.tolist(), [[0, 0, 0]])

    def test_token_id_mapping_uses_sampler_order(self):
        output = self._capture(
            weights=torch.tensor([[0.7, 0.3, 0.0, 0.0]]),
            token_ids=torch.tensor([[3, 1, 2, 0]], dtype=torch.int32),
            sampled_tokens=[1],
        )

        length = int(output.lengths[0])
        self.assertEqual(set(output.token_ids[0, :length].tolist()), {1, 3})
        self.assertAlmostEqual(float(output.selected_logprobs[0]), math.log(0.3))

    def test_only_requested_rows_are_captured(self):
        output = self._capture(
            weights=torch.tensor([[0.7, 0.3, 0.0]]),
            sampled_tokens=[2, 0, 1],
            batch_indices=torch.tensor([1]),
        )

        self.assertEqual(output.batch_indices.tolist(), [1])
        self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.OK])

    def test_greedy_support_is_singleton(self):
        output = self.sampler._build_greedy_sampling_mask_output(
            torch.tensor([0, 2]), torch.tensor([4, 5, 6])
        )

        self.assertEqual(output.batch_indices.tolist(), [0, 2])
        self.assertEqual(output.token_ids.tolist(), [[4], [6]])
        self.assertEqual(output.lengths.tolist(), [1, 1])
        self.assertEqual(output.selected_logprobs.tolist(), [0.0, 0.0])

    def test_materialization_preserves_batch_rows_and_status(self):
        sampling_output = self._capture(
            weights=torch.tensor(
                [
                    [0.6, 0.4, 0.0, 0.0],
                    [0.4, 0.3, 0.2, 0.1],
                ]
            ),
            sampled_tokens=[1, 9, 0],
            batch_indices=torch.tensor([0, 2]),
        )
        output = LogitsProcessorOutput(
            next_token_logits=None, sampling_mask_output=sampling_output
        )

        SchedulerBatchResultProcessor.materialize_sampling_mask_output(3, output)

        self.assertEqual(set(output.next_token_sampling_mask_idx[0]), {0, 1})
        self.assertEqual(output.next_token_sampling_mask_idx[1:], [None, None])
        self.assertEqual(
            output.next_token_sampling_mask_status,
            [SamplingMaskStatus.OK, None, SamplingMaskStatus.OVERFLOW],
        )
        self.assertIsNone(output.sampling_mask_output)

    def test_pipeline_parallel_round_trip_preserves_tensor_output(self):
        sampling_output = self._capture(
            weights=torch.tensor([[0.6, 0.4, 0.0]]), sampled_tokens=[0]
        )
        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(
                next_token_logits=None, sampling_mask_output=sampling_output
            )
        )

        received, _, _ = get_logprob_from_pp_outputs(
            PPProxyTensors(get_logprob_dict_from_result(result))
        )

        self.assertIsNotNone(received.sampling_mask_output)
        self.assertTrue(
            torch.equal(
                received.sampling_mask_output.token_ids, sampling_output.token_ids
            )
        )
        self.assertTrue(
            torch.equal(
                received.sampling_mask_output.statuses, sampling_output.statuses
            )
        )

    def test_pytorch_capture_reuses_actual_filtered_weights(self):
        probs = torch.tensor([[0.30, 0.20, 0.18, 0.17, 0.15]])
        _, filtered, token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
            probs,
            top_ks=torch.tensor([2]),
            top_ps=torch.tensor([0.5]),
            min_ps=torch.tensor([0.0]),
            need_min_p_sampling=False,
            sampling_seed=None,
            positions=torch.tensor([0]),
            return_filtered_probs=True,
        )

        kept_ids = token_ids[filtered > 0].tolist()
        self.assertEqual(set(kept_ids), {0, 1})

    def test_pytorch_capture_supports_top_p_without_top_k(self):
        probs = torch.tensor([[0.30, 0.20, 0.18, 0.17, 0.15]])
        _, filtered, token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
            probs,
            top_ks=torch.tensor([TOP_K_ALL]),
            top_ps=torch.tensor([0.49]),
            min_ps=torch.tensor([0.0]),
            need_min_p_sampling=False,
            sampling_seed=None,
            positions=torch.tensor([0]),
            return_filtered_probs=True,
        )

        kept_ids = token_ids[filtered > 0].tolist()
        self.assertEqual(set(kept_ids), {0, 1})


if __name__ == "__main__":
    unittest.main()
