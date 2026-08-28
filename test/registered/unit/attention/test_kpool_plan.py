import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsa.kpool_plan import (
    _decompose_compress,
    _kpool_cpu_plan,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKPoolPlan(unittest.TestCase):
    def test_hybrid_backend_delegates_indexer_metadata(self):
        expected = object()
        full_backend = SimpleNamespace(
            token_to_kv_pool=object(),
            req_to_token_pool=object(),
            get_indexer_metadata=lambda layer_id, forward_batch: expected,
        )
        backend = HybridLinearAttnBackend(full_backend, object(), [3])

        self.assertIs(backend.get_indexer_metadata(3, object()), expected)

    def test_explicit_linear_layer_wins_during_adaptive_rebuild(self):
        backend = object.__new__(HybridLinearAttnBackend)
        backend.full_attn_layers = [0]
        layer = RadixLinearAttention(0, 1, 1, 1, 1, 1, 1)

        self.assertFalse(backend._is_full_attn(layer))

    def test_decompose_crosses_multiple_pools(self):
        self.assertEqual(_decompose_compress(3, 6, 4), (3, 0, 2, 1))
        self.assertEqual(_decompose_compress(8, 2, 4), (0, 2, 0, 2))

    def test_multi_request_plan_keeps_pool_and_tail_rows_separate(self):
        batch = SimpleNamespace(
            batch_size=2,
            extend_seq_lens_cpu=[6, 2],
            seq_lens_cpu=torch.tensor([9, 10]),
            req_pool_indices=torch.tensor([7, 11]),
        )

        plan = _kpool_cpu_plan(batch, pool_size=4, slots_per_page=64)

        self.assertEqual(plan.pool_req, [7, 7])
        self.assertEqual(plan.pool_pool_id, [0, 1])
        self.assertEqual(plan.pool_n_from_tail, [3, 0])
        self.assertEqual(plan.pool_chunk_src, [0, 1])
        self.assertEqual(plan.tail_req, [7, 11])
        self.assertEqual(plan.tail_n_write, [1, 2])
        self.assertEqual(plan.ragged_q_len, [6, 2])


if __name__ == "__main__":
    unittest.main()
