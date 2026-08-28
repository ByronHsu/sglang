from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_overlap_result_uses_the_forward_ping_pong_position():
    req = SimpleNamespace(
        rid="req",
        decode_batch_idx=9,
        kv_committed_len=129,
        mamba_ping_pong_track_buffer=torch.tensor([10, 11]),
        mamba_next_track_idx=1,
        mamba_last_track_idx=None,
        mamba_last_track_seqlen=None,
    )
    batch = SimpleNamespace(
        mamba_decode_batch_idx_cpu=[8],
        mamba_track_buffer_indices=[0],
        req_to_token_pool=SimpleNamespace(
            get_mamba_ping_pong_other_idx=lambda idx: 1 - idx
        ),
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
    )
    server_args = SimpleNamespace(
        mamba_track_interval=64,
        enable_mamba_extra_buffer_lazy=lambda: False,
    )

    processor = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)
    with patch(
        "sglang.srt.managers.scheduler_components.batch_result_processor."
        "get_global_server_args",
        return_value=server_args,
    ):
        processor._mamba_prefix_cache_update(
            req, batch, SimpleNamespace(), 0, known_boundary=False
        )

    assert req.mamba_last_track_idx == 0
    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 1
