"""Unit tests for the raw-frame ZMQ transport codec for multimodal tensors.

Pins the wire contract of mm_utils.extract_tensor_frames /
rehydrate_tensor_frames (see the raw-frame transport block in
python/sglang/srt/managers/mm_utils.py):

  1. extract -> pickle header -> (simulated) wire -> unpickle -> rehydrate
     reproduces the original tensors exactly (values, dtypes, shapes) for
     every feature shape: torch tensor, list of tensors, ndarray,
     precomputed_embeddings, and None.
  2. The pickled header does NOT contain the tensor payload — the entire
     point of the transport (relays forward payload frames opaquely).
  3. No-op (legacy single-frame) on text-only requests and on features
     already wrapped by the SHM transport.
  4. restore() hands the sender back its exact original objects, so the
     request stays intact after (or despite a failure of) the send.
"""

import pickle
import unittest
from multiprocessing import shared_memory

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from sglang.srt.managers.io_struct import TokenizedGenerateReqInput  # noqa: E402
from sglang.srt.managers.mm_utils import (  # noqa: E402
    ShmPointerMMData,
    extract_tensor_frames,
    rehydrate_tensor_frames,
)
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402


def _make_req(mm_items):
    mm_inputs = MultimodalInputs(mm_items=mm_items) if mm_items is not None else None
    return TokenizedGenerateReqInput(
        input_text="describe the image",
        input_ids=[1, 2, 3, 4],
        mm_inputs=mm_inputs,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        rid="test-rid",
    )


def _roundtrip_over_wire(req):
    """extract -> pickle -> raw bytes frames (as zmq would deliver them) ->
    unpickle -> rehydrate. Restores the sender's object before returning."""
    frames, restore = extract_tensor_frames(req)
    assert frames is not None
    try:
        header = pickle.dumps(req)
        wire_frames = [bytes(memoryview(f)) for f in frames]
    finally:
        restore()
    return rehydrate_tensor_frames(pickle.loads(header), wire_frames)


class TestRawFrameCodec(CustomTestCase):
    def test_wire_roundtrip_all_feature_shapes(self):
        feat_tensor = torch.randn(3, 5, dtype=torch.float32)
        feat_list = [
            torch.randn(2, 4, dtype=torch.float64),
            torch.arange(6, dtype=torch.int32),
        ]
        feat_ndarray = np.arange(28, dtype=np.float16).reshape(4, 7)
        precomputed = torch.randn(2, 8, dtype=torch.bfloat16)

        req = _make_req(
            [
                MultimodalDataItem(modality=Modality.IMAGE, feature=feat_tensor),
                MultimodalDataItem(modality=Modality.IMAGE, feature=list(feat_list)),
                MultimodalDataItem(modality=Modality.IMAGE, feature=feat_ndarray),
                MultimodalDataItem(
                    modality=Modality.IMAGE, precomputed_embeddings=precomputed
                ),
                MultimodalDataItem(modality=Modality.IMAGE),  # feature=None
            ]
        )

        out = _roundtrip_over_wire(req).mm_inputs.mm_items

        self.assertTrue(torch.equal(out[0].feature, feat_tensor))
        self.assertEqual(out[0].feature.dtype, feat_tensor.dtype)
        self.assertEqual(out[0].feature.shape, feat_tensor.shape)

        self.assertIsInstance(out[1].feature, list)
        self.assertEqual(len(out[1].feature), len(feat_list))
        for got, want in zip(out[1].feature, feat_list):
            self.assertTrue(torch.equal(got, want))
            self.assertEqual(got.dtype, want.dtype)

        np.testing.assert_array_equal(out[2].feature, feat_ndarray)
        self.assertEqual(out[2].feature.dtype, feat_ndarray.dtype)

        self.assertTrue(torch.equal(out[3].precomputed_embeddings, precomputed))
        self.assertEqual(out[3].precomputed_embeddings.dtype, torch.bfloat16)
        self.assertIsNone(out[3].feature)

        self.assertIsNone(out[4].feature)
        self.assertIsNone(out[4].precomputed_embeddings)

        # The sender's request is intact after the round trip.
        self.assertIs(req.mm_inputs.mm_items[0].feature, feat_tensor)

    def test_header_excludes_tensor_payload(self):
        big = torch.zeros(1 << 20, dtype=torch.float32)  # 4 MB
        req = _make_req([MultimodalDataItem(modality=Modality.IMAGE, feature=big)])
        frames, restore = extract_tensor_frames(req)
        try:
            header = pickle.dumps(req)
        finally:
            restore()
        self.assertEqual(
            sum(len(memoryview(f)) for f in frames), big.numel() * big.element_size()
        )
        # Header carries only the placeholder + request fields, not 4 MB.
        self.assertLess(len(header), 64 * 1024)

    def test_non_contiguous_tensor_roundtrip(self):
        nc = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
        self.assertFalse(nc.is_contiguous())
        req = _make_req([MultimodalDataItem(modality=Modality.IMAGE, feature=nc)])

        out = _roundtrip_over_wire(req).mm_inputs.mm_items[0]

        self.assertTrue(torch.equal(out.feature, nc))
        self.assertEqual(out.feature.shape, nc.shape)
        # restore() handed back the original (still non-contiguous) tensor.
        self.assertIs(req.mm_inputs.mm_items[0].feature, nc)

    def test_restore_leaves_sender_object_intact(self):
        feat = torch.randn(2, 3)
        lst = [torch.randn(3), np.ones((2, 2), dtype=np.float64)]
        item_a = MultimodalDataItem(modality=Modality.IMAGE, feature=feat)
        item_b = MultimodalDataItem(modality=Modality.IMAGE, feature=lst)
        req = _make_req([item_a, item_b])

        frames, restore = extract_tensor_frames(req)
        self.assertIsNotNone(frames)
        # Extraction replaced the values with placeholders (what the header
        # pickles)...
        self.assertIsNot(item_a.feature, feat)
        restore()
        # ...and restore() puts back the identical original objects.
        self.assertIs(item_a.feature, feat)
        self.assertIs(item_b.feature, lst)
        self.assertIs(item_b.feature[0], lst[0])
        self.assertIs(item_b.feature[1], lst[1])

    def test_noop_on_text_only(self):
        req = _make_req(None)
        frames, restore = extract_tensor_frames(req)
        self.assertIsNone(frames)
        restore()  # must be callable even when nothing was extracted

    def test_noop_on_shm_wrapped_feature(self):
        # In SHM/cuda-ipc transport modes wrap_shm_features has already
        # replaced tensors with small proxies; the extractor must leave them
        # alone so the message stays single-frame.
        wrapped = ShmPointerMMData(torch.randn(4, 4))
        try:
            item = MultimodalDataItem(modality=Modality.IMAGE, feature=wrapped)
            req = _make_req([item])
            frames, restore = extract_tensor_frames(req)
            self.assertIsNone(frames)
            self.assertIs(item.feature, wrapped)
            restore()
        finally:
            shm = shared_memory.SharedMemory(name=wrapped.shm_name)
            shm.close()
            shm.unlink()

    def test_empty_tensor_falls_back_to_legacy(self):
        # torch.frombuffer cannot express a zero-length buffer; the extractor
        # must skip empty tensors so the message stays single-frame.
        item = MultimodalDataItem(modality=Modality.IMAGE, feature=torch.empty(0))
        req = _make_req([item])
        frames, restore = extract_tensor_frames(req)
        self.assertIsNone(frames)
        restore()

    def test_rehydrate_is_noop_without_refs(self):
        # Receivers call rehydrate only for multi-frame messages, but a
        # legacy object passing through must come back unchanged.
        feat = torch.randn(2, 2)
        req = _make_req([MultimodalDataItem(modality=Modality.IMAGE, feature=feat)])
        out = rehydrate_tensor_frames(req, [])
        self.assertIs(out.mm_inputs.mm_items[0].feature, feat)


if __name__ == "__main__":
    unittest.main()
