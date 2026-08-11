"""Unit tests pinning the wire contract of the raw-frame mm transport codec
(mm_utils.extract_tensor_frames / rehydrate_tensor_frames): supported tensors
round-trip exactly, the pickled header excludes the payload, everything
outside the supported scope falls through to legacy single-frame untouched,
and restore() hands the sender back its exact original objects."""

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
    def test_wire_roundtrip_tensor_features(self):
        feats = [
            torch.randn(3, 5, dtype=torch.float32),
            torch.randn(2, 8, dtype=torch.bfloat16),  # no numpy equivalent
            torch.arange(6, dtype=torch.int32),
        ]
        req = _make_req(
            [MultimodalDataItem(modality=Modality.IMAGE, feature=f) for f in feats]
            + [MultimodalDataItem(modality=Modality.IMAGE)]  # feature=None
        )

        out = _roundtrip_over_wire(req).mm_inputs.mm_items

        for got, want in zip(out, feats):
            self.assertTrue(torch.equal(got.feature, want))
            self.assertEqual(got.feature.dtype, want.dtype)
            self.assertEqual(got.feature.shape, want.shape)
        self.assertIsNone(out[3].feature)

        # The sender's request is intact after the round trip.
        for item, want in zip(req.mm_inputs.mm_items, feats):
            self.assertIs(item.feature, want)

    def test_unsupported_shapes_fall_through_to_legacy(self):
        # The narrowed scope is CPU torch.Tensor in mm_item.feature only.
        # Everything else must produce a single-frame legacy message with
        # the values left untouched: the extractor must never extract
        # something the receiver will not rehydrate.
        feat_ndarray = np.arange(28, dtype=np.float16).reshape(4, 7)
        feat_list = [torch.randn(2, 4), torch.arange(6, dtype=torch.int32)]
        feat_tuple = (torch.randn(3),)
        precomputed = torch.randn(2, 8, dtype=torch.bfloat16)

        items = [
            MultimodalDataItem(modality=Modality.IMAGE, feature=feat_ndarray),
            MultimodalDataItem(modality=Modality.IMAGE, feature=feat_list),
            MultimodalDataItem(modality=Modality.IMAGE, feature=feat_tuple),
            MultimodalDataItem(
                modality=Modality.IMAGE, precomputed_embeddings=precomputed
            ),
        ]
        req = _make_req(items)

        frames, restore = extract_tensor_frames(req)

        self.assertIsNone(frames)
        self.assertIs(items[0].feature, feat_ndarray)
        self.assertIs(items[1].feature, feat_list)
        self.assertIs(items[2].feature, feat_tuple)
        self.assertIs(items[3].precomputed_embeddings, precomputed)
        restore()  # must be callable even when nothing was extracted

    def test_mixed_supported_and_unsupported(self):
        # A tensor feature is extracted while an ndarray feature stays
        # inside the pickled header; both survive the round trip.
        feat_tensor = torch.randn(3, 5, dtype=torch.float32)
        feat_ndarray = np.arange(28, dtype=np.float16).reshape(4, 7)
        req = _make_req(
            [
                MultimodalDataItem(modality=Modality.IMAGE, feature=feat_tensor),
                MultimodalDataItem(modality=Modality.IMAGE, feature=feat_ndarray),
            ]
        )

        out = _roundtrip_over_wire(req).mm_inputs.mm_items

        self.assertTrue(torch.equal(out[0].feature, feat_tensor))
        np.testing.assert_array_equal(out[1].feature, feat_ndarray)
        self.assertEqual(out[1].feature.dtype, feat_ndarray.dtype)

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
        feat_a = torch.randn(2, 3)
        feat_b = torch.randn(3)
        item_a = MultimodalDataItem(modality=Modality.IMAGE, feature=feat_a)
        item_b = MultimodalDataItem(modality=Modality.IMAGE, feature=feat_b)
        req = _make_req([item_a, item_b])

        frames, restore = extract_tensor_frames(req)
        self.assertIsNotNone(frames)
        # Extraction replaced the values with placeholders (what the header
        # pickles)...
        self.assertIsNot(item_a.feature, feat_a)
        self.assertIsNot(item_b.feature, feat_b)
        restore()
        # ...and restore() puts back the identical original objects.
        self.assertIs(item_a.feature, feat_a)
        self.assertIs(item_b.feature, feat_b)

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
