import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import TransferInfo as MooncakeTransferInfo
from sglang.srt.disaggregation.nixl.conn import TransferInfo as NixlTransferInfo
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDecodeRadixTransferProtocol(unittest.TestCase):
    @patch("sglang.srt.disaggregation.common.conn.requests.get")
    def test_decode_radix_rejects_legacy_prefill_protocol(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "attn_tp_size": 1,
            "attn_cp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "page_size": 1,
            "kv_cache_dtype": "auto",
            "follow_bootstrap_room": True,
            "disagg_transfer_protocol_version": 0,
        }

        manager = object.__new__(CommonKVManager)
        manager.prefill_info_table = {}
        manager.kv_args = SimpleNamespace(page_size=1)
        manager.server_args = SimpleNamespace(
            kv_cache_dtype="auto",
            disaggregation_decode_enable_radix_cache=True,
        )

        with self.assertRaisesRegex(RuntimeError, "transfer protocol version"):
            manager.try_ensure_parallel_info("127.0.0.1:8998")

    def test_prefix_length_mismatch_marks_room_failed(self):
        manager = object.__new__(CommonKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.req_to_decode_prefix_len = {}
        manager.request_status = {42: KVPoll.WaitingForInput}
        manager.failure_records = {}
        manager.failure_lock = threading.Lock()

        self.assertTrue(manager.record_decode_prefix_len(42, 16))
        self.assertFalse(manager.record_decode_prefix_len(42, 8))
        self.assertEqual(manager.request_status[42], KVPoll.Failed)
        self.assertIn("mismatch", manager.failure_records[42])

    def test_nixl_empty_kv_with_non_dummy_flag_is_real_participant(self):
        info = NixlTransferInfo.from_zmq(
            [
                b"42",
                b"127.0.0.1",
                b"1234",
                b"agent",
                np.array([], dtype=np.int32).tobytes(),
                b"7",
                b"1",
                b"",
                b"16",
                b"0",
            ]
        )

        self.assertFalse(info.is_dummy())
        self.assertEqual(info.decode_prefix_len, 16)
        self.assertEqual(info.dst_kv_indices.size, 0)
        self.assertEqual(info.dst_aux_index, 7)

    def test_nixl_legacy_empty_kv_message_still_defaults_to_dummy(self):
        info = NixlTransferInfo.from_zmq(
            [
                b"42",
                b"127.0.0.1",
                b"1234",
                b"agent",
                np.array([], dtype=np.int32).tobytes(),
                b"7",
                b"1",
                b"",
            ]
        )

        self.assertTrue(info.is_dummy())
        self.assertEqual(info.decode_prefix_len, 0)

    def test_nixl_legacy_full_hit_is_not_dummy(self):
        info = NixlTransferInfo.from_zmq(
            [
                b"42",
                b"127.0.0.1",
                b"1234",
                b"agent",
                b"",
                b"7",
                b"1",
                b"",
                b"16",
            ]
        )

        self.assertFalse(info.is_dummy())
        self.assertEqual(info.decode_prefix_len, 16)

    def test_mooncake_empty_kv_with_aux_is_real_participant(self):
        info = MooncakeTransferInfo.from_zmq(
            [
                b"42",
                b"127.0.0.1",
                b"1234",
                b"session",
                np.array([], dtype=np.int32).tobytes(),
                b"7",
                b"",
                b"1",
                b"16",
            ]
        )

        self.assertFalse(info.is_dummy)
        self.assertEqual(info.decode_prefix_len, 16)
        self.assertEqual(info.dst_kv_indices.size, 0)
        self.assertEqual(info.dst_aux_index, 7)

    def test_mooncake_empty_kv_without_aux_is_dummy(self):
        info = MooncakeTransferInfo.from_zmq(
            [
                b"42",
                b"127.0.0.1",
                b"1234",
                b"session",
                b"",
                b"",
                b"",
                b"1",
                b"16",
            ]
        )

        self.assertTrue(info.is_dummy)
        self.assertEqual(info.decode_prefix_len, 16)


if __name__ == "__main__":
    unittest.main()
