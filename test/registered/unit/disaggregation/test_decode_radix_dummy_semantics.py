import unittest

import numpy as np

from sglang.srt.disaggregation.mooncake.conn import TransferInfo as MooncakeTransferInfo
from sglang.srt.disaggregation.nixl.conn import TransferInfo as NixlTransferInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDecodeRadixTransferDummySemantics(unittest.TestCase):
    def test_nixl_empty_kv_message_defaults_to_dummy(self):
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
        self.assertIsNone(info.decode_prefix_len)

    def test_nixl_full_hit_is_not_dummy(self):
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
