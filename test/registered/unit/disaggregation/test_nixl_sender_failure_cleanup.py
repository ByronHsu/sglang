import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVSender
from sglang.srt.disaggregation.nixl.conn import NixlKVSender
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNixlSenderFailureCleanup(unittest.TestCase):
    def test_constructor_arms_bootstrap_timeout(self):
        with (
            patch.object(CommonKVSender, "__init__", return_value=None),
            patch("sglang.srt.disaggregation.nixl.conn.time.time", return_value=123.0),
        ):
            sender = NixlKVSender(object(), "prefill:123", 7, [0], 0)

        self.assertEqual(sender.init_time, 123.0)

    def test_poll_applies_common_bootstrap_timeout(self):
        room = 7
        sender = NixlKVSender.__new__(NixlKVSender)
        sender.bootstrap_room = room
        sender.conclude_state = None
        sender._send_failed = False
        sender._transfer_start_time = None
        sender.init_time = 10.0
        sender.kv_mgr = SimpleNamespace(
            bootstrap_timeout=5.0,
            request_status={room: KVPoll.Bootstrapping},
            failure_records={},
            failure_lock=threading.Lock(),
            check_status=lambda requested_room: KVPoll.Bootstrapping,
        )
        sender.kv_mgr.record_failure = (
            lambda requested_room, reason: sender.kv_mgr.failure_records.__setitem__(
                requested_room, reason
            )
        )
        sender.kv_mgr.update_status = (
            lambda requested_room, status: sender.kv_mgr.request_status.__setitem__(
                requested_room, status
            )
        )

        with patch(
            "sglang.srt.disaggregation.common.conn.time.time", return_value=20.0
        ):
            self.assertEqual(sender.poll(), KVPoll.Failed)

        self.assertEqual(sender.kv_mgr.request_status[room], KVPoll.Failed)
        self.assertIn("timed out", sender.kv_mgr.failure_records[room])

    def test_failure_exception_cleans_room_state_before_raising(self):
        room = 7
        expected_exc = RuntimeError("transfer failed")
        sender = NixlKVSender.__new__(NixlKVSender)
        sender.bootstrap_room = room
        sender.conclude_state = None
        sender._send_failed = False
        sender._send_error = None
        staging_ctx = SimpleNamespace(
            prefetched_rooms={room, 8},
            prefetch_requested={(room, 0, "session-a"), (8, 0, "session-b")},
        )
        sender.kv_mgr = SimpleNamespace(
            enable_staging=True,
            _staging_ctx=staging_ctx,
            request_status={room: object()},
            req_to_decode_prefix_len={room: 3},
            transfer_infos={room: object()},
            exceptions={room: expected_exc},
            failure_records={room: "transfer failed"},
            failure_lock=threading.Lock(),
        )

        with self.assertRaises(RuntimeError) as cm:
            sender.failure_exception()

        self.assertIs(cm.exception, expected_exc)
        self.assertTrue(sender._send_failed)
        self.assertEqual(sender.conclude_state, KVPoll.Failed)
        self.assertNotIn(room, sender.kv_mgr.request_status)
        self.assertNotIn(room, sender.kv_mgr.req_to_decode_prefix_len)
        self.assertNotIn(room, sender.kv_mgr.transfer_infos)
        self.assertNotIn(room, sender.kv_mgr.exceptions)
        self.assertNotIn(room, sender.kv_mgr.failure_records)
        self.assertNotIn(room, staging_ctx.prefetched_rooms)
        self.assertNotIn((room, 0, "session-a"), staging_ctx.prefetch_requested)
        self.assertIn(8, staging_ctx.prefetched_rooms)
        self.assertIn((8, 0, "session-b"), staging_ctx.prefetch_requested)


if __name__ == "__main__":
    unittest.main()
