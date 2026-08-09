"""Unit tests for srt/disaggregation/common/conn — heartbeat hardening.

Covers the PD hotfix "heartbeat + registration hardening":
1. The heartbeat failure counter is reset on node kill and on re-add, so a
   recovered prefill gets a full max_failures budget instead of being
   re-killed by a single subsequent miss.
2. The probe timeout flows from SGLANG_DISAGGREGATION_HEARTBEAT_TIMEOUT
   (floor-clamped to 2.0s) into the /health request.
3. register_dp_rank_async returns without blocking, the pooled POST lands on
   the bootstrap server, and failures log the canary text and invalidate the
   pooled session.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import json
import socket
import threading
import time
import unittest
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

import sglang.srt.disaggregation.common.conn as conn_mod
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVSender
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase


def _make_manager(mode: DisaggregationMode, bootstrap_port: int = 8998):
    """Build a real CommonKVManager with external dependencies (dist rank
    getters, zmq bind, local-ip detection) patched out. The PREFILL mode
    performs a real HTTP PUT /route to 127.0.0.1:bootstrap_port during
    construction, so prefill callers must have a mock bootstrap server
    listening there."""
    kv_args = SimpleNamespace(
        kv_item_lens=[1],
        state_item_lens=[[1]],
        page_size=16,
        engine_rank=0,
        system_dp_rank=0,
        pp_rank=0,
    )
    server_args = SimpleNamespace(
        host="127.0.0.1",
        disaggregation_bootstrap_port=bootstrap_port,
        dist_init_addr=None,
        nnodes=1,
        enable_dp_attention=False,
        dp_size=1,
        pp_size=1,
        kv_cache_dtype="auto",
        load_balance_method="round_robin",
        disaggregation_decode_enable_radix_cache=False,
    )
    with ExitStack() as stack:
        for name in (
            "get_attention_tp_size",
            "get_attention_tp_rank",
            "get_attention_cp_size",
            "get_attention_cp_rank",
            "get_attention_dp_size",
            "get_attention_dp_rank",
        ):
            stack.enter_context(
                patch.object(
                    conn_mod, name, return_value=1 if name.endswith("size") else 0
                )
            )
        stack.enter_context(
            patch.object(conn_mod, "get_local_ip_auto", return_value="127.0.0.1")
        )
        stack.enter_context(
            patch.object(
                conn_mod,
                "get_zmq_socket_on_host",
                return_value=(23456, MagicMock()),
            )
        )
        stack.enter_context(patch.object(conn_mod.zmq, "Context", MagicMock()))
        if mode == DisaggregationMode.PREFILL:
            stack.enter_context(patch.object(conn_mod, "get_pp_group", MagicMock()))
        return CommonKVManager(kv_args, mode, server_args)


class _BootstrapHandler(BaseHTTPRequestHandler):
    """Minimal stand-in for the PD bootstrap server: accepts PUT /route
    (prefill registration during manager construction) and records
    POST /register_dp_rank payloads."""

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def _respond(self, status):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_PUT(self):
        self._read_body()
        self._respond(200)

    def do_POST(self):
        body = self._read_body()
        if self.server.post_delay:
            time.sleep(self.server.post_delay)
        self.server.posts.append((self.path, json.loads(body)))
        self.server.post_event.set()
        self._respond(self.server.post_status)

    def log_message(self, format, *args):
        pass


def _start_bootstrap_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BootstrapHandler)
    server.posts = []
    server.post_event = threading.Event()
    server.post_status = 200
    server.post_delay = 0.0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _free_dead_port():
    """A port with nothing listening: connections get refused immediately."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


_PREFILL_INFO_PAYLOAD = {
    "attn_tp_size": 1,
    "attn_cp_size": 1,
    "dp_size": 1,
    "pp_size": 1,
    "page_size": 16,
    "kv_cache_dtype": None,
    "follow_bootstrap_room": False,
}


class TestHeartbeatFailureBudget(CustomTestCase):
    """State machine of the decode-side heartbeat failure counter."""

    ADDR = "127.0.0.1:7000"

    def _failing_session(self):
        session = MagicMock()
        session.get.side_effect = requests.exceptions.ConnectionError("down")
        return session

    def test_kill_pops_counter_and_readd_survives_single_miss(self):
        mgr = _make_manager(DisaggregationMode.DECODE)
        addr = self.ADDR
        mgr.prefill_info_table[addr] = MagicMock()
        mgr.session_pool[addr] = self._failing_session()
        mgr.addr_to_rooms_tracker[addr].add(101)
        mgr.update_status(101, KVPoll.Bootstrapping)

        # Miss 1: counted, node still alive.
        mgr._heartbeat_check_once()
        self.assertEqual(mgr.heartbeat_failures[addr], 1)
        self.assertIn(addr, mgr.prefill_info_table)

        # Miss 2 == max_failures: node killed, in-flight room failed,
        # and the stale miss counter is popped along with the other state.
        mgr._heartbeat_check_once()
        self.assertNotIn(addr, mgr.prefill_info_table)
        self.assertEqual(mgr.check_status(101), KVPoll.Failed)
        self.assertNotIn(addr, mgr.heartbeat_failures)

        # Prefill recovers and is re-added via try_ensure_parallel_info.
        response = MagicMock(status_code=200)
        response.json.return_value = dict(_PREFILL_INFO_PAYLOAD)
        with patch.object(conn_mod.requests, "get", return_value=response):
            self.assertTrue(mgr.try_ensure_parallel_info(addr))
        self.assertIn(addr, mgr.prefill_info_table)

        # A single subsequent miss must NOT re-kill the recovered node:
        # it gets a full max_failures budget again.
        mgr.session_pool[addr] = self._failing_session()
        mgr._heartbeat_check_once()
        self.assertEqual(mgr.heartbeat_failures[addr], 1)
        self.assertIn(addr, mgr.prefill_info_table)

    def test_readd_pops_stale_counter(self):
        """try_ensure_parallel_info itself clears any stale miss count,
        independent of how the addr previously disappeared."""
        mgr = _make_manager(DisaggregationMode.DECODE)
        addr = self.ADDR
        mgr.heartbeat_failures[addr] = mgr.max_failures  # stale count

        response = MagicMock(status_code=200)
        response.json.return_value = dict(_PREFILL_INFO_PAYLOAD)
        with patch.object(conn_mod.requests, "get", return_value=response):
            self.assertTrue(mgr.try_ensure_parallel_info(addr))

        self.assertIn(addr, mgr.prefill_info_table)
        self.assertNotIn(addr, mgr.heartbeat_failures)


class TestHeartbeatTimeoutEnv(CustomTestCase):
    ADDR = "127.0.0.1:7000"

    def test_timeout_flows_from_env_to_probe(self):
        with envs.SGLANG_DISAGGREGATION_HEARTBEAT_TIMEOUT.override(7.5):
            mgr = _make_manager(DisaggregationMode.DECODE)
        self.assertEqual(mgr.heartbeat_timeout, 7.5)

        session = MagicMock()
        session.get.return_value = MagicMock(status_code=200)
        mgr.prefill_info_table[self.ADDR] = MagicMock()
        mgr.session_pool[self.ADDR] = session
        mgr._heartbeat_check_once()

        session.get.assert_called_once_with(
            f"http://{self.ADDR}/health",
            timeout=(7.5, 7.5),
            headers={"Connection": "keep-alive"},
        )
        self.assertEqual(mgr.heartbeat_failures[self.ADDR], 0)

    def test_timeout_floor_clamped_to_two_seconds(self):
        with envs.SGLANG_DISAGGREGATION_HEARTBEAT_TIMEOUT.override(0.5):
            mgr = _make_manager(DisaggregationMode.DECODE)
        self.assertEqual(mgr.heartbeat_timeout, 2.0)


class TestRegisterDpRankAsync(CustomTestCase):
    def setUp(self):
        self.server = _start_bootstrap_server()
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]
        self.addr = f"127.0.0.1:{self.port}"
        self.mgr = _make_manager(DisaggregationMode.PREFILL, bootstrap_port=self.port)
        self.addCleanup(self.mgr.registration_executor.shutdown)

    def test_returns_without_blocking_and_post_lands(self):
        self.server.post_delay = 1.0

        start = time.monotonic()
        future = self.mgr.register_dp_rank_async(self.addr, 42, 3)
        elapsed = time.monotonic() - start
        self.assertLess(
            elapsed, 0.5, "register_dp_rank_async blocked on the HTTP round-trip"
        )

        future.result(timeout=10)
        self.assertEqual(
            self.server.posts,
            [("/register_dp_rank", {"bootstrap_room": 42, "dp_rank": 3})],
        )
        # Successful POST keeps the pooled keep-alive session.
        self.assertIn(self.addr, self.mgr.session_pool)
        self.assertEqual(self.mgr._registration_inflight, 0)

    def test_post_exception_logs_and_drops_session(self):
        dead_addr = f"127.0.0.1:{_free_dead_port()}"
        with self.assertLogs(
            "sglang.srt.disaggregation.common.conn", level="ERROR"
        ) as logs:
            future = self.mgr.register_dp_rank_async(dead_addr, 7, 1)
            future.result(timeout=10)
        self.assertTrue(
            any("Failed to register prefill dp_rank" in line for line in logs.output),
            logs.output,
        )
        # The pooled session is invalidated so a half-dead keep-alive
        # connection is not reused.
        self.assertNotIn(dead_addr, self.mgr.session_pool)
        self.assertEqual(self.mgr._registration_inflight, 0)

    def test_non_200_logs_canary_text(self):
        self.server.post_status = 500
        with self.assertLogs(
            "sglang.srt.disaggregation.common.conn", level="ERROR"
        ) as logs:
            future = self.mgr.register_dp_rank_async(self.addr, 8, 2)
            future.result(timeout=10)
        self.assertTrue(
            any("Failed to register prefill dp_rank" in line for line in logs.output),
            logs.output,
        )
        # HTTP-level errors mean the connection itself worked: keep the session.
        self.assertIn(self.addr, self.mgr.session_pool)

    def test_sender_hot_path_is_non_blocking(self):
        """CommonKVSender._register_prefill_dp_rank (the scheduler hot path)
        must route through the executor rather than blocking on the POST."""
        self.server.post_delay = 1.0
        sender = MagicMock(spec=CommonKVSender)
        sender.kv_mgr = self.mgr
        sender.bootstrap_server_url = self.addr
        sender.bootstrap_room = 9

        start = time.monotonic()
        CommonKVSender._register_prefill_dp_rank(sender)
        elapsed = time.monotonic() - start
        self.assertLess(
            elapsed, 0.5, "_register_prefill_dp_rank blocked the scheduler hot path"
        )

        self.assertTrue(self.server.post_event.wait(timeout=10))
        self.assertEqual(
            self.server.posts,
            [
                (
                    "/register_dp_rank",
                    {"bootstrap_room": 9, "dp_rank": self.mgr.attn_dp_rank},
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
