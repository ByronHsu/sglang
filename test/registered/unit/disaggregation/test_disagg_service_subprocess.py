"""Unit tests for srt/managers/disagg_service — bootstrap server process isolation.

The PD bootstrap server (the /health, /route, dp-rank-registry control plane)
must run in a dedicated spawned process so GIL/CPU pressure in the tokenizer
manager / router process cannot delay it. These tests exercise the subprocess
lifecycle end-to-end on CPU (spawn, readiness, HTTP round-trip, reaping), the
loud-failure path for an unbindable port, and the
SGLANG_DISABLE_BOOTSTRAP_SERVER_SUBPROCESS kill switch.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="base-a-test-cpu")

import socket
import time
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.managers.disagg_service import (
    BootstrapServerProcHandle,
    start_disagg_service,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

HOST = "127.0.0.1"

# Minimal single-rank registration payload accepted by /route (PUT). With all
# sizes 1, one registration makes the server "ready" for /route GETs.
ROUTE_PUT_PAYLOAD = {
    "attn_tp_size": 1,
    "attn_tp_rank": 0,
    "attn_cp_size": 1,
    "attn_cp_rank": 0,
    "attn_dp_size": 1,
    "attn_dp_rank": 0,
    "pp_size": 1,
    "pp_rank": 0,
    "system_dp_size": 1,
    "system_dp_rank": 0,
    "rank_ip": "10.1.2.3",
    "rank_port": 7777,
    "page_size": 16,
    "kv_cache_dtype": "auto",
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _prefill_server_args(port: int) -> ServerArgs:
    return ServerArgs(
        model_path="dummy",
        disaggregation_mode="prefill",
        disaggregation_bootstrap_port=port,
        host=HOST,
    )


def _assert_route_round_trip(test: unittest.TestCase, base_url: str):
    """PUT a rank registration and read it back via GET /route."""
    resp = requests.put(f"{base_url}/route", json=ROUTE_PUT_PAYLOAD, timeout=5)
    test.assertEqual(resp.status_code, 200)

    resp = requests.get(
        f"{base_url}/route",
        params={
            "prefill_dp_rank": "0",
            "prefill_cp_rank": "0",
            "target_tp_rank": "0",
            "target_pp_rank": "0",
        },
        timeout=5,
    )
    test.assertEqual(resp.status_code, 200)
    info = resp.json()
    test.assertEqual(info["rank_ip"], ROUTE_PUT_PAYLOAD["rank_ip"])
    test.assertEqual(info["rank_port"], ROUTE_PUT_PAYLOAD["rank_port"])


class TestBootstrapServerSubprocess(CustomTestCase):
    def test_subprocess_lifecycle_and_http_round_trip(self):
        port = _free_port()
        handle = start_disagg_service(_prefill_server_args(port))
        try:
            # A dedicated child process hosts the server.
            self.assertIsInstance(handle, BootstrapServerProcHandle)
            self.assertTrue(handle.is_alive())

            base_url = f"http://{HOST}:{port}"
            # start_disagg_service returned only after readiness, so /health
            # must answer immediately.
            resp = requests.get(f"{base_url}/health", timeout=5)
            self.assertEqual(resp.status_code, 200)

            # Registration state lives in (and round-trips through) the child.
            _assert_route_round_trip(self, base_url)
        finally:
            handle.close()

        # close() reaps the child and the server stops listening.
        self.assertFalse(handle.is_alive())
        self.assertIsNotNone(handle.proc.exitcode)
        with self.assertRaises(requests.ConnectionError):
            requests.get(f"http://{HOST}:{port}/health", timeout=5)

    def test_port_in_use_fails_loudly(self):
        # With the legacy in-thread server a bind failure was only a buried
        # log line; the subprocess path must surface it as a startup error.
        with socket.socket() as blocker:
            blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            blocker.bind((HOST, 0))
            blocker.listen(1)
            port = blocker.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, "failed to start"):
                start_disagg_service(_prefill_server_args(port))

    def test_kill_switch_reverts_to_in_thread_server(self):
        port = _free_port()
        with envs.SGLANG_DISABLE_BOOTSTRAP_SERVER_SUBPROCESS.override(True):
            server = start_disagg_service(_prefill_server_args(port))
        try:
            # Legacy behavior: the server object itself, hosted on a daemon
            # thread in this process — no child process.
            self.assertNotIsInstance(server, BootstrapServerProcHandle)
            self.assertTrue(server.thread.is_alive())

            base_url = f"http://{HOST}:{port}"
            deadline = 10.0
            start = time.monotonic()
            while True:
                # The in-thread server (unlike the subprocess path) offers no
                # readiness guarantee; poll briefly.
                try:
                    resp = requests.get(f"{base_url}/health", timeout=2)
                    if resp.status_code == 200:
                        break
                except requests.RequestException:
                    if time.monotonic() - start > deadline:
                        raise
                time.sleep(0.05)
            _assert_route_round_trip(self, base_url)
        finally:
            server.close()

    def test_non_prefill_modes_start_nothing(self):
        for mode in ("null", "decode"):
            self.assertIsNone(
                start_disagg_service(
                    ServerArgs(model_path="dummy", disaggregation_mode=mode)
                )
            )


if __name__ == "__main__":
    unittest.main()
