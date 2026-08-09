import asyncio
import json
import os
import signal
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from types import SimpleNamespace

import aiohttp
import openai
import psutil
import requests
from transformers import AutoTokenizer

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.pause_generation_kit import PauseResumeInPlaceMixin
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TARGET_MODEL_EAGLE3,
)

register_cuda_ci(est_time=760, stage="base-b", runner_config="2-gpu-large")


class TestDisaggregationAccuracy(PauseResumeInPlaceMixin, PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.pause_generate_url = cls.lb_url
        cls.pause_target_urls = [cls.prefill_url, cls.decode_url]
        cls.launch_all()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.62)

    def test_logprob(self):
        prompt = "The capital of france is "
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0},
                "return_logprob": True,
                "return_input_logprob": True,
                "logprob_start_len": 0,
            },
        )

        j = response.json()
        completion_tokens = j["meta_info"]["completion_tokens"]
        input_logprobs = j["meta_info"]["input_token_logprobs"]
        output_logprobs = j["meta_info"]["output_token_logprobs"]

        assert (
            len(output_logprobs) == completion_tokens
        ), f"output_logprobs and completion_tokens should have the same length, but got {len(output_logprobs)} and {completion_tokens}"
        assert (
            len(input_logprobs) > 0
        ), f"input_logprobs should have at least one token, but got {len(input_logprobs)}"

    def test_chat_completion_top_logprobs(self):
        client = openai.Client(api_key="empty", base_url=f"{self.lb_url}/v1")
        response = client.chat.completions.create(
            model="dummy",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            temperature=0,
            max_tokens=8,
            logprobs=True,
            top_logprobs=5,
        )

        self.assertIsNotNone(response.choices[0].logprobs)
        content_logprobs = response.choices[0].logprobs.content
        self.assertGreater(len(content_logprobs), 0)

        first_top_logprobs = next(
            (item.top_logprobs for item in content_logprobs if item.top_logprobs),
            None,
        )
        self.assertIsNotNone(first_top_logprobs)
        self.assertEqual(len(first_top_logprobs), 5)
        self.assertIsInstance(first_top_logprobs[0].token, str)
        self.assertIsInstance(first_top_logprobs[0].logprob, float)

    def test_structured_output(self):
        json_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[\\w]+$"},
                    "population": {"type": "integer"},
                },
                "required": ["name", "population"],
            }
        )

        # JSON
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": "Here is the information of the capital of France in the JSON format.\n",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 64,
                    "json_schema": json_schema,
                },
            },
        )
        output = response.json()["text"]
        # ensure the output is a valid JSON
        json.loads(output)

    def test_pause_resume_in_place(self):
        """Send requests, pause mid-generation, verify no progress during pause, resume."""
        NUM_REQUESTS = 32
        MAX_NEW_TOKENS = 512
        REQUEST_TIMEOUT = 180
        PAUSE_DURATION = 5

        def _generate(prompt_id):
            return requests.post(
                self.lb_url + "/generate",
                json={
                    "text": f"Question {prompt_id}: Write a short essay about the number {prompt_id}.",
                    "sampling_params": {
                        "temperature": 0.8,
                        "max_new_tokens": MAX_NEW_TOKENS,
                    },
                },
                timeout=REQUEST_TIMEOUT,
            )

        with ThreadPoolExecutor(max_workers=NUM_REQUESTS) as executor:
            futures = {executor.submit(_generate, i): i for i in range(NUM_REQUESTS)}

            time.sleep(1)

            requests.post(
                self.prefill_url + "/pause_generation",
                json={"mode": "in_place"},
                timeout=30,
            ).raise_for_status()
            requests.post(
                self.decode_url + "/pause_generation",
                json={"mode": "in_place"},
                timeout=30,
            ).raise_for_status()

            time.sleep(0.5)
            done_before = sum(1 for f in futures if f.done())

            time.sleep(PAUSE_DURATION)
            done_after = sum(1 for f in futures if f.done())

            self.assertLess(
                done_before,
                NUM_REQUESTS,
                "All requests completed before pause took effect — "
                "increase MAX_NEW_TOKENS to make the test meaningful.",
            )

            self.assertEqual(
                done_after - done_before,
                0,
                f"{done_after - done_before} requests completed during pause "
                f"({done_before} before, {done_after} after) — "
                f"pause_generation was not respected by the disagg scheduler.",
            )

            requests.post(
                self.decode_url + "/continue_generation",
                json={},
                timeout=30,
            ).raise_for_status()
            requests.post(
                self.prefill_url + "/continue_generation",
                json={},
                timeout=30,
            ).raise_for_status()

            completed = 0
            errors = []
            for future in as_completed(futures, timeout=REQUEST_TIMEOUT):
                prompt_id = futures[future]
                try:
                    resp = future.result()
                    if resp.status_code == 200:
                        body = resp.json()
                        self.assertIn("text", body)
                        self.assertGreater(len(body["text"]), 0)
                        completed += 1
                    else:
                        errors.append(f"Request {prompt_id}: status={resp.status_code}")
                except Exception as e:
                    errors.append(f"Request {prompt_id}: exception={e}")

        self.assertEqual(
            completed + len(errors),
            NUM_REQUESTS,
            "Some requests did not resolve within the timeout — likely hung during pause.",
        )
        self.assertEqual(
            completed,
            NUM_REQUESTS,
            f"Some requests failed: {completed}/{NUM_REQUESTS} succeeded. Errors: {errors}",
        )

    def test_first_token_finish(self):
        client = openai.Client(api_key="empty", base_url=f"{self.lb_url}/v1")
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        eos_token = tokenizer.eos_token_id
        prompt = "The best programming language for AI is"

        # First token EOS
        res = client.completions.create(
            model="dummy", prompt=prompt, logit_bias={eos_token: 42}
        ).model_dump()
        print(f"{res=}")

        assert res["usage"]["completion_tokens"] == 1, (
            "Expected completion_tokens to be 1 when first token is EOS, "
            f"but got {res['usage']['completion_tokens']}"
        )

        # First token EOS with ignore_eos
        res = client.completions.create(
            model="dummy",
            prompt=prompt,
            logit_bias={eos_token: 42},
            extra_body={"ignore_eos": True},
        ).model_dump()
        print(f"{res=}")

        assert res["usage"]["completion_tokens"] > 1, (
            "Expected completion_tokens to be greater than 1 when ignore_eos is True, "
            f"but got {res['usage']['completion_tokens']}"
        )

        # First token with specified stop token
        stop_token_id = tokenizer.encode(" hello", add_special_tokens=False)[0]
        res = client.completions.create(
            model="dummy",
            prompt=prompt,
            logit_bias={stop_token_id: 42},
            stop=[" hello"],
        ).model_dump()
        print(f"{res=}")

        assert res["usage"]["completion_tokens"] == 1, (
            "Expected completion_tokens to be 1 when first token is stop token, "
            f"but got {res['usage']['completion_tokens']}"
        )

    def test_bootstrap_server_subprocess_running(self):
        """The prefill instance must host its PD bootstrap server in a
        dedicated subprocess (proc title `sglang::disagg_bootstrap_server`).
        Catches a silent fallback to the legacy in-thread server."""
        import psutil

        prefill = psutil.Process(self.process_prefill.pid)
        cmdlines = []
        for child in prefill.children(recursive=True):
            try:
                cmdlines.append(" ".join(child.cmdline()))
            except psutil.NoSuchProcess:
                continue
        self.assertTrue(
            any("sglang::disagg_bootstrap_server" in c for c in cmdlines),
            "No sglang::disagg_bootstrap_server child found under prefill pid "
            f"{self.process_prefill.pid}; children: {cmdlines}",
        )


class TestDisaggregationMooncakeFailure(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # set DISAGGREGATION_TEST_FAILURE_PROB to simulate failure
        os.environ["DISAGGREGATION_TEST_FAILURE_PROB"] = "0.05"
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("DISAGGREGATION_TEST_FAILURE_PROB")
        super().tearDownClass()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )

        # Expect lots of failure but the server cannot crash
        try:
            metrics = run_eval(args)
            print(f"Evaluation metrics: {metrics}")
        except Exception as e:
            print(f"Test encountered expected errors: {e}")
            # Check if servers are still healthy
            try:
                response = requests.get(self.prefill_url + "/health_generate")
                assert response.status_code == 200
                response = requests.get(self.decode_url + "/health_generate")
                assert response.status_code == 200
            except Exception as health_check_error:
                # If health check fails, re-raise the original exception
                raise e from health_check_error


class TestDisaggregationHeartbeatFailover(PDDisaggregationServerBase):
    """Decode-side heartbeat hardening.

    1. A transient stall of the process hosting the prefill bootstrap server,
       shorter than one full miss cycle (interval + probe timeout), must NOT
       abort any in-flight room ("Lost connection with prefill" mass-kill).
    2. A real prefill death must still be detected within the configured
       window, (interval + timeout) * max_failures, instead of requests
       hanging until the 300s bootstrap/waiting timeouts.
    """

    capture_per_side_logs = True

    # Pinned via env overrides in setUpClass so the timing math below is
    # deterministic. 2.0 is the floor-clamp minimum for interval and timeout.
    HEARTBEAT_INTERVAL = 2.0
    HEARTBEAT_TIMEOUT = 2.0
    HEARTBEAT_MAX_FAILURES = 2

    LOST_CONNECTION_CANARY = "Lost connection with prefill"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls._heartbeat_env = ExitStack()
        cls._heartbeat_env.enter_context(
            envs.SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL.override(
                cls.HEARTBEAT_INTERVAL
            )
        )
        cls._heartbeat_env.enter_context(
            envs.SGLANG_DISAGGREGATION_HEARTBEAT_TIMEOUT.override(cls.HEARTBEAT_TIMEOUT)
        )
        cls._heartbeat_env.enter_context(
            envs.SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE.override(
                cls.HEARTBEAT_MAX_FAILURES
            )
        )
        # Subprocesses launched inside the override blocks inherit the values.
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            cls._heartbeat_env.close()

    def _bootstrap_host_processes(self):
        """Process(es) hosting the prefill bootstrap server.

        Today the bootstrap server runs as a daemon thread inside the prefill
        launch_server (tokenizer manager) process. If/when it moves into a
        dedicated subprocess (proctitle sglang::disagg_bootstrap_server),
        stall that child instead so the test keeps targeting the right
        process.
        """
        parent = psutil.Process(self.process_prefill.pid)
        bootstrap_procs = []
        for proc in [parent] + parent.children(recursive=True):
            try:
                ident = " ".join([proc.name(), *proc.cmdline()])
            except psutil.Error:
                continue
            if "disagg_bootstrap_server" in ident:
                bootstrap_procs.append(proc)
        return bootstrap_procs or [parent]

    def _decode_log_snapshot(self):
        return tuple(
            len(buf.getvalue()) if buf is not None else 0
            for buf in (self._decode_stdout_buf, self._decode_stderr_buf)
        )

    def _decode_log_since(self, snapshot):
        parts = []
        for buf, start in zip(
            (self._decode_stdout_buf, self._decode_stderr_buf), snapshot
        ):
            if buf is not None:
                parts.append(buf.getvalue()[start:])
        return "".join(parts)

    def _generate(self, prompt, max_new_tokens=32, timeout=90):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=timeout,
        )
        return response

    def _request_worker(self, worker_id, results, stop_event):
        seq = 0
        while not stop_event.is_set():
            try:
                response = self._generate(f"[w{worker_id}-{seq}] What is 2+2?")
                results.append(
                    {
                        "worker": worker_id,
                        "seq": seq,
                        "ok": response.status_code == 200,
                        "detail": f"status={response.status_code} body={response.text[:200]}",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "worker": worker_id,
                        "seq": seq,
                        "ok": False,
                        "detail": f"exception={e!r}",
                    }
                )
            seq += 1

    def test_transient_stall_no_abort_then_real_death_detected(self):
        miss_cycle = self.HEARTBEAT_INTERVAL + self.HEARTBEAT_TIMEOUT

        # Warm up: the decode side only heartbeats prefill addrs it has seen
        # a request for (prefill_info_table entry), so pair P and D first.
        warmup = self._generate("warmup", max_new_tokens=4)
        self.assertEqual(warmup.status_code, 200, warmup.text)

        # ---- Phase 1: transient bootstrap stall, zero aborted rooms ----
        log_snapshot = self._decode_log_snapshot()
        results = []
        stop_event = threading.Event()
        workers = [
            threading.Thread(
                target=self._request_worker,
                args=(i, results, stop_event),
                daemon=True,
            )
            for i in range(4)
        ]
        for worker in workers:
            worker.start()
        time.sleep(2)  # ensure rooms are in flight

        stall_duration = miss_cycle - 1.0  # at most ONE probe miss can land
        bootstrap_procs = self._bootstrap_host_processes()
        try:
            for proc in bootstrap_procs:
                proc.send_signal(signal.SIGSTOP)
            time.sleep(stall_duration)
        finally:
            for proc in bootstrap_procs:
                proc.send_signal(signal.SIGCONT)

        # Let the heartbeat recover (counter resets on next 200) and let
        # requests that spanned the stall drain.
        time.sleep(2 * miss_cycle)
        stop_event.set()
        for worker in workers:
            worker.join(timeout=90)

        failed = [r for r in results if not r["ok"]]
        self.assertGreater(len(results), 0, "no requests completed at all")
        self.assertEqual(
            failed,
            [],
            f"{len(failed)}/{len(results)} request(s) aborted during a "
            f"transient {stall_duration:.1f}s bootstrap stall "
            f"(< interval + timeout = {miss_cycle:.1f}s): {failed[:5]}",
        )
        self.assertNotIn(
            self.LOST_CONNECTION_CANARY,
            self._decode_log_since(log_snapshot),
            "decode mass-killed rooms during a transient bootstrap stall",
        )

        # ---- Phase 2: real prefill death is detected within the window ----
        log_snapshot = self._decode_log_snapshot()
        kill_results = []
        kill_stop = threading.Event()
        kill_workers = [
            threading.Thread(
                target=self._request_worker,
                args=(100 + i, kill_results, kill_stop),
                daemon=True,
            )
            for i in range(4)
        ]
        for worker in kill_workers:
            worker.start()
        time.sleep(1)  # rooms in flight at kill time

        # The fixture's fail-fast watcher would abort the whole test run when
        # a server process dies; the kill below is intentional.
        if self._fail_fast_stop is not None:
            self._fail_fast_stop.set()
        kill_process_tree(self.process_prefill.pid)

        # Worst case: max_failures misses, each costing up to
        # interval + probe timeout, plus scheduling slack.
        detection_window = miss_cycle * self.HEARTBEAT_MAX_FAILURES + 22.0
        deadline = time.monotonic() + detection_window
        detected = False
        while time.monotonic() < deadline:
            if self.LOST_CONNECTION_CANARY in self._decode_log_since(log_snapshot):
                detected = True
                break
            time.sleep(1)
        kill_stop.set()
        self.assertTrue(
            detected,
            f"decode did not mark the dead prefill lost within "
            f"{detection_window:.0f}s (rooms would hang until the 300s "
            f"bootstrap/waiting timeouts)",
        )

        # In-flight rooms must resolve (as failures) promptly after
        # detection, not linger toward the 300s per-request timeouts.
        for worker in kill_workers:
            worker.join(timeout=60)
        hung = [w for w in kill_workers if w.is_alive()]
        self.assertEqual(
            len(hung),
            0,
            f"{len(hung)} request worker(s) still blocked after the "
            "heartbeat declared the prefill dead",
        )


class TestDisaggregationMooncakeSpec(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_TARGET_MODEL_EAGLE3
        spec_args = [
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            DEFAULT_DRAFT_MODEL_EAGLE3,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "4",
            "--speculative-num-draft-tokens",
            "16",
            "--cuda-graph-max-bs",
            "8",
            "--dtype=float16",
        ]
        cls.extra_prefill_args = spec_args
        cls.extra_decode_args = [
            *spec_args,
            "--disaggregation-decode-enable-radix-cache",
        ]
        cls.launch_all()

    def test_gsm8k(self):
        decode_info = requests.get(f"{self.decode_url}/server_info", timeout=10).json()
        self.assertFalse(decode_info.get("disable_radix_cache", True))

        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.74)


class TestDisaggregationSimulatedRetract(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ["SGLANG_TEST_RETRACT"] = "true"
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("SGLANG_TEST_RETRACT")
        super().tearDownClass()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.62)


class TestDisaggregationPauseResumePrefillLeak(PDDisaggregationServerBase):
    """Regression test: pause_generation must not leak prefill requests into
    running_batch.  With a small --max-running-requests the leak fills the
    scheduling budget and blocks all subsequent prefills."""

    MAX_RUNNING = 4

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = [
            "--max-running-requests",
            str(cls.MAX_RUNNING),
            "--enable-metrics",
        ]
        cls.launch_all()

    def test_retract_pause_no_leak_on_prefill(self):
        """Retract-mode pause on a disagg prefill node must not leak prefill
        requests into running_batch. Without the fix, each retract pause merges
        last_batch into running_batch, but the prefill event loop never cleans
        them up via update_running_batch. After enough cycles the
        max-running-requests budget is exhausted and all new prefills hang."""
        asyncio.run(self._run_pause_resume_leak_test("retract"))

    def test_retract_pause_empty_running_batch(self):
        """Retract-mode pause must not crash when running_batch is empty.
        Regression test for issue #20272."""
        asyncio.run(self._run_pause_on_idle("retract"))

    async def _run_pause_on_idle(self, mode):
        """Pause/resume on an idle prefill node (no in-flight requests)."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.prefill_url + "/pause_generation",
                json={"mode": mode},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
            async with session.post(
                self.prefill_url + "/continue_generation",
                json={},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()

            # Verify the engine still works after pause/resume
            async with session.post(
                self.lb_url + "/generate",
                json={
                    "text": "What is 1+1?",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                body = await resp.json()
                self.assertIn("text", body)
                self.assertGreater(len(body["text"]), 0)

    async def _get_num_running_reqs(self, session):
        """Query sglang:num_running_reqs from prefill node's /metrics."""
        async with session.get(
            self.prefill_url + "/metrics",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            resp.raise_for_status()
            text = await resp.text()
            for line in text.splitlines():
                # Match the gauge line, skip HELP/TYPE comments and
                # per-priority breakdowns (which have priority="<int>")
                if (
                    line.startswith("sglang:num_running_reqs{")
                    and "priority=" not in line
                ):
                    return int(float(line.split()[-1]))
            return 0

    async def _run_pause_resume_leak_test(self, mode):
        NUM_WORKERS = 64
        NUM_PAUSE_RESUME_CYCLES = self.MAX_RUNNING * 4
        MAX_NEW_TOKENS = 1
        LONG_PROMPT = "Tell me a story. " * 200

        async def _background_worker(session, worker_id, cancel_event):
            """Send requests sequentially until cancelled."""
            seq = 0
            while not cancel_event.is_set():
                try:
                    async with session.post(
                        self.lb_url + "/generate",
                        json={
                            "text": f"[w{worker_id}-{seq}] {LONG_PROMPT}",
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": MAX_NEW_TOKENS,
                            },
                        },
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        await resp.read()
                except Exception:
                    pass
                seq += 1

        async def _post(session, url, json_data):
            async with session.post(
                url,
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()

        cancel_event = asyncio.Event()

        async with aiohttp.ClientSession() as session:
            workers = [
                asyncio.create_task(_background_worker(session, i, cancel_event))
                for i in range(NUM_WORKERS)
            ]

            for _ in range(NUM_PAUSE_RESUME_CYCLES):
                await _post(
                    session,
                    self.prefill_url + "/pause_generation",
                    {"mode": mode},
                )
                await _post(
                    session,
                    self.prefill_url + "/continue_generation",
                    {},
                )
                await asyncio.sleep(0.1)

            # Stop workers and abort all in-flight requests
            cancel_event.set()
            await _post(
                session, self.prefill_url + "/abort_request", {"abort_all": True}
            )
            await _post(
                session, self.decode_url + "/abort_request", {"abort_all": True}
            )
            await asyncio.gather(*workers, return_exceptions=True)

            # Wait for abort cleanup, then check for leaked phantom requests.
            # With the bug, running_batch accumulates phantom prefill requests
            # that are never cleaned up.
            await asyncio.sleep(2)
            num_running = await self._get_num_running_reqs(session)
            self.assertEqual(
                num_running,
                0,
                f"Prefill node has {num_running} phantom running requests "
                f"after abort — pause_generation is leaking into running_batch",
            )


if __name__ == "__main__":
    unittest.main()
