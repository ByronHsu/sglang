"""Start bootstrap/kv-store-related server"""

import logging
import multiprocessing as mp
import os
import sys
import time

from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# How long the child polls its own /health endpoint before declaring startup
# failed. Must be comfortably smaller than _PARENT_READY_TIMEOUT_S so the
# child's error report always reaches the parent before the parent gives up.
_CHILD_HEALTH_TIMEOUT_S = 25.0
# How long the parent waits for the child's readiness report. The spawned
# child re-imports sglang (torch etc.) before it can report, so this budgets
# for import time on top of _CHILD_HEALTH_TIMEOUT_S. Genuine failures (bind
# errors, child death) surface immediately; only a hung child waits this long.
_PARENT_READY_TIMEOUT_S = 60.0


class BootstrapServerProcHandle:
    """Handle to the PD bootstrap server running in a dedicated subprocess.

    Call sites only retain the returned object (to prevent GC) and never call
    server methods on it, so this exposes lifecycle operations only.
    ``close()`` mirrors the bounded-shutdown semantics of the in-thread
    ``CommonKVBootstrapServer.close()`` (stop, wait up to 2s, then force).
    """

    def __init__(self, proc: mp.Process):
        self.proc = proc

    def is_alive(self) -> bool:
        return self.proc.is_alive()

    def close(self):
        if not self.proc.is_alive():
            return
        self.proc.terminate()
        self.proc.join(timeout=2)
        if self.proc.is_alive():
            self.proc.kill()


def _self_health_url(server_args: ServerArgs) -> str:
    host = server_args.host
    # A wildcard bind address is not reliably connectable; probe via loopback.
    if host in ("0.0.0.0", ""):
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # IPv6 literal
    return f"http://{host}:{server_args.disaggregation_bootstrap_port}/health"


def _run_bootstrap_server_process(server_args: ServerArgs, ready_writer):
    """Entry point of the dedicated PD bootstrap-server process.

    The backend-specific bootstrap server class is resolved here, in the
    child, so the parent only pickles ``server_args`` (spawn-safe: plain
    dataclass, no torch/CUDA state).
    """
    import requests
    import setproctitle

    from sglang.srt.utils.common import configure_logger, kill_itself_when_parent_died

    setproctitle.setproctitle("sglang::disagg_bootstrap_server")
    kill_itself_when_parent_died()
    configure_logger(server_args)

    try:
        transfer_backend = TransferBackend(server_args.disaggregation_transfer_backend)
        kv_bootstrap_server_class = get_kv_class(
            transfer_backend, KVClassType.BOOTSTRAP_SERVER
        )
        bootstrap_server = kv_bootstrap_server_class(
            host=server_args.host,
            port=server_args.disaggregation_bootstrap_port,
        )

        # The aiohttp server runs on a daemon thread inside this process; a
        # bind failure only logs there (``CommonKVBootstrapServer._run_server``)
        # and leaves the server silently dead. Poll our own /health endpoint so
        # startup failures surface loudly in the parent via the readiness pipe.
        health_url = _self_health_url(server_args)
        deadline = time.monotonic() + _CHILD_HEALTH_TIMEOUT_S
        while True:
            if not bootstrap_server.thread.is_alive():
                raise RuntimeError(
                    "bootstrap server thread exited during startup "
                    "(port already in use? see logs above)"
                )
            try:
                if requests.get(health_url, timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"bootstrap server did not answer {health_url} within "
                    f"{_CHILD_HEALTH_TIMEOUT_S}s"
                )
            time.sleep(0.05)
    except Exception as e:
        logger.exception("PD bootstrap server startup failed")
        ready_writer.send({"status": "error", "message": f"{type(e).__name__}: {e}"})
        ready_writer.close()
        sys.exit(1)

    ready_writer.send({"status": "ready"})
    ready_writer.close()
    # This process exists solely to host the bootstrap server; block on its
    # serving thread forever.
    bootstrap_server.thread.join()


def _start_bootstrap_server_subprocess(
    server_args: ServerArgs,
) -> BootstrapServerProcHandle:
    """Spawn the bootstrap server in its own process and wait for readiness."""
    ctx = mp.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_run_bootstrap_server_process,
        args=(server_args, writer),
        name="disagg_bootstrap_server",
        daemon=True,
    )
    proc.start()
    # Close the parent's copy of the write end so the reader can observe EOF
    # if the child dies before reporting.
    writer.close()
    handle = BootstrapServerProcHandle(proc)

    msg = None
    deadline = time.monotonic() + _PARENT_READY_TIMEOUT_S
    try:
        while True:
            if reader.poll(0.1):
                try:
                    msg = reader.recv()
                except EOFError:
                    pass  # child died without reporting
                break
            if not proc.is_alive():
                # Child died between polls; drain any buffered error report.
                if reader.poll(0):
                    try:
                        msg = reader.recv()
                    except EOFError:
                        pass
                break
            if time.monotonic() > deadline:
                break
    finally:
        reader.close()

    if msg is None or msg.get("status") != "ready":
        addr = f"{server_args.host}:{server_args.disaggregation_bootstrap_port}"
        if msg is not None:
            reason = msg.get("message", str(msg))
        elif not proc.is_alive():
            reason = f"process died with exit code {proc.exitcode}"
        else:
            reason = f"no readiness report within {_PARENT_READY_TIMEOUT_S}s"
        handle.close()
        raise RuntimeError(
            f"PD bootstrap server failed to start on {addr}: {reason}. "
            f"A common cause is the bootstrap port already being in use."
        )

    logger.info(
        f"PD bootstrap server subprocess ready "
        f"(pid={proc.pid}, port={server_args.disaggregation_bootstrap_port})"
    )
    return handle


def start_disagg_service(
    server_args: ServerArgs,
):
    # Start kv bootstrap server on prefill
    disagg_mode = DisaggregationMode(server_args.disaggregation_mode)
    transfer_backend = TransferBackend(server_args.disaggregation_transfer_backend)

    if disagg_mode != DisaggregationMode.PREFILL:
        return None

    # Only start the bootstrap server on the prefill tokenizer manager/router.
    # It is the PD control plane (/health, /route, dp-rank registry), so by
    # default it runs in a dedicated subprocess where GIL/CPU pressure in this
    # process cannot delay it.
    if envs.SGLANG_DISABLE_BOOTSTRAP_SERVER_SUBPROCESS.get():
        # Kill switch: legacy behavior — run the server as a daemon thread
        # inside this process.
        kv_bootstrap_server_class = get_kv_class(
            transfer_backend, KVClassType.BOOTSTRAP_SERVER
        )
        bootstrap_server = kv_bootstrap_server_class(
            host=server_args.host,
            port=server_args.disaggregation_bootstrap_port,
        )
    else:
        bootstrap_server = _start_bootstrap_server_subprocess(server_args)

    # The Ascend memfabric store is independent of the bootstrap server object
    # and stays in this (parent) process.
    is_create_store = (
        server_args.node_rank == 0 and transfer_backend == TransferBackend.ASCEND
    )
    if is_create_store:
        try:
            from memfabric_hybrid import create_config_store

            ascend_url = os.getenv("ASCEND_MF_STORE_URL")
            create_config_store(ascend_url)
        except Exception as e:
            error_message = f"Failed create mf store, invalid ascend_url."
            error_message += f" With exception {e}"
            raise error_message

    return bootstrap_server
