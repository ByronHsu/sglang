import math
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.scheduler_components import metrics_reporter


def test_fwd_occupancy_survives_active_window_boundary():
    reporter = metrics_reporter.SchedulerMetricsReporter.__new__(
        metrics_reporter.SchedulerMetricsReporter
    )
    reporter.forward_pass_device_timer = Mock()
    reporter._device_timer_window_batch_count = 0
    reporter._device_timer_window_gpu_time = 0.0
    reporter._device_timer_window_start = None
    reporter.fwd_occupancy = 42.0
    reporter.scheduler = SimpleNamespace(
        server_args=SimpleNamespace(decode_log_interval=2)
    )

    with (
        patch.object(metrics_reporter, "ENABLE_METRICS_DEVICE_TIMER", True),
        patch.object(
            metrics_reporter.time, "perf_counter", side_effect=[10.0, 11.0, 12.0]
        ),
    ):
        reporter.update_device_timer()
        assert reporter.fwd_occupancy == 42.0

        reporter._device_timer_window_gpu_time = 0.5
        reporter.update_device_timer()
        assert reporter.fwd_occupancy == 50.0

        reporter.update_device_timer()
        assert reporter.fwd_occupancy == 50.0

        reporter.reset_device_timer_window()
        assert math.isnan(reporter.fwd_occupancy)
