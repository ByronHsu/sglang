import unittest
from types import SimpleNamespace

from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _LoadObserved(Exception):
    pass


class TestQwenTokenIdLoader(unittest.IsolatedAsyncioTestCase):
    async def test_pre_tokenized_prompt_uses_id_preserving_loader(self):
        processor = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
        processor.mm_tokens = object()
        input_ids = [100, 101, 102]
        observed: dict[str, object] = {}

        async def observe_load(**kwargs):
            observed.update(kwargs)
            raise _LoadObserved

        async def reject_legacy_load(**kwargs):
            self.fail("Qwen pre-tokenized prompts must not use the legacy loader")

        processor.load_mm_data = observe_load
        processor.legacy_load_mm_data = reject_legacy_load
        request = SimpleNamespace(video_data=None, audio_data=None)

        with self.assertRaises(_LoadObserved):
            await processor.process_mm_data_async(
                image_data=["unused"],
                input_text=input_ids,
                request_obj=request,
            )

        self.assertIs(observed["prompt"], input_ids)


if __name__ == "__main__":
    unittest.main()
