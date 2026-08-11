"""PD multimodal integration test: mm x multi-tokenizer.

Covers the core shape of the K2.6 PD incident (PJ job 2765018.3), which
previously had zero CI coverage: a VLM behind prefill/decode disaggregation
with ``tokenizer_worker_num > 1``, so image requests carry mm feature
tensors through the MultiTokenizerRouter relay on both the prefill and
decode side, exercising the raw-frame mm transport end-to-end.
"""

import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import openai
import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.mmmu_vlm_kit import MMMUMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)
from sglang.test.vlm_utils import IMAGE_MAN_IRONING_URL

register_cuda_ci(est_time=500, stage="base-b", runner_config="2-gpu-large")


class TestPDMultimodalMultiTokenizer(MMMUMixin, PDDisaggregationServerBase):
    """VLM + PD disaggregation + tokenizer_worker_num=2.

    Uses 2 GPUs: prefill on GPU 0, decode on GPU 1.
    """

    TOKENIZER_WORKER_NUM = 2

    # Qwen2.5-VL-3B-Instruct scores ~0.40 on the 50-sample MMMU subset
    # (same threshold as the EPD tests in this directory).
    accuracy = 0.40
    mmmu_args = ["--limit", "50"]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST
        # A healthy pairing completes in seconds; the 300s production default
        # only stretches each failed-pairing test (x the CustomTestCase retry
        # count) from minutes into a half-hour. Entered before launch_all so
        # the server subprocesses inherit the env.
        cls._env_stack = ExitStack()
        cls._env_stack.enter_context(
            envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.override(30)
        )
        cls._env_stack.enter_context(
            envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.override(30)
        )
        cls.addClassCleanup(cls._env_stack.close)
        cls.launch_all()

    @classmethod
    def _incident_shape_args(cls):
        return [
            "--trust-remote-code",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tokenizer-worker-num",
            str(cls.TOKENIZER_WORKER_NUM),
        ]

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--disaggregation-mode",
            "prefill",
        ] + cls._incident_shape_args()
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--disaggregation-mode",
            "decode",
            "--base-gpu-id",
            "1",
        ] + cls._incident_shape_args()
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def _image_request(self, prompt_text, max_tokens=128):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.lb_url}/v1")
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_MAN_IRONING_URL},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def test_incident_configuration_active(self):
        """Guard against silent coverage loss: if a future change drops or
        rejects the multi-tokenizer flag in PD mode, this test would
        otherwise keep passing while no longer covering the router relay."""
        for side, url in (("prefill", self.prefill_url), ("decode", self.decode_url)):
            info = requests.get(url + "/server_info", timeout=10).json()
            self.assertEqual(
                info.get("tokenizer_worker_num"),
                self.TOKENIZER_WORKER_NUM,
                f"{side}: tokenizer_worker_num not applied: "
                f"{info.get('tokenizer_worker_num')}",
            )

    def test_image_description(self):
        """Single image request through the LB: mm tensors traverse the
        multi-tokenizer router on both P and D sides and the output must
        still be grounded in the image."""
        text = self._image_request("Describe this image in a sentence.")
        print(f"[PD-mm] Image response:\n{text}")
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        text_lower = text.lower()
        self.assertTrue(
            any(w in text_lower for w in ("man", "person", "driver")),
            f"Image response should mention a person: {text}",
        )
        self.assertTrue(
            any(w in text_lower for w in ("iron", "cloth", "hang", "holding")),
            f"Image response should mention ironing/clothes: {text}",
        )

    def test_concurrent_image_requests(self):
        """Concurrent image requests: with 8 in-flight requests spread across
        both tokenizer workers, responses must route back through the correct
        worker and no request may be lost or corrupted in the relay."""
        num_requests = 8
        prompts = [
            f"Request {i}: describe what you see in this image in one sentence."
            for i in range(num_requests)
        ]
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            texts = list(executor.map(self._image_request, prompts))

        for i, text in enumerate(texts):
            self.assertIsNotNone(text, f"Request {i} returned no content")
            self.assertGreater(
                len(text.strip()), 0, f"Request {i} returned empty content"
            )

    # test_mmmu (accuracy vs baseline) is inherited from MMMUMixin.


if __name__ == "__main__":
    unittest.main()
