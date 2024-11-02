import logging
import random
from enum import Enum, auto
from typing import Dict, List

import httpx

from sglang.srt.router.worker import Worker
from sglang.srt.router.radix_tree import RadixTree

logger = logging.getLogger(__name__)


class BaseRouter:
    def __init__(self, server_urls: List[str]):
        self.worker_list: List[Worker] = []
        self.server_url_to_worker: Dict[str, Worker] = {}
        self._init_worker_list(server_urls)

    ####################
    # Public Method
    ####################
    def if_exist(self, server_url: str) -> bool:
        return server_url in self.server_url_to_worker

    def remove_worker(self, server_url: str):
        for worker in self.worker_list:
            if worker.server_url == server_url:
                self.worker_list.remove(worker)

        self.server_url_to_worker.pop(server_url, None)

    def add_worker(self, server_url: str):
        if server_url in self.server_url_to_worker:
            raise ValueError(f"Worker with url {server_url} already exists")
        worker = Worker()
        worker.server_url = server_url

        # disable timeout (setting to None)
        worker.client = httpx.AsyncClient(base_url=server_url, timeout=None)
        self.worker_list.append(worker)
        self.server_url_to_worker[server_url] = worker

    def calc_priority(self) -> Worker:
        raise NotImplementedError

    ####################
    # Private Method
    ####################
    def _init_worker_list(self, server_urls):
        for server_url in server_urls:
            self.add_worker(server_url)


class RandomRouter(BaseRouter):
    def calc_priority(self) -> Worker:
        idx = random.choice(self.worker_list)
        return self.worker_list[idx]



from sglang.srt.managers.io_struct import GenerateReqInput
from dataclasses import asdict
import json

class RoundRobinRouter(BaseRouter):
    def __init__(self, server_urls: List[str]):
        super().__init__(server_urls)
        self.idx = -1

    def calc_priority(self) -> Worker:
        ...

    async def dispatch(self, obj: GenerateReqInput):
        self.idx = (self.idx + 1) % len(self.worker_list)
        worker = self.worker_list[self.idx]
        res = await worker.client.post("/generate", json=asdict(obj))
        return res


class ApproxTreeRouter(BaseRouter):
    def __init__(self, server_urls: List[str]):
        super().__init__(server_urls)
        self.url_to_tree = {url: RadixTree() for url in server_urls}
        self.url_to_count = {url: 0 for url in server_urls} # count the in-processing requests for workers
        from sglang.srt.hf_transformers_utils import get_tokenizer
        self.tokenizer = get_tokenizer("/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/")
    
    def calc_priority(self, input_ids):
        ...

    async def dispatch(self, obj: GenerateReqInput):
        """
        1. Match with each radix tree and select the highest match
        2. If the match is above the threshold, send the request to the worker, if not send the request with the shortest request queue
        3. Before the request is sent, insert the request into the radix tree
        4. After the request returned, remove the request from the radix tree and insert the cached response into the radix tree
        """

        # TODO: cached_tokens seems to be 1 digit off when perfectly matched 
        
        input_ids = self.tokenizer.encode(obj.text)
        # print("input_ids", input_ids)

        THRESHOLD = 0.80
        highest_rate = float("-inf")
        highest_url = None

        for url, tree in self.url_to_tree.items():
            matched_id = tree.prefix_match(input_ids)

            # print("input_id_len", len(input_ids))
            # print("matched_id_len", len(matched_id))

            rate = len(matched_id) / len(input_ids)
            if rate > highest_rate:
                highest_rate = rate
                highest_url = url
        
        # print("highest rate", highest_rate)
        # print("highest url", highest_url)

        selected_url = None
        if highest_rate > THRESHOLD:
            selected_url = highest_url
        else:
            # select the worker with the shortest queue
            selected_url = min(self.url_to_count, key=self.url_to_count.get)

        # insert input_ids to the selected tree
        self.url_to_tree[selected_url].insert(input_ids)
        self.url_to_count[selected_url] += 1
        res = await self.server_url_to_worker[selected_url].client.post("/generate", json=asdict(obj))

        # import pdb; pdb.set_trace()

        cached_tokens = json.loads(res.content)["meta_info"]["cached_tokens"]
        # print("cached_tokens", cached_tokens)

        # remove input_ids from the selected tree
        self.url_to_tree[selected_url].delete(input_ids)
        # insert the cached part of input_ids to the selected tree
        self.url_to_tree[selected_url].insert(input_ids[:cached_tokens])
        self.url_to_count[selected_url] -= 1

        # self.url_to_tree[selected_url].pretty_print()

        # print(self.url_to_count)

        return res

# {"text":
#   " thousands of years ago, probably soon after the Flood, a very curious man named Enmeduranki, a Temple Priest of the Chaldeans, climbed up a mountain called Temek.\nEveryone climbed tall mountains. From the summit one could see what happened to the parallel world, called the Nephilim world, and what was deferred. Most of the peaks in that world were sacred mountains, and the high angels that resided there were called ‘Watchers’. The Sumerian priest was advised to go to the mountain by the big god, Ea, but there were some precautions and questions. Enmeduranki, a good",
#   "meta_info": {
#       "prompt_tokens":6,
#       "completion_tokens":128,
#       "completion_tokens_wo_jump_forward":128,
#       "cached_tokens":1,
#       "finish_reason":{"type":"length","length":128},
#       "id":"16b2e316c4174845a59a012bcb3c6a96"
#   },
#   "index":0
# }

# Extend your router here
class RoutingPolicy(Enum):
    ROUND_ROBIN = auto()
    RANDOM = auto()

    @classmethod
    def from_str(cls, policy: str):
        policy = policy.upper()
        try:
            return cls[policy]
        except KeyError as exc:
            valid_options = ", ".join(member.name for member in cls)
            raise ValueError(
                f"Invalid routing policy: {policy}. The valid options are {valid_options}"
            ) from exc


def get_router_class(policy_name: str):
    policy = RoutingPolicy.from_str(policy_name)

    if policy == RoutingPolicy.ROUND_ROBIN:
        return RoundRobinRouter
    elif policy == RoutingPolicy.RANDOM:
        return RandomRouter

"""
curl -X POST http://127.0.0.1:8080/generate  -H "Content-Type: application/json" -d '{
    "text": "Kanye west is, ",
    "sampling_params": {
      "temperature": 0
    }
  }'


curl -X POST http://127.0.0.1:8080/generate  -H "Content-Type: application/json" -d '{
    "text": "CUDA MODE means, ",
    "sampling_params": {
      "temperature": 0
    }
  }'
"""