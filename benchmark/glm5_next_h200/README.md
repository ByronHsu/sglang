# GLM-5.3 Flash H200 comparison

This directory is the reproducibility record for comparing the curated backport with upstream `aa8c950a3df62b6642c4ea60a93a5e3eb1a1450e`. Use two separate clean worktrees and the exact model revision in `manifest.json`.

For each worktree, capture the environment commands listed in the manifest, then start the server with its common arguments. Append the MTP arguments only for `low-latency-mtp` and the MTP accuracy run. Do not set `SGLANG_SIMULATE_ACC_LEN`.

Run full accuracy with:

```bash
python benchmark/gsm8k/bench_sglang.py \
  --num-questions 1319 --num-shots 5 --parallel 128 \
  --temperature 0 --top-p 1 --max-new-tokens 32768 \
  --enable-thinking --tokenizer-path "$MODEL_PATH" \
  --backend srt --host 127.0.0.1 --port 30000 \
  --result-file benchmark/glm5_next_h200/raw/accuracy-summary.jsonl \
  --raw-result-file benchmark/glm5_next_h200/raw/accuracy.jsonl
```

`--enable-thinking` is required: the pinned tokenizer then emits the checkpoint's
default `Reasoning Effort: Max` chat template. Raw completion prompts are not a
valid accuracy comparison for this model.

For every performance row, call `POST /flush_cache`, discard two runs, and retain three runs. A representative measured command is:

```bash
python -m sglang.bench_serving \
  --backend sglang --dataset-name random --random-range-ratio 1 \
  --random-input-len 1024 --random-output-len 256 \
  --max-concurrency 16 --num-prompts 80 --seed 20260827 \
  --output-file benchmark/glm5_next_h200/raw/performance.jsonl
```

Record `avg_spec_accept_length` from `/get_server_info` after each real MTP run, peak memory from `nvidia-smi`, and any request failures, worker restarts, NaNs, or CUDA-graph errors from the server log. Preserve unedited command output under `raw/`; update `raw_results.status` only after both implementations complete the entire matrix.

## Current validation status

The focused regression suite passes. All four official thinking-enabled GSM8K
runs exceed 97%. A dependency-preserving MTP recheck on 2026-08-31 scored
97.35% for the backport and 97.12% for pinned upstream, a 0.23 percentage-point
gap in the backport's favor. The backport stop rate was 99.85%; every recorded
stop rate exceeds 99.5%.

Real NEXTN acceptance also passes. In the dependency-preserving two-warmup,
three-measurement 1,024/256 recheck, concurrency 1 remains within the relaxed
10% gate: throughput retention is 95.5%, TTFT is 7.3% better, and TPOT is 9.1%
worse. At concurrency 16, TTFT is within the gate at 7.0% worse, but throughput
retention is 88.0% and TPOT is 16.4% worse.

The recheck did not change or overlay dependencies. The backport used its pinned
Torch 2.11 and sglang-kernel 0.4.3 environment; pinned upstream used its native
Torch 2.13 and sglang-kernel 0.4.6.post1 environment. Cross-loading binary
kernels is not ABI-compatible. Keep this PR in draft until the remaining
high-concurrency and long-context rows are resolved or explicitly accepted.

The optional image tower was validated separately with `--enable-multimodal`
using the same backport dependencies. Three generated solid-color PNGs were
identified as red, blue, and green, and a text-only request returned `4` for
`2+2`. The run had no request failures, NaNs, CUDA errors, or worker restarts.
Video preprocessing remains out of scope.
