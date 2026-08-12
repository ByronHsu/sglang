# Sampling-mask replay

Status: implemented

Scope: expose faithful sampling-support metadata from SGLang for general downstream use.

## 1. End-to-end data workflow by component

For each generated token, transport one atomic record:

```text
(y_t, S_t, log q_sample(y_t))

y_t = sampled token
S_t = discrete token IDs with positive sampling weight
```

Weights for every support token are not transported.

```text
Client
  | GenerateReqInput
  |   sampling_params
  |   return_sampling_mask=true
  v
TokenizerManager
  | tokenize prompt; preserve flag
  | TokenizedGenerateReqInput
  v
Scheduler
  | validate; create Req
  | batch Req rows into SamplingBatchInfo
  v
Model / LogitsProcessor
  | logits [batch, vocab]
  v
Sampler
  | temperature -> top-k -> top-p -> min-p
  | sample y_t
  | capture S_t and log q_sample(y_t)
  v
SamplingMaskOutput on GPU
  | batch_indices, token_ids, lengths, logprobs, statuses
  | opted-in rows only
  v
GenerationBatchResult
  | async D2H
  | result_queue: (batch.copy(), result)
  v
BatchResultProcessor
  | validate and atomically append (y_t, S_t, logprob)
  v
OutputStreamer
  | BatchTokenIDOutput containing unsent chunks
  v
DetokenizerManager
  | decode y_t; pass mask IDs/logprobs unchanged
  | BatchStrOutput
  v
TokenizerManager
  | accumulate by request ID; build meta_info
  v
Downstream consumer
  | use ragged support directly or encode it sparsely
  | optionally replay the sampled distribution
```

Overlap scheduling:

```text
time ------------------------------------------------------>

forward stream: [forward/sample N] [forward/sample N+1]
                          |
                          +--- event ---> [D2H N]  copy stream

CPU scheduler:  [process N-1] ---------> [process N]
                      |                       |
                      v                       v
              commit triple N-1       commit triple N
```

D2H must not serialize the next CUDA forward. CPU result processing waits for copy_done before reading the mask. If overlap produces a stale token after finish or retraction, discard its token, mask, logprob, and temporary KV state together.

Invariant at every boundary:

```text
len(output_ids)
  == len(output_token_sampling_mask)
  == len(output_token_sampling_logprobs)
```

## 2. End-to-end code changes by file

```text
Request API
  entrypoints/openai/{protocol.py,serving_chat.py}
        |
        v
Request and IPC types
  managers/io_struct.py
  managers/{tokenizer_manager.py,multi_tokenizer_mixin.py}
  session/session_controller.py
        |
        v
Scheduler and batch state
  managers/{schedule_batch.py,scheduler.py}
  sampling/sampling_batch_info.py
        |
        v
Capture
  layers/{logits_processor.py,sampler.py} *
        |
        v
Tensor result and overlap D2H
  managers/{utils.py,scheduler.py}
        |
        v
Validate and commit
  managers/scheduler_components/batch_result_processor.py *
        |
        v
Chunk output
  managers/scheduler_components/output_streamer.py
  managers/io_struct.py
        |
        v
Decode text and pass metadata
  managers/detokenizer_manager.py
        |
        v
Accumulate response
  managers/tokenizer_manager.py
```

Side paths:

```text
server_args.py
  -> mask-size cap -> sampler / scheduler

scheduler_pp_mixin.py
  -> PP tensor transport

disaggregation/{prefill.py,decode.py,utils.py}
  -> PD first-token metadata and wire capacity

[periodiclabs/sgl-router-for-miles#20](https://github.com/periodiclabs/sgl-router-for-miles/pull/20)
  -> preserve return_sampling_mask through typed unified and PD /generate routing
```

## 3. Detailed code changes

### 3.1 API, types, and request state

Files:

```text
entrypoints/openai/{protocol.py,serving_chat.py}
managers/{io_struct.py,tokenizer_manager.py,multi_tokenizer_mixin.py}
managers/schedule_batch.py
session/session_controller.py
```

Add return_sampling_mask to GenerateReqInput, TokenizedGenerateReqInput, and Req. Support /generate and /v1/chat/completions; chat requires return_meta_info=true. Leave /v1/completions out of scope.

Req stores cumulative masks/logprobs and an offset for the next unsent row. Session and multi-tokenizer copies preserve the flag and metadata.

The router integration lives in [periodiclabs/sgl-router-for-miles#20](https://github.com/periodiclabs/sgl-router-for-miles/pull/20), not this repository. Its typed SGLang `/generate` request preserves the optional flag through unified routing and both PD stage requests.

For T output tokens, meta_info returns:

```text
output_token_sampling_mask:         list[list[int]]  # [T][M_t]
output_token_sampling_logprobs:     list[float]      # [T]
output_token_sampling_mask_length:  int              # T
```

The flag-off path remains unchanged.

### 3.2 Scheduler and batch state

Files:

```text
managers/scheduler.py
sampling/sampling_batch_info.py
```

Admit a mask request when any truncation is active:

```python
top_p < 1.0 or top_k > 0 or min_p > 0.0
```

Reject an explicit full-vocabulary mask request. Greedy may return a singleton.

SamplingBatchInfo carries per-row opt-in flags through create, merge, and filter. A normal row sharing a batch with a mask row performs no capture or mask D2H.

### 3.3 Faithful sampler capture

Files:

```text
layers/{logits_processor.py,sampler.py}
```

For actual filtered sampling weights w_t:

```text
S_t = {token_id | w_t[token_id] > 0}
log q_t(y_t) = log(w_t[y_t] / sum(w_t))
```

Return one final support, not separate top-k and top-p masks.

- PyTorch reuses the filtered weights and token permutation passed to multinomial.
- FlashInfer uses its actual top-k/top-p/min-p support.
- Greedy returns [y_t] and logprob 0.
- Do not reconstruct after sampling or append y_t to repair a mismatch.

Return a tensor-backed SamplingMaskOutput:

```text
batch_indices       int32[R]
token_ids           int32[R, cap]
lengths             int32[R]
selected_logprobs   float32[R]
statuses            int32[R]

R = opted-in rows
```

### 3.4 Bounds, overlap, and distributed paths

Files:

```text
server_args.py
managers/{scheduler.py,utils.py,scheduler_pp_mixin.py}
managers/scheduler_components/batch_result_processor.py
disaggregation/{prefill.py,decode.py,utils.py}
```

Add --sampling-mask-max-tokens=4096. The cap applies to one output position.

- OVERFLOW: support exceeds the cap; fail that request with HTTP 400.
- INVALID: sampled token is outside support or the logprob is invalid.
- Never truncate support or change sampling to meet the cap.
- Reach TP/CP consensus before divergent scheduler control flow.
- Commit no token or associated state on failure.

On CUDA, copy SamplingMaskOutput to pinned CPU memory on a dedicated copy stream and wait only when consuming the result. Keep (batch.copy(), result) paired so row ownership survives overlap batch mutation. Audit delayed-sampling and HIP same-stream paths. If fixed [R, cap] D2H is not hidden, replace it with compact CSR D2H.

PP carries SamplingMaskOutput in tensor metadata. PD carries the first-token support/logprob/status to decode, bounded by min(server cap, wire capacity). Preserve the fork's unconditional send_kv_chunk behavior.

Reject MLX, Ascend, custom, and speculative producers. Add DFlash separately.

### 3.5 Commit, stream, and detokenize

Files:

```text
managers/scheduler_components/{batch_result_processor.py,output_streamer.py}
managers/{io_struct.py,detokenizer_manager.py,tokenizer_manager.py}
```

BatchResultProcessor waits for copy_done, restores compact rows to batch rows, validates status, and atomically appends token/mask/logprob.

OutputStreamer slices unsent mask rows with send_output_sampling_mask_offset and includes them in BatchTokenIDOutput.

DetokenizerManager decodes output token IDs but passes mask IDs and logprobs unchanged into BatchStrOutput. TokenizerManager accumulates chunks and exposes the response fields.

### 3.6 Downstream consumer contract

Downstream integration is outside this repository. SGLang returns token-aligned ragged support IDs and one selected-token sampling logprob per position. Consumers may keep the ragged form or encode it as IDs plus offsets.

A policy-replay consumer applies the same replayable transforms, masks its current logits to S_t, and computes:

```
log q_current(y_t) = log_softmax(mask(z_current, S_t))[y_t]
ratio_t = exp(log q_current(y_t) - log q_sample(y_t))
```

With identical source and consumer weights, ratio is 1 and the sampled-distribution mismatch is 0 within tolerance. Missing or misaligned support is an error; consumers must not silently fall back to a full-vocabulary normalization.

## 4. References

PR state and SHA are pinned as of 2026-08-11.

| PR                                                                | State / SHA        | Use                                                           |
| ----------------------------------------------------------------- | ------------------ | ------------------------------------------------------------- |
| [SGLang #27408](https://github.com/sgl-project/sglang/pull/27408) | merged, a909077d22 | Base request, sampler, API, PP/PD, and response plumbing.     |
| [SGLang #32108](https://github.com/sgl-project/sglang/pull/32108) | merged, 955dbcc24d | #27408 backport adapted to the internal fork; port this base. |
| [SGLang #33593](https://github.com/sgl-project/sglang/pull/33593) | open, 109119f137   | Top-p-only admission and chat API.                            |
| [SGLang #34037](https://github.com/sgl-project/sglang/pull/34037) | open, 32beb1deda   | Faithful bounded capture and atomic failure.                  |
| [SGLang #32410](https://github.com/sgl-project/sglang/pull/32410) | merged, 5f330004bd | Legacy top_k+1 repair; superseded by #34037.                  |
| [SGLang #34201](https://github.com/sgl-project/sglang/pull/34201) | draft, 68839a8343  | DFlash support; defer.                                        |
| [slime #2102](https://github.com/THUDM/slime/pull/2102)           | merged, 8f5e215194 | Independent end-to-end reference.                             |
| [sgl-router-for-miles #20](https://github.com/periodiclabs/sgl-router-for-miles/pull/20) | draft, 32522da | Typed unified and PD `/generate` passthrough. |

Integration order:

1. Port #32108 as the internal-fork base.
2. Overlay pinned #33593 and #34037; #34037 owns sampler semantics.
3. Add DFlash #34201 separately.
