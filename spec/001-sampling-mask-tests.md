# Sampling-mask end-to-end tests

Status: validated on one H200 node (2026-08-11)

Feature spec: [000-sampling-mask.md](000-sampling-mask.md)

Model: `Qwen/Qwen3-8B`

Use one H200 node. Run both tests with the same checkpoint, sampling backend, deterministic-inference setting, request payload, and sampling seed.

Shared request:

```json
{
  "text": "Write a short explanation of tensor parallelism.",
  "sampling_params": {
    "temperature": 1.0,
    "top_k": 64,
    "top_p": 0.9,
    "sampling_seed": 1234,
    "max_new_tokens": 16,
    "ignore_eos": true
  },
  "return_sampling_mask": true,
  "return_logprob": true,
  "top_logprobs_num": 64
}
```

For a response with `T` output tokens, both tests require:

```text
len(output_ids)
  == len(output_token_sampling_mask)
  == len(output_token_sampling_logprobs)
  == output_token_sampling_mask_length
  == T
```

Every support is non-empty, contains its sampled token, has at most 64 IDs, and has a finite selected-token sampling logprob.

## Test 1: unified serving

Topology:

```text
client
  |
unified SGLang server
  |  Qwen/Qwen3-8B
  |  TP=1
  |  one GPU
  v
response
```

Procedure:

1. Launch one unified server with deterministic inference and sampling-mask capture enabled.
2. Send the shared request.
3. Send the same request again with `return_sampling_mask=false`.
4. Save the mask-enabled response as the reference for Test 2.

Assertions:

- Both requests return HTTP 200.
- Enabling mask return does not change `output_ids`.
- The common alignment and membership assertions pass.
- For each token, reconstruct the selected-token probability from the returned top logprobs restricted to the returned support and compare it with `output_token_sampling_logprobs[t]` within the existing logprob tolerance.

## Test 2: PD serving with one prefill and one decode worker

Topology:

```text
                         +------------------+
client -> PD router ---> | prefill worker P |  GPU 0
                         | Qwen/Qwen3-8B    |
                         +--------+---------+
                                  |
                                  | first token + KV handoff
                                  | sampling mask + sampling logprob
                                  v
                         +------------------+
                         | decode worker D  |  GPU 1
                         | Qwen/Qwen3-8B    |
                         +--------+---------+
                                  |
                                  v
                              response
```

Configure both workers with the same sampling-mask cap and enough PD metadata capacity for `top_k=64`.

Procedure:

1. Launch one prefill worker and one decode worker on separate GPUs.
2. Route the shared request through the PD endpoint.
3. Compare the PD response with the saved unified response.

Assertions:

- The request returns HTTP 200.
- The common alignment and membership assertions pass.
- Token 0 has a non-empty mask and finite sampling logprob transferred from prefill to decode.
- Tokens 1 through `T-1` have masks and logprobs produced by decode.
- The handoff introduces no missing, duplicated, or shifted metadata rows.
- With the same deterministic configuration and seed, PD and unified `output_ids`, support rows, and sampling logprobs match within the existing numeric tolerance.
