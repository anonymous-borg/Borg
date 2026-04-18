# Workloads

`workloads/*.jsonl` is the canonical simulator input format.
Legacy flat request rows are not supported.

Each line is one request object:

```json
{
  "request_id": 0,
  "arrival_time": 0.0469268,
  "initial": [0],
  "sub_requests": [
    {
      "kind": "llm",
      "input_tokens": 3,
      "output_tokens": 5,
      "next": [],
      "input_token_ids": [],
      "output_token_ids": []
    }
  ]
}
```

Each `llm` sub-request is authoritative: its `input_tokens` already represent the full input context for that step, and `input_token_ids` / `output_token_ids` describe that step directly rather than being reconstructed from predecessors. Token-id traces live on the individual `llm` sub-request so multi-sub-request requests can carry distinct prompt traces per step. They are consumed by prefix-caching schedulers.

Top-level `input_tokens` and `output_tokens` are intentionally absent. Token counts belong to each sub-request.

If a system creates a synthetic seeded child request, it may also set `known_tokens` and `kv_tokens` on that child `llm` sub-request. Fresh committed workloads normally omit those fields, which means `known_tokens = input_tokens` and `kv_tokens = 0`.

Agentic workloads use the same top-level shape, but `sub_requests` may contain multiple `llm` and `tool_call` nodes linked by `next` indices. Tool-call nodes keep only the fields the simulator actually uses:

- `initial`: ordered list of root sub-request indices that start at request arrival; it must exactly match the indegree-0 nodes implied by `next`
- `kind = "tool_call"`
- `duration` in seconds
- `input_tokens`
- `output_tokens`
- `next`

Verbose raw trace metadata is intentionally excluded from the committed simulator input.
