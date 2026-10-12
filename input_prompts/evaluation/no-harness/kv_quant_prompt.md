# Task: FP8 KV cache memory modeling

## Working instructions

Work from the prepared simulator directory and follow its native contributor and architecture guidance. Implement the requested feature and validate it using native commands and task-local diagnostics.

Ask me when required semantics are missing; choose implementation details within the declared scope. Continue authorized work directly. All paths in this request are relative to the prepared simulator directory unless stated otherwise.

Use prepared source, supplied inputs and public authoritative technical sources as needed. Do not consult other trials or historical implementations. Keep implementation changes local and uncommitted; do not push or publish.

## Goal and scope

Add FP8 KV cache memory modeling for Qwen3-32B to the supplied simulator. FP8 must change internal memory state and memory-sensitive admission/eviction decisions. This task models resource behavior, not exact allocator implementation or FP8 compute speedups.

Preserve the target model geometry, including the explicit `head_dim = 128`; deriving it from hidden size and attention-head count gives 80 and undercounts memory. Inspect the model configuration and input descriptions in `inputs/metadata.json`. Supplied model fields are task facts, not newly measured observations. FP8 KV storage and active-memory observations remain to be implemented.

Support the feature within this single-instance scope, including TP2. Parameterize memory by model shape, KV dtype, TP, and block size rather than embedding values for this one scenario. Multi-instance/P-D export and changes to kernel throughput are outside scope. Preserve existing prefix caching and non-FP8 behavior.

## Target scenario and supplied inputs

These are fixed target inputs; do not substitute values from an unrelated scenario or request reconfirmation.

| Setting | Value |
|---|---|
| Model | Qwen3-32B |
| GPUs | 2 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB per GPU |
| `device_mem` | 102641958912 bytes per device |
| `link_bw`, `link_latency` | 16000000000.0 bytes/s, 0.00002 seconds |
| TP, block size | 2, 16 |
| Memory utilization | 0.9 |
| `max_batch`, `max_num_batched_tokens` | 128, 8192 |
| `max_model_len` | 32768 |
| Prefix caching | Enabled |
| Operator latency table | `inputs/latency.csv` |
| Input metadata | `inputs/metadata.json` |
| Request stream | `inputs/workload.jsonl`; use the selected native execution format |

Paths are relative to the prepared simulator directory. `metadata.json` describes the selected native inputs and profile coverage. Profiling-only settings do not override the target scenario above. The supplied profile contains Qwen3-32B measurements for the target hardware; verify the needed coverage and interpretation from the actual files. Ask me for missing target facts. The scale-metadata approximation and implementation details below are delegated choices; this does not authorize changing the target settings or output semantics.

## Required behavior

- **Memory mechanism:** derive KV storage from dtype and model geometry. For FP8, include a small, documented, configurable, nonzero per-block quantization-metadata overhead. Its approximation is your choice within this task; include it in capacity, admission, and eviction accounting.
- **State evolution:** account for allocation, decode growth, sharing, completion, and eviction. Prefix-cached and private KV must use a consistent dtype-aware model. Use the existing prefix-cache mechanism.
- **Metric semantics:** active usage counts bytes in physical KV blocks with at least one live reference. Cached-but-inactive blocks with zero references remain resident and evictable, consume capacity, and are excluded from active usage. Shared references do not multiply a physical block's bytes. Total residency drives memory decisions.
- **Capacity denominator:** export `usage_bytes` and `usage_pct`, with the latter on the 0-100 scale. The denominator is allocatable KV capacity after memory utilization and model-reserved bytes, not raw device memory. Document any unmodeled reservation rather than presenting an assumption as measurement.
- **Observability:** expose the current usage state and an optional separate CSV in long format: `time,instance_id,usage_bytes,usage_pct`. Sample at a configurable simulated-time interval, include t=0 and end-of-run, and observe state after all events at that timestamp. Avoid duplicate final samples. Use integer instance IDs and the single-instance per-device view, without whole-system aggregation. Keep normal result output separate.
- **Runtime cost:** avoid adding prefix-ancestor walks, prefix-key vector copies, or full usage recomputation on each lock/unlock or scheduler iteration. Use inexpensive state updates and defer expensive observation aggregation to sample boundaries. Observation off/on must produce exactly the same request results and simulated completion time. Keep actual execution time close; measure repeated matched runs and report the overhead, including CSV output. Also compare the baseline and changed implementation under the same conditions. Historical absolute wall-time figures are not an acceptance threshold.

## Task-specific evidence

Connect each check to its expected behavior:

| Check | Expected observation |
|---|---|
| Dtype and metadata arithmetic | For matching model/token state, FP8 uses fewer bytes than BF16/FP16 while including its nonzero metadata overhead. Changing supported model shape, TP, or block size changes the result according to the documented formula. |
| Memory pressure | A boundary case admits an additional request with FP8 when BF16 cannot, or demonstrates correspondingly earlier eviction with BF16. The difference reaches the scheduler decision. |
| Growth, completion, and eviction | Decode growth increases usage; completion releases active usage. A completed but cached block remains resident until eviction, which reduces residency. |
| Prefix sharing | Multiple live references count a shared physical block once. Cached and private KV use the same dtype-aware accounting. |
| Non-FP8 regression | Disabling FP8 preserves existing supported behavior. |
| CSV sampling | Columns and post-event sample values match the specified semantics, including t=0, interval boundaries, and a final sample without duplication. Enabling CSV preserves result/event behavior. |

Run the supplied full workload with the target scenario and record completion, observation coverage, and runtime cost with CSV off and on. Select further diagnostics based on observed failures or uncertainty.

Keep the baseline's native engine observations usable. No independent real-system validation trace is supplied. Check memory observations and sampling directly, report empirical limits, and note any additional real reference data needed without collecting it.

## Artifacts

Retain the working implementation, runnable target scenario, native checks, commands, input identities, raw observations, and a validation report under `tasks/kv-quant/`. Document consequential modeling assumptions and incomplete behavior.

## borg simulator execution context

The prepared baseline supports `single_instance`, `chunked_prefill` and the dense `qwen3` model. Use that system and scheduler with prefix caching enabled. Run `inputs/workload.jsonl` with `inputs/latency.csv`; table-lookup profile paths are relative to the native config file directory.

`inputs/metadata.json` preserves the profiling setup for the supplied Qwen3-32B FP8-KV table. Its profiling-only layer count, TP bootstrap and scheduler settings do not override the target deployment. Verify the needed lookup coverage and interpretation from the files.

Keep the native engine trace usable. The separate memory CSV has no declared Borg KV comparison mapping; do not relabel its percentage as a different observation or claim empirical accuracy from synthetic checks.
