# Task: EAGLE speculative decoding with prefix caching

## Working instructions

Work from the prepared simulator directory. Use `.agents/skills/task-design/SKILL.md`, then `.agents/skills/sim-mapping/SKILL.md`, then `.agents/skills/sim-dev/SKILL.md`, followed by the independent `.agents/skills/sim-validation/SKILL.md`. Read each skill when entering its stage. Produce the spec and implementation map before dependent implementation, and carry them through validation in the same session.

Ask me when required semantics are missing; choose implementation details within the declared scope. Continue authorized work directly. All paths in this request are relative to the prepared simulator directory unless stated otherwise.

Use prepared source, supplied inputs and public authoritative technical sources as needed. Do not consult other trials or historical implementations. Keep implementation changes local and uncommitted; do not push or publish.

The selected adapter guide is `../tools/SIMULATOR_TOOLS.md`; common comparison commands are in `../tools/TRACE_COMPARISON_TOOLS.md`. Read them when checking observation support or using their tools. Keep supplied tools unchanged and report failures. The sim-validation skill selects trace-guided or reference-guided validation and routes findings back to development without a fixed iteration count.

Save `spec.md`, `implementation-map.md` and `validation-report.md` in the task-local output directory specified in the request. Include design decisions and assumptions in the relevant artifact, and retain runnable target and diagnostic scenarios, commands, identities, raw observations and round records. Report incomplete required behavior and empirical limits explicitly.

Where the selected adapter supports engine observations, keep those observations usable and exercise native export, `pair`, and `validate` on a small permitted workload. Label a copied self-reference and a deliberately perturbed trace as synthetic tool checks. Confirm zero difference for identical observations and a visible, localized difference for the perturbation; a perturbation to one instance must remain local to its signals. Retain source rows and concurrent signals in the interpretation. Use default evidence mode without the optional acceptance gate. These checks establish tooling behavior only. Where no conformance-tested engine mapping exists, record engine comparison as unavailable and use the native diagnostics described below.

## Requested feature and target

Add EAGLE-style speculative decoding to the supplied simulator with correct coexistence with the existing prefix cache. Read the supplied EAGLE paper for the algorithmic foundation; it is required reading. `inputs/EAGLE_paper.txt` provides searchable text from `inputs/EAGLE_paper.pdf`; consult the PDF for equation/layout details. Preserve the existing behavior when speculation is disabled. Parameterize the mechanism so that changing valid K, model, workload, TP degree, or hardware parameters does not require code changes; document the supported range.

| Parameter | Value |
|---|---|
| Model | Qwen3-32B, existing dense model |
| Model geometry | Hidden=5120, intermediate=25600, 64 layers, 64 attention heads, 8 KV heads, explicit head_dim=128, vocabulary=151936 |
| GPUs | 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB each |
| Device memory / utilization | 102641958912 bytes per GPU / 0.9 |
| TP | 2 |
| Link bandwidth / latency | 16000000000 bytes/s / 0.00001 seconds |
| Prefix caching | Enabled |
| Block size / max batch / token budget | 16 / 128 / 8192 |
| Speculation | EAGLE, K=`num_speculative_tokens`=2 for this target |
| Draft-token latency (required assumption) | 0.02 ms per generated draft token for a single request |

Use the following inputs for the target run:

- `inputs/workload.jsonl`: the 400-request ShareGPT training workload, including original arrival times and token IDs.
- `inputs/latency.csv`: measured Qwen3-32B operator timings with TP=1,2,4 coverage, collected with eager execution.
- `inputs/spec_decoding_params.jsonl`: measured, section-keyed acceptance statistics for the target workload and K=2.
- `inputs/EAGLE_paper.pdf` and `inputs/EAGLE_paper.txt`: algorithm reference and its text extraction.
- `inputs/metadata.json`: artifact descriptions, actual coverage, and evidence limits.

Workload output token IDs represent the predetermined committed stream; they are not draft proposals or acceptance measurements. Preserve the supplied input bytes and identities. Use separate, labeled diagnostic workloads/configurations when needed.

## Draft, verify, and acceptance behavior

Model chain drafting by a lightweight head on the same device: K sequential draft candidates, one target forward over K+1 tokens, then sequential acceptance that stops at the first rejection. A round produces M=A+1 tokens, where A is the number of accepted draft tokens; handle the requested output boundary explicitly.

Document the engine-step pipeline: one target verification plus the next draft phase's K substeps, with drafts from step N verified in step N+1. Resolve startup, final-round, and emission timing from the algorithm and declared target semantics. Preserve dependencies and charge work through native simulation events. A verification is one wide query, not K+1 independent decode passes. Each speculative decode request consumes K+1 scheduled tokens from the batch token budget, with the applicable terminal behavior documented.

Acceptance must be stochastic and use per-position conditional probabilities; do not substitute a fixed acceptance length. Load the supplied statistics defensively, allowing absent optional sections and ignoring unknown sections. Recognize `metadata`, `summary`, `mean_acceptance_len_distribution`, `load_conditioned_statistics`, and optional `time_series`. Prefer the measured load-conditioned statistics over one global mean, using concurrent running requests as their load signal.

In these artifacts, `mean_acceptance_len` and `spec_mean_acceptance_len` mean accepted draft tokens A, so E[M]=E[A]+1. The distribution summarizes one-second interval means, not individual rounds. The current file has four sections and no `time_series` or direct per-position probabilities. Specify a justified stochastic fallback when only aggregates are available, and label the probability reconstruction as a modeling assumption. Conditional probabilities and their cumulative prefix products have different meanings. Explain how unsupported K/statistic combinations are handled; K=2 aggregate measurements do not determine an arbitrary K distribution.

Use a fixed draft latency of **0.02 ms per generated draft token for a single request**, including rejected tokens and excluding target verification and request queue waiting. Apply this modeling assumption in all target and evaluation runs, and document how it scales with batch size.

## Prefix cache, memory, and timing outputs

Use the existing concrete prefix cache. Preserve valid committed-prefix reuse while preventing pending or rejected draft tokens from becoming reusable prefixes. Distinguish verified work, committed progress, computed KV, and private lookahead slots. Pre-allocate K lookahead slots, retain reusable reservation for overwrite after rejection, and account for block rounding, memory pressure, completion, and preemption. Define startup and terminal cleanup without leaking or double-counting state.

Maintain two distinct timing views when speculation is enabled:

- `token_latency`: for a round of duration D emitting M tokens, append M entries of D/M for throughput and completion-time reconstruction. Do not encode `[D, 0, ...]`.
- `round_latency`: one entry per emission round; entry 0 is speculative TTFT and later entries are intervals between rounds. This is the comparison view for callback-level ITL, subject to matching reference semantics.

Define activation/emission boundaries and waiting/prefill/draft attribution so elapsed time is accounted for exactly once, including startup and completion. Preserve non-speculative per-token semantics and export round timing through native results. Any output-token counters introduced or changed for this feature must count committed output tokens. Observation off/on must leave request results and simulated completion time unchanged.

## Validation and deliverables

Run the full Qwen3-32B TP=2 target with the supplied workload, measured target profile, and acceptance-statistics loader. Retain the effective draft-cost parameter and statistical fallback assumptions in the scenario and report.

Use controlled checks for all-accept/all-reject/mixed acceptance, deterministic seeded replay, several valid K values, missing/unknown statistics sections, load-bucket selection, and correct E[A] interpretation. Check actual verification width and cost, token-budget pressure, mixed prefill/decode, lookahead admission/release, prefix reuse, rejected-prefix exclusion, output tails, and both latency-array cardinality and elapsed-time accounting. Preserve disabled-speculation behavior and run relevant native tests. Small synthetic mechanism checks may supplement the target run.

Preserve the baseline's native engine observations. No independent end-to-end reference engine trace is supplied; existing operator measurements and acceptance summaries are model inputs. If a later reference is timeout-truncated, do not compare its duration as though all requests completed.

If further diagnosis needs real reference data, state the need briefly and finish without collecting it. Retain the working implementation, target and diagnostic scenarios, commands, input identities, observations, and a validation report under `tasks/spec-decoding/`. Report incomplete required behavior and empirical limits explicitly.

## borg simulator execution context

Use the existing dense `qwen3` model and `chunked_prefill` scheduler with prefix caching enabled. Run `inputs/workload.jsonl` with `inputs/latency.csv` and `inputs/spec_decoding_params.jsonl`; table-lookup profile paths are relative to the native config file directory.

Read `inputs/profile.yaml` for the exact profiling setup and layer mapping. Its one-layer override and bootstrap settings do not replace the target model or deployment. The receipt lists TP=1,4; the actual CSV also contains TP=2.

Preserve native engine observations. Engine generation counters must count committed output tokens.
