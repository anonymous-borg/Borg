# Task: Nemotron hybrid model and GPU memory observations

## Working instructions

Work from the prepared simulator directory. Use `.agents/skills/task-design/SKILL.md`, then `.agents/skills/sim-mapping/SKILL.md`, then `.agents/skills/sim-dev/SKILL.md`, followed by the independent `.agents/skills/sim-validation/SKILL.md`. Read each skill when entering its stage. Produce the spec and implementation map before dependent implementation, and carry them through validation in the same session.

Ask me when required semantics are missing; choose implementation details within the declared scope. Continue authorized work directly. All paths in this request are relative to the prepared simulator directory unless stated otherwise.

Use prepared source, supplied inputs and public authoritative technical sources as needed. Do not consult other trials or historical implementations. Keep implementation changes local and uncommitted; do not push or publish.

The selected adapter guide is `../tools/SIMULATOR_TOOLS.md`; common comparison commands are in `../tools/TRACE_COMPARISON_TOOLS.md`. Read them when checking observation support or using their tools. Keep supplied tools unchanged and report failures. The sim-validation skill selects trace-guided or reference-guided validation and routes findings back to development without a fixed iteration count.

Save `spec.md`, `implementation-map.md` and `validation-report.md` in the task-local output directory specified in the request. Include design decisions and assumptions in the relevant artifact, and retain runnable target and diagnostic scenarios, commands, identities, raw observations and round records. Report incomplete required behavior and empirical limits explicitly.

Where the selected adapter supports engine observations, keep those observations usable and exercise native export, `pair`, and `validate` on a small permitted workload. Label a copied self-reference and a deliberately perturbed trace as synthetic tool checks. Confirm zero difference for identical observations and a visible, localized difference for the perturbation; a perturbation to one instance must remain local to its signals. Retain source rows and concurrent signals in the interpretation. Use default evidence mode without the optional acceptance gate. These checks establish tooling behavior only. Where no conformance-tested engine mapping exists, record engine comparison as unavailable and use the native diagnostics described below.

## Requested features

Integrate `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` into the supplied simulator and add general GPU memory tracking with optional CSV export. The target is the full 52-layer hybrid model. Use its authoritative HuggingFace configuration and architecture to establish dimensions, block composition, weight shapes, and recurrent-state shapes; retain the source identities.

The configurable `hybrid_override_pattern` maps `M` to Mamba2/SSM, `*` to GQA attention, `E` to routed/shared MoE, and `-` to dense MLP. Model the ordered block composition, corresponding execution costs, and block-specific memory. The model must work with the existing MoE EP system and attention prefix cache. Generalize to valid model configurations within the documented support range.

## Target scenario and inputs

| Parameter | Value |
|---|---|
| Model | NVIDIA Nemotron-3-Nano-30B-A3B, full 52-layer hybrid configuration |
| GPUs | 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB each |
| Device memory / utilization | 102641958912 bytes per GPU / 0.9 |
| TP / EP | TP=1 per EP rank; EP=2 with two DP instances |
| Link bandwidth / latency | 16000000000 bytes/s (16 GB/s) / 0.00001 seconds |
| Block size / max batch / token budget | 16 / 128 / 2048 per scheduler |
| Prefix caching | Enabled for attention KV only |
| SSM temporal-state dtype | float32 (`mamba_ssm_cache_dtype`) |
| Memory observation interval | Configurable; exercise 1 simulated second |

Read the following inputs:

- `inputs/workload.jsonl`: the 400-request ShareGPT training workload, including arrival times and input/output token IDs.
- `inputs/latency.csv`: measured Nemotron operator costs, including Mamba and MoE rows.
- `inputs/metadata.json`: input descriptions, actual profile coverage, and evidence limits.

Inspect the native table lookup implementation and supplied profile keys. Mamba and MoE have different key semantics from dense operators. Missing cost coverage must be reported; any synthetic diagnostic table must remain separate from the supplied measured table.

## Memory behavior and observations

Derive each block's weight and runtime-state footprint from the architecture. Distinguish EP-sharded expert weights from replicated weights, attention KV growth from fixed per-request convolution/SSM state, and active usage from cached-but-inactive residency. Include runtime state in memory admission and eviction decisions, not only in printed usage. Define its allocation, retention, preemption/reconstruction, and release lifecycle.

Attention prefix caching remains enabled. Mamba recurrent state is not prefix-cached: an attention cache hit does not provide that state. Explicitly model the work and state needed to establish recurrent context when reusing attention prefixes. Document architectural approximations.

Memory tracking must work for existing models as well as the hybrid model. Maintain queryable internal active usage, excluding refcount-zero cached attention entries. For the hybrid model include live recurrent state alongside live attention KV, counting shared physical KV blocks once. Keep total residency distinct for capacity decisions.

Export an optional separate CSV with `time,instance_id,usage_bytes,usage_pct`. Report each scheduler/device independently with integer IDs 0 and 1. Define allocatable capacity from per-device memory, utilization, weights, and modeled reservations; express usage as a percentage of that capacity. Define infeasible/zero-capacity behavior clearly.

Sample at t=0, a configurable fixed interval, and final completion without duplicate final rows. Each row reflects state after all events at that timestamp; include idle instances. Export must preserve request results, event ordering, and simulated completion time exactly. Preserve the baseline's native per-instance observations. Keep actual execution time close and measure repeated matched off/on runs, including CSV output.

## Validation and deliverables

Use the original target and full supplied workload for the end-to-end run. Additional small, labeled diagnostic configurations are permitted for discriminating checks; they do not replace the target scenario.

Check all four block kinds and ordered patterns, invalid configurations, hand-computable weight/state arithmetic, attention growth versus fixed recurrent state, admission under memory pressure, completion/preemption release, and attention-prefix reuse with private recurrent state. Verify per-device EP accounting, CSV boundaries, and existing non-hybrid behavior. Check actual profile-key inputs with a case that distinguishes correct and incorrect keys. Run the relevant native tests and simulation checks.

Run the target workload with observation off/on and compare results. Preserve each instance's baseline engine signals and check the new memory CSV directly. No independent real-system engine trace is supplied. Existing measured operator costs do not establish end-to-end accuracy. If more real reference data is needed, state that limitation briefly and finish without collecting it.

Retain the working implementation, full target scenario, diagnostic scenarios, commands, input identities, observations, and a validation report under `tasks/mamba/`. Report incomplete required behavior explicitly.

## borg simulator execution context

Use `moe_ep_multi_instance` with router `moe_load_balancing`, model runner `moe_ep`, and ep=2. Each EP rank uses TP=1. Run `inputs/workload.jsonl` with `inputs/latency.csv`; table-lookup profile paths are relative to the native config file directory.

Read `inputs/profile.yaml` for the profiling setup, layer mapping and `layer_modes`. The three-layer `ME*` override profiles representative operators; it does not replace the 52-layer target, token budget or enabled prefix caching. Cross-reference `layer_modes` with the native compute simulator to determine each operator's lookup keys.

Preserve the baseline's per-instance engine signals. Check memory CSV directly because no task-specific Borg memory mapping is supplied. Build and run the relevant native tests.
