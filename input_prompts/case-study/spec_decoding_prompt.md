# Task: Speculative decoding in Borg

## User request

Extend Borg to support EAGLE-style speculative decoding with prefix caching, and validate the implementation against the supplied real-system measurements using trace-guided validation.

## Required outputs and validation

Preserve existing behavior when speculation is disabled. Run trace-guided validation on all three supplied development cases and report both runtime discrepancies and final performance metrics.

When speculation is enabled, export both timing arrays in each native request result, in seconds. This request extends the baseline result schema; update the affected result interfaces and their documentation to satisfy it:

- `round_latency`: one entry per emission round. Entry 0 is TTFT from request arrival; later entries are intervals between successive emissions.
- `token_latency`: one entry per committed output token. For a round lasting D seconds that emits M tokens, append M entries of D/M, not `[D, 0, ...]`.

Both arrays must account for the same elapsed time and reconstruct request completion. Compare TTFT, TPOT, and ITL using the round intervals, as defined in `../tools/TGV.md`. Generation counters must count committed output tokens. Recording observations must not change simulated timing or request results.

## Workflow and workspace

Work in the prepared simulator directory. Read its `AGENTS.md` and `ARCHITECTURE.md`. Follow `.agents/skills/task-design/SKILL.md`, `.agents/skills/sim-mapping/SKILL.md`, `.agents/skills/sim-dev/SKILL.md`, then the independent `.agents/skills/sim-validation/SKILL.md`, loading each at its stage. Save the specification, implementation map, validation report and supporting artifacts under `tasks/spec-decoding/`.

Workloads, profiles, acceptance statistics, algorithm references and measured serving configurations are available under `inputs/`. Real-system traces and final performance measurements are under `../references/`. The configurations under `tasks/spec-decoding/` are baseline starting points, not completed feature implementations.

Use `../tools/SIMULATOR_TOOLS.md` for the simulator adapter, `../tools/TRACE_COMPARISON_TOOLS.md` for common comparison tools, and `../tools/TGV.md` for the supplied trace bundles and bounded execution commands. Read the output and metric definitions in `../tools/TGV.md` during task-design. Read `../tools/VALIDATION_WORKFLOW.md` during validation and record assessments at `.borg/validation-state.json`.

Use task-design to identify what this request and the supplied artifacts leave unspecified, and ask me about the missing requirements and modeling choices before finalizing the specification or starting dependent implementation. Resolve one question at a time with me, including any follow-up needed before moving to the next. Do not select defaults or declare assumptions delegated without an explicit answer. End the turn with your current question and wait for my reply in this same session. Choose internal implementation details within the agreed semantics autonomously.

Use the supplied repository and authoritative technical references. Keep input artifacts unchanged, preserve prior validation rounds, and keep changes local and uncommitted. Do not consult other trials or collect new real-system measurements. Follow the execution limits in the supplied tool guide.
