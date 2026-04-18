---
name: task-design
description: Use when writing or reviewing a task spec that will be given to an implementation agent. Guides task design through interactive Q&A so the agent can succeed on the first attempt without relying on general harness principles.
---

# Task Design

This skill helps you write task specs for implementation agents working in this simulator repository. It works as an **interactive conversation** — the user describes what they want, and you fill in the gaps by researching, asking, and proposing until the spec is complete.

## Core Principle

**The spec's job is to make task-specific facts — intent, scope, metric definitions, mechanism semantics — concrete and unambiguous.** These are things the implementation agent cannot know without being told, and cannot be pre-encoded in any general harness.

Generic implementation guardrails (how to verify model configs, avoid statistical-proxy shortcuts, model hardware topology correctly, reuse existing modules) belong in the implementation-time harness (`.agents/skills/sim-development`), not here. Do not duplicate that content in the spec — a spec that pre-bakes general harness material absorbs the harness's role and undermines the separation of concerns.

Turn every task-specific fact that matters into a concrete, falsifiable instruction the agent cannot misinterpret. Leave generic implementation principles to sim-dev.

## This Is a Simulator

This repository is a **discrete-event simulator** for LLM serving. Users may want to model:

1. **Existing real-system behavior** — e.g., "model vLLM's FP8 KV cache memory behavior"
2. **Novel ideas with no real-world artifact** — e.g., "model a hypothetical scheduling policy I designed"
3. **Hybrid** — e.g., "follow vLLM's metric definition, but model a mechanism that doesn't exist in vLLM yet"

The spec must be **self-contained**. When the user references an external system (vLLM, SGLang, etc.), extract the specific semantics the simulator needs and write them into the spec as standalone definitions. Do not make the spec depend on the implementation agent reading external source code.

**What this means in practice:**
- If the user says "follow vLLM's KV cache usage definition" → write the definition in the spec ("active usage excluding refcount-zero cached blocks"), do not link to vLLM source files and expect the agent to figure it out
- If the user describes a novel mechanism → the spec IS the ground truth; there is no external system to reference
- Do not include links to external source code repositories. The spec should define semantics, not delegate them to external code that may change or that the agent may over-emulate.

## When To Use This Skill

- Before writing a new task spec from scratch
- Before handing an existing spec to an agent for the first time
- After an agent fails a task — to diagnose what the spec should have said

---

## Information Boundaries

Before doing any research or drafting, understand what you may and may not use.

### What you MUST get from the user

These cannot be inferred, defaulted, or looked up. If the user has not provided them, **ask explicitly**.

1. **Feature intent** — what to build and why it matters
2. **Target hardware and serving config** — GPU, TP degree, memory, interconnect, engine launch parameters. The user must either state these or point to a specific config file to use. Do NOT default to whatever config happens to exist in the repo.
3. **Input workload** — which workload to use. The user decides this, not you.
4. **Expected simulator output** — what the simulator should return or export after implementation. This includes:
   - What metric(s) the simulator must produce
   - The precise definition of each metric — written out in the spec, not delegated to an external reference
   - Required granularity (e.g., per-iteration, per-request, end-of-run aggregate)
   - Any distinction between internal state and exported metric semantics (e.g., if the simulator tracks total residency internally but should export an "active usage" view)
5. **Scope boundary** — what level of realism is expected vs. what is out of scope

**Important framing:** Ask the user what the simulator should *produce*, not how it will be *evaluated*. The spec should describe required simulator behavior and output, not reverse-engineer a validation pipeline.

### When something is ambiguous: know when to ask vs. when to decide

Not all ambiguities are equal. Distinguish between **user-owned decisions** and **implementation details**.

**STOP and ask** for user-owned decisions — things where a wrong choice changes what the simulator does or produces:
- Metric definitions, output semantics, config values, scope boundaries
- Anything where two reasonable choices lead to meaningfully different simulator behavior

**Make a reasonable choice yourself** for implementation details — things where the user does not care as long as the behavior is sensible:
- Edge-case boundary behavior (e.g., before vs. after processing at exact timestamps)
- Data format minutiae (e.g., column ordering, header naming)
- Internal bookkeeping granularity choices

When you make an implementation-detail choice yourself, briefly document it as an explicit assumption in the spec so the implementation agent knows it was intentional. Do not ask the user about every micro-decision.

Making a "conservative assumption" for **user-owned decisions** and noting it in the spec is **not acceptable**. The whole point of this skill is to get those decisions right before an implementation agent sees the spec.

### How to ask: present options, not open-ended questions

When you need user input, do not ask open-ended questions that force the user to compose an answer from scratch. Instead, **present 2-4 concrete options** with your recommendation, and let the user pick or override.

**Bad:**
> "What format should the KV usage CSV use?"

**Good:**
> I need to decide the CSV format for KV usage export.
>
> ⭐ **(A) `Long format`** — *recommended*
> One row per `(time, instance_id)` sample. Handles multi-instance without schema changes.
>
> **(B) `Wide format`**
> One row per timestamp, one column per instance. Compact but harder to extend.
>
> **(C) `One CSV per instance`**
> Separate files per instance. Clean separation but more file management.
>
> Pick one, or describe something different.

**Rules for presenting options:**
- Provide 2-4 options. More than 4 creates decision fatigue.
- Each option label uses **bold** + `` `code` `` for visual contrast against surrounding text.
- Mark the recommended option with ⭐ and *recommended*. Non-recommended options get bold + code formatting but no emoji marker.
- Each option gets a one-sentence tradeoff description on the next line.
- Always include an escape hatch: "or describe something different."
- Batch related decisions into one message when possible, so the user can resolve multiple ambiguities in one response instead of going back and forth.

This pattern keeps the feedback loop fast: the user can reply "(A)" instead of writing a paragraph. It also ensures you have resolved all ambiguities before starting to write the spec.

**Critical: block all other work while waiting for user input.** When you present options, stop everything — do not continue researching, reading code, or drafting other sections in parallel. The user needs to read your options and decide without their screen scrolling away. Resume work only after the user responds.

### What you CAN research yourself

These are safe to look up and propose for user confirmation:

- **Codebase structure** — what code paths exist, what state is tracked, what decisions are memory-sensitive, whether prefix caching exists, what traits/registries to use
- **Artifact schemas** — open real workload/stats files and document their actual format

**Do not pre-verify model architecture from HuggingFace.** Record the model name the user provides. Fetching `config.json`, comparing derived vs explicit dimensions, and surfacing divergences like `head_dim` is the implementation harness's responsibility at implementation time. If you do it here, you duplicate sim-dev's role and the spec becomes fragile (hardcoded values in the spec that may diverge from reality, and that remove the implementation agent's need to read the real config itself).

### What you must NOT touch

- **Validation data, evaluation scripts, or answer keys.** These exist to evaluate the entire process. If you read them and encode their content into the spec, the evaluation becomes meaningless.
- **Hidden evaluator implementations.** Do not inspect them or reverse-engineer validation behavior from them.

### Config values: the dangerous middle ground

Existing scenario configs in the repo (`col.toml`, etc.) are **not safe to adopt by default**. They may:
- Target different hardware than what the user wants
- Contain incorrect or outdated values
- Have been written for a different experiment with different assumptions

**Rules for config values:**
1. If the user says "use the config in `scenarios/X/col.toml`" → use it, but still verify critical values with the user
2. If the user provides explicit values (e.g., "TP4, A100-80GB, block_size 16") → use those
3. If the user names hardware but not specific params → research specs for that hardware and propose values for confirmation. Do NOT silently copy from existing repo configs.
4. If the user says nothing about hardware/config → **ask**. Do not fill this from existing configs.

The general rule: **code structure is safe to reference; config values require user authority.**

---

## Workflow: From Natural Language to Complete Spec

### Phase 1 — Gather Intent

Accept whatever the user provides. Identify which of the "MUST get from user" items are covered and which are missing.

Do not ask for everything at once. Prioritize the most critical gaps.

### Phase 2 — Research and Fill Gaps

For things you CAN research (model architecture, codebase structure, artifact schemas), do so proactively and present findings for confirmation.

For things you MUST get from the user, ask with specific options when possible.

**Research-first pattern:**

| Gap | Do NOT do this | Do this instead |
|-----|----------------|-----------------|
| Codebase state | "Does the simulator have prefix caching?" | Read the scheduler code, then say: "The simulator has prefix caching in `chunked_prefill`. The spec should require FP8 consistency for prefix-cached blocks too. Agreed?" |
| Artifact schema | "What format is the workload?" | Open the actual file the user specified, read its header/first rows, and document what you see. |

**Ask-the-user pattern (for things that require user authority):**

| Gap | Do NOT do this | Do this instead |
|-----|----------------|-----------------|
| Hardware config | Silently copy from existing `col.toml` | "What hardware are you targeting? I see an existing scenario with 4x A100-80GB TP4, but I can't assume that's your target." |
| Workload | Pick a workload from `workloads/` | "Which workload should the agent use for this task?" |
| Expected output | Guess from feature description | "After implementing this feature, what should the simulator export? What metric, what definition, what granularity?" |
| Serving engine config | Copy from existing scenario | "What are the serving engine params? I won't assume the existing configs are correct for this task." |

### Phase 3 — Draft and Iterate

Write the spec section by section. After each major section, briefly state what you wrote and any assumptions you made. The user can correct or approve as you go.

### Phase 4 — Final Review

Run through the Quality Checks (at the bottom). Flag anything still missing or assumed.

---

## Spec Sections

### 1. Purpose and Scope

State what the agent must build and what it must not build.

**Checklist:**
- [ ] One sentence stating the goal
- [ ] Explicit non-goals (what looks related but is out of scope)
- [ ] "Simulator task vs exact emulation" boundary — how much realism is expected

**Bad:** "Implement FP8 KV cache support."
**Good:** "Implement FP8-aware KV cache *memory modeling*. This is a simulator task: the goal is decision-relevant memory state, not allocator-level byte accuracy. Kernel throughput changes are out of scope."

### 2. Mechanism Background

If the feature is based on a published algorithm, paper, or external system behavior, provide a self-contained conceptual summary before getting into requirements.

This section gives the implementation agent a **mental model** — not just a list of rules, but an understanding of *why* the rules exist. An agent that understands the mechanism can handle edge cases the spec didn't anticipate. An agent that only has rules will break on the first uncovered situation.

**Checklist:**
- [ ] The core algorithm or mechanism is summarized in 1-2 pages
- [ ] Key invariants the simulator must preserve are stated as properties, not code
- [ ] The summary is self-contained — the agent should not need to read the original paper to understand what to implement, though reading it deepens understanding
- [ ] If the mechanism has phases or a loop structure, each phase is described with its inputs, outputs, and relationship to adjacent phases
- [ ] Terminology is defined here so later sections can use it without ambiguity
- [ ] **⚠️ MANDATORY — Concrete computation trace.** This is the single most important item in this checklist. Specs without a worked trace have a ~50% chance of off-by-one or off-by-factor modeling errors that silently produce wrong results. **You MUST include at least one worked example that traces a single iteration end-to-end with specific numbers.** For each step in the iteration, state: (1) exact number of tokens processed as input, (2) what computation occurs, (3) exact number of tokens produced as output, (4) what the `q_len`, `kv_len`, and `lm_head_len` values are for the simulator's compute model. Use a concrete scenario (e.g., "3 requests in decode, each with kv_len = 100, 200, 300") and walk through every step showing the real numbers. Do not leave any quantity as "K" or "N" — substitute the actual task parameter values and compute the result. This forces precision on quantities that are easy to get wrong by ±1 or ×2 in abstract descriptions. **A spec missing this trace is incomplete and must not be handed off.**
- [ ] **⚠️ MANDATORY — Cross-consistency check on the trace.** After writing the computation trace, you MUST perform an explicit self-review. For every pair of quantities in the same computation step, determine whether they are independent or derived from each other. If two quantities are derived from the same operation (e.g., both come from the same forward pass, both depend on the same input), they MUST be mutually consistent. **If two dependent quantities have different values, you MUST explain the specific mechanism that causes them to differ. If you cannot provide a concrete physical explanation, your trace contains an error — fix it before proceeding.** This is not optional. Trace inconsistencies propagate directly into implementation bugs that are silent and hard to diagnose. Common failure mode: stating two quantities that come from the same forward pass but giving them different values without justification.

**When to include this section:**
- The feature is based on a published paper → summarize the relevant algorithm
- The feature models a real system's behavior (e.g., vLLM's scheduler) → describe that behavior as the simulator needs to understand it
- The feature is a novel idea from the user → this section may be short or merged into §1

**When to skip:** Pure configuration changes or simple extensions that don't introduce a new mechanism.

**Bad:** "Read the EAGLE paper for details."
**Good:** A concise summary of draft-verify-accept loop, what each phase does, what the acceptance semantics are, and what invariants hold — written so someone who hasn't read the paper can implement correctly.

### 3. Target System Details

**Checklist:**
- [ ] GPU model, count, memory, interconnect type and topology — **from user, not from existing configs**
- [ ] Serving engine config — **from user, not from existing configs**
- [ ] Model name — **from user**. Do not pre-verify architecture dimensions against HuggingFace; that is the implementation harness's job.
- [ ] Workload location — **from user**

**How to fill this section:**

1. Hardware and serving config: get from user. If user says "same as scenario X", verify those values with user before adopting.
2. Workload: get from user. Do not pick one yourself.
3. Model: record the name the user provides. If the user volunteers architectural facts relevant to scope (e.g., "this model has MoE routing that affects memory"), record what they say. Do not go to HuggingFace yourself to look up or hardcode dimension values like `head_dim` — the implementation harness handles authoritative config verification at implementation time.

### 4. Expected Simulator Output

Describe what the simulator must return or export after this feature is implemented. This is about **simulator behavior**, not about how it will be evaluated.

**Checklist:**
- [ ] What metric(s) the simulator must produce
- [ ] Precise definition of each metric, written as a standalone definition in the spec
- [ ] Required granularity (per-iteration, per-request, end-of-run)
- [ ] If the simulator's internal state semantics differ from the exported metric, both are stated and the mapping is explicit
- [ ] Output format or channel (stats struct, log line, result field)

**If the user hasn't specified this:** Ask directly. Do not infer what the simulator should output from an evaluation pipeline.

**Bad:** "Export stats that support later validation."
**Good:** "The simulator must export per-iteration KV cache memory usage in bytes, following vLLM's definition of active KV usage (excluding refcount-zero cached blocks). This must be available as a time series, not only as an end-of-run aggregate."

### 5. Required Mechanism Behavior

Frame requirements as **observable behavioral properties**, not implementation instructions. The spec should say *what must be true*, not *how to code it*.

**Checklist:**
- [ ] Each requirement is falsifiable ("X must increase when Y happens")
- [ ] State evolution: what grows, what shrinks, on which events
- [ ] Integration points: which existing decisions must use the new model
- [ ] Consistency requirements across subsystems

**How to fill this section:**

Read the existing codebase to understand what state exists and which decisions are memory-sensitive. Frame new requirements in terms of those existing paths. This is codebase-structure research — safe to do without user input.

**Statistical modeling guidance:**

When a feature involves stochastic behavior driven by external statistics (e.g., acceptance rates, arrival distributions, cache hit patterns), the spec should push toward structured, conditional models rather than single global numbers:

- If the provided statistics include conditioning variables (e.g., load level, request phase, batch size), the spec should require the implementation to use that structure rather than collapsing to a single unconditional mean.
- Prefer load-conditioned, time-varying, or per-category statistics over one global average whenever the input data supports it.
- The spec should state what conditioning variables exist in the data, so the implementation agent knows to use them.

This does not mean inventing structure that isn't in the data. It means: if the data has richer structure than a single number, the spec should require the implementation to use that richness.

**Distinguish three levels of detail:**

| Level | In spec? | Example |
|-------|----------|---------|
| **Target system mechanism** | ✅ Yes — describe in detail | "The verify pass processes K+1 tokens in one forward pass, not K+1 separate decode steps" |
| **Navigation hints** | ✅ Yes — as an appendix or brief pointers | "The existing prefix cache lives near the scheduler" |
| **Code-level prescriptions** | ❌ No | "Add `enum SchedulerWork { ... }` to scheduler.rs", specific TOML field names, file-by-file change lists |

Labeling code-level content as "recommended" or "suggested" does not make it appropriate for the spec. The implementation agent will treat any specific file path, API signature, or data structure definition as a requirement regardless of how it is labeled.

**Target system mechanism** is the core of the spec. Describe how the real system (or the hypothetical mechanism) works in as much detail as needed. This tells the implementation agent *what to model*.

**Navigation hints** save the implementation agent time. List relevant code areas briefly, but label them as hints: "These are navigation pointers, not a mandated edit list." Do not turn them into a file-level implementation plan.

**Code-level prescriptions** are harmful because:
- The task-design agent explores the codebase but does not compile, run, or debug it. The implementation agent understands it more deeply.
- A wrong API prescription (e.g., a specific enum shape) locks the agent into a suboptimal approach.
- Specific file lists become a checklist the agent follows mechanically rather than understanding the codebase.

If you find yourself writing specific API signatures, Rust enums, TOML schemas, or file-by-file change lists, you have crossed from spec into implementation plan. Remove that content or move it to a clearly-labeled "Optional Navigation Hints" appendix.

**What to include instead:**
- DO describe what behavioral properties the implementation must satisfy
- DO describe the target system's mechanism in detail (step structure, budgeting, state transitions)
- DO name which existing simulator mechanisms must be affected (e.g., "admission checks must use the new memory model")
- DO include expected qualitative relationships for sanity-checking (e.g., "FP8 should roughly halve KV memory compared to FP16")
- DO briefly note where relevant code lives, labeled as hints

The implementation agent is an expert coder. Give it a detailed mechanism description and correct inputs — it will design the code structure itself.

### 5.1 "Good Enough" Boundary

Every spec should include an explicit quality bar — what counts as sufficient and what is overengineering.

**Checklist:**
- [ ] "Good enough" criteria: minimum behavioral properties that constitute a complete implementation
- [ ] "Not required" list: things that look tempting but are out of scope
- [ ] If the simulator already has a relevant abstraction, say "use it" rather than requiring a new one

### 6. Forbidden Implementations (Task-Specific Only)

This section is for **task-specific shortcuts** that the positive requirements in §4 and §5 do not already rule out — things the implementation agent might plausibly try that are wrong *for this feature's specific data, scope, or semantics*.

**Do NOT list generic implementation anti-patterns here.** Patterns like "don't accept a hit_rate parameter instead of implementing prefix caching", "don't alias one model as another", "don't assume `head_dim = hidden_size / num_attention_heads`", or "don't use theoretical PCIe bandwidth" are the implementation-time harness's responsibility (`.agents/skills/sim-development`). Duplicating them in per-task specs defeats the separation between task design and implementation guardrails.

**Checklist:**
- [ ] Each item is a task-specific shortcut, not a general implementation anti-pattern
- [ ] Each item explains *why* it is wrong **for this task** (the workload's data structure, the metric's semantics, the scope boundary)
- [ ] Zero items is acceptable if positive requirements already cover what matters; many tasks will have 0-2 entries

**Examples of task-specific forbidden items (good):**
- "Do not collapse acceptance rate to a single unconditional mean — the provided workload statistics are conditioned on batch load, and the implementation must preserve that structure."
- "Do not export only an end-of-run aggregate — this task's metric is defined as a per-iteration time series."
- "Do not disable prefix caching in the comparison runs — the workload is prefix-heavy and the validation assumes caching is on."

**Examples of generic anti-patterns (do NOT put here — belong in sim-dev):**
- "Don't use a speedup factor instead of the draft-verify-accept mechanism."
- "Don't assume derived `head_dim`."
- "Don't alias Qwen3 as Llama3.1."
- "Don't use theoretical PCIe bandwidth."

### 7. Artifact and Schema Details

**Checklist:**
- [ ] For each input artifact: file path, format, key fields
- [ ] For each output: required fields and their semantics
- [ ] Any legacy or compatibility quirks
- [ ] Field names were verified against actual semantics — legacy or misleading names are common in real artifacts. Cross-check field values against related fields or documentation to confirm what each field actually represents.

**How to fill this section:**

Open the real files (that the user specified) and document their actual schema. Do not describe from memory.

When documenting numeric summary fields, sanity-check the values: does the field name's plain-English reading produce plausible numbers given the task context? If a field named "length" has a mean of 1.07 but the feature allows values 0-2, consider whether the name refers to the raw count or a derived quantity. Field names in real artifacts are frequently legacy labels whose plain-English meaning no longer matches what the field actually stores.

**Important:** Include only schema *facts*. Do not repeat information already stated in other sections. If the target system config is already in §3, do not list it again here.

### 8. Self-Validation Requirements

Write validation as a **requirement checklist**, not as a sequence of numbered iterations. Each item states a property that must be verified and an expected outcome. The agent (or agent prompt) decides iteration count and order.

**Checklist for this section:**
- [ ] Each validation item is a falsifiable property with an expected outcome
- [ ] Items cover: correctness, sensitivity (feature changes behavior), consistency (subsystems agree), regression (old paths still work), observability (exported data is usable)
- [ ] A mechanism audit checklist summarizing all behavioral invariants from §4
- [ ] No prescribed iteration ordering — the spec says *what* must be verified, not *when*

### 9. Deliverables

Always include a concrete list of what the implementation agent must produce.

**Recommended minimum:**
1. Working implementation
2. Reproducible test or scenario demonstrating the feature has real behavioral effect
3. Self-validation report (iteration logs)
4. Brief design notes (key decisions and rationale)

---

## Spec Writing Discipline

### Every feature must be general

A spec describes a feature for a specific task, but **the feature itself must work for any valid configuration**. The task's specific parameters (e.g., K=2, TP=2, a particular workload) are inputs to validate against — they are not the feature's scope.

This is a universal rule, not a per-task judgment call:

- If the task uses `num_speculative_tokens = 2`, the implementation must work for any K.
- If the task targets TP=2 on a specific GPU, the implementation must work for any TP on any supported hardware.
- If the current artifact omits a field, the implementation must use it when present.
- If someone changes any config parameter to a different valid value, the simulator must handle it with zero code modifications.

**What this means for the spec:**
- Frame requirements in terms of the general mechanism, not the specific parameter values.
- When you mention specific values (e.g., "K=2 for this task"), make it clear these are the current task's inputs, not architectural constraints.
- Include an explicit statement: "This implementation must work for any valid configuration of [feature], not only the specific values in this task."
- Do not write validation criteria that only make sense for specific parameter values.

**Bad:** "For K=2, validate that acceptance_ratio ≈ mean_acceptance_len / 2."
**Good:** "For any K, validate that the acceptance sampler's empirical mean converges to the configured E[A] over many rounds."

### Define semantics in the spec, not by reference

When the user references an external system, extract the semantics the simulator needs and write them as standalone definitions:

| Do NOT do this | Do this instead |
|---|---|
| "Follow vLLM's KV cache usage metric (see source)" | "Active KV usage: bytes occupied by KV blocks with at least one live reference. Excludes cached blocks with zero references." |
| Link to 6 vLLM source files | Define the metric in one paragraph |
| "Match the behavior described in paper X" | Extract the relevant behavioral properties and write them out |

External system references are useful **context for the user** during the design conversation, but the spec itself must stand alone.

### Qualitative expectations, not derived numbers

Do not derive step-by-step formulas or pre-compute expected values in the spec. If the inputs (model config, dtype, hardware) are correctly specified, the agent can compute the numbers itself. Pre-computed numbers risk the agent hardcoding them instead of deriving them from the model.

Instead, state **qualitative relationships** the agent can sanity-check against:

| Do NOT do this | Do this instead |
|---|---|
| "kv_dim = 4 * 128 = 512, then payload = 2 * 512 * 64 * 1 = 65536" | "FP8 should roughly halve per-token KV bytes compared to FP16" |
| "FP8 capacity = 56853 blocks" | "FP8 should roughly double KV block capacity compared to FP16" |
| Derive block capacity step by step | "Under memory pressure, FP8 should admit more requests than FP16 for the same config" |
| "one K scale + one V scale per layer per block, stored as 4-byte floats" | "FP8 must include per-block metadata overhead that is documented, parameterized, and nonzero" |

This rule applies equally to overhead and metadata modeling — not just payload formulas. Prescribing a specific overhead structure (e.g., scale count, storage format, granularity) risks the agent hardcoding that structure instead of deriving it from the codebase or making its own design choice.

### Avoid redundancy across sections

Each fact should appear in exactly one section. If the target system config is in §3, other sections should say "see §3" instead of re-listing the same values.

---

## Quality Checks Before Handing Off

- [ ] **Mechanism background is present (if applicable).** The spec summarizes the relevant algorithm or system behavior so the implementation agent has a mental model, not just rules.
- [ ] **⚠️ Computation trace is present.** The mechanism background includes at least one worked example with concrete numbers tracing a full iteration. Every quantity (token counts, q_len, kv_len, lm_head_len) is computed with real numbers, not left as symbolic variables. A spec without this trace is incomplete.
- [ ] **⚠️ Cross-consistency check was performed on the trace.** Every pair of dependent quantities in the trace is mutually consistent. No two values derived from the same operation contradict each other. Any differences between dependent quantities have an explicit physical justification written in the spec.
- [ ] **Feature is general.** Requirements are framed for any valid configuration, not just the task's specific parameter values. An explicit generalization statement is included.
- [ ] **No implicit assumptions.** Every value is either stated or has a pointer to where to find it.
- [ ] **Hardware/serving config came from the user.** Not silently adopted from existing scenario configs.
- [ ] **Workload was specified by the user.** Not picked by you from the repo.
- [ ] **Expected simulator output is concrete.** Metric name, definition, granularity, and format are stated.
- [ ] **Output requirements are framed as simulator behavior.** Not as "export what the evaluator needs."
- [ ] **Semantics are self-contained.** Metric definitions are written out in the spec, not delegated to external source code links.
- [ ] **No prescriptive formulas or pre-computed numbers.** Qualitative expectations are stated, but step-by-step derivations and hardcoded target values are not.
- [ ] **No code-level prescriptions.** No specific API signatures, Rust enums, TOML schemas, or file-by-file change lists. Navigation hints are labeled as hints.
- [ ] **No redundancy.** Each fact appears in exactly one section.
- [ ] **No section-numbering collisions.** Every section and subsection has a unique number.
- [ ] **Scope and forbidden implementations are distinct.** Out-of-scope items define what is outside the task boundary. Forbidden implementations describe common mistakes *within* scope. The same item should not appear in both.
- [ ] **Forbidden implementations are task-specific only.** No generic anti-patterns duplicated from the implementation harness. Zero items is acceptable if positive requirements cover what matters.
- [ ] **Self-validation is specific.** Test scenarios with properties and expected outcomes.
- [ ] **Artifacts were inspected.** Schemas verified against real files.
- [ ] **No forbidden data was referenced.** No validation data, evaluation scripts, or answer-key files.
- [ ] **Codebase structure was surveyed.** The spec references actual code paths, not hypothetical ones.
- [ ] **Reading the harness is fallback, not primary guidance.** The spec carries everything critical.
