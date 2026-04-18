---
name: sim-development
description: Use when implementing, modifying, or debugging simulator behavior in this col.sim repository, especially under crates/col-core/. This includes adding or changing schedulers, models, model runners, systems, compute simulators, network simulators, scenario configs in col.toml, validating simulation output against real system data, and diagnosing discrepancies between simulated and real system behavior. Use before writing Rust code that changes simulator logic.
---

# Simulator Development Harness

This skill enforces development guardrails for the `col.sim` discrete-event LLM serving simulator. It captures hard-won lessons from developing and validating this simulator against real systems.

## Before You Write Any Code

Read `ARCHITECTURE.md` in the workspace root, then complete these checks:

1. Verify target system information is complete (see §1 below).
2. Verify artifact and metric semantics are explicit (see §2 below).
3. If any of that is missing, ask the user now. Do not proceed with assumptions.

---

## 1. Target System Information Must Be Complete

**This is the single most important rule.**

You must have concrete values for all of the following before writing any code:

- **Hardware:** GPU model, memory per device, number of devices, interconnect type (NVLink, PCIe gen/width), host-to-device topology
- **Serving engine config:** TP degree, max_model_len, max_num_batched_tokens, max_batch, block_size, enforce_eager, enable_prefix_caching, enable_chunked_prefill
- **Model config:** model name, num_hidden_layers, hidden_size, num_attention_heads, num_kv_heads, intermediate_size, vocab_size, and any architecture-specific dimensions that affect tensor widths or KV size such as explicit `head_dim`
- **Workload characteristics:** request arrival pattern, prompt/completion length distribution, token ID traces (if prefix caching is involved)

### When information is missing

1. **Ask the user.** Be specific: "What is the PCIe generation and width between GPUs? Are they connected through a PCIe switch or directly to the CPU?"
2. **If the user doesn't know**, consult official documentation or source code for default values, and state explicitly which defaults you are using and why.
3. **Never silently assume.** If you use a default or estimate, document it clearly.

### Why this matters

A simulator with perfect feature logic but wrong system parameters produces misleading results. In validation experiments, discrepancies were most often caused by incorrect hardware parameters (especially communication bandwidth), not by bugs in feature logic.

---

## 2. Artifact Schema, Metric Semantics, and Completion Policy

Accurate work depends on understanding:

- how input artifacts are serialized
- what summary fields actually mean
- how simulator outputs are turned into evaluation metrics
- whether reference runs are complete or truncated

If the task text gives these semantics, follow that text. If they are missing and the feature's accuracy depends on them, stop and ask the user.

### Common mistakes

- **Assuming the idealized schema exists.** Always open the real file and inspect the actual schema before coding.
- **Treating legacy field names literally.** A field like `mean_acceptance_len` can refer to accepted draft tokens rather than produced tokens.
- **Normalizing compatibility quirks in the wrong place.** Handle workload shape drift once at ingestion boundaries, not inside scheduler logic.
- **Ignoring metric projection rules.** A simulator can be mechanism-correct internally but score badly because its exported format doesn't match expectations.
- **Confusing internal state with exported metric semantics.** The simulator may need one state view for decisions and another for comparison against an external metric.
- **Comparing against the wrong reference-run type.** A timeout-truncated real run is not directly comparable to a simulator run that finishes every request.

### Rules

1. **Treat artifact loaders as compatibility boundaries.** Parse defensively, tolerate missing optional sections, normalize legacy forms once at IO boundaries.
2. **Make metric semantics explicit before coding.** If internal state and exported metric differ, document both and the mapping.
3. **Understand completion policy.** Determine whether the reference is "finish all requests", fixed-duration, or timeout-truncated. Compare like-for-like.
4. **Expose underspecified costs as assumptions.** Surface missing latency components as config knobs or documented assumptions, not hidden constants.

---

## 3. Hardware Topology and Communication Modeling

Hardware topology — especially communication paths — has an outsized impact on accuracy.

### Common mistakes

- **Using theoretical PCIe bandwidth.** In practice, PCIe contention under TP4 is far worse than under TP2.
- **Ignoring topology.** TP4 over PCIe with a switch has fundamentally different characteristics than TP2 with direct peer-to-peer links, or TP4 over NVLink.
- **Wrong diagnosis order.** Agents blame profiling data quality before checking whether the hardware model is correct.

### Correct diagnosis order when simulation diverges

1. Verify `col.toml` hardware parameters match the target system exactly.
2. Verify model and scheduler config match the target system's launch config.
3. Verify profiled compute data was generated under matching conditions.
4. Only then consider systemic gaps like unmodeled overhead.

### Concrete example

On 4× GPUs via PCIe Gen4 through a switch:
- TP2: ~25 GB/s effective bidirectional bandwidth (direct peer)
- TP4: effective per-pair bandwidth drops dramatically due to switch contention
- Using theoretical bandwidth for TP4 overestimates throughput by 30-50%

The `link_bw` parameter in `col.toml` must reflect **measured or realistic effective bandwidth**, not theoretical peak.

---

## 4. Reuse Before Rebuild

Follow this strict order when implementing a new feature:

1. **Reuse existing implementations.** Extend `ChunkedPrefillScheduler`, `SingleInstanceSystem`, etc.
2. **Compose within existing modules.** Implement traits and use `register_*!` macros.
3. **Create new modules only when necessary.** Follow `ARCHITECTURE.md` patterns.

The codebase has specific architectural invariants that are easy to violate when building from scratch. Code that bypasses them may compile but will break assumptions the rest of the system depends on.

---

## 5. Implement Mechanisms, Not Statistical Proxies

Always implement the actual mechanism before falling back to statistical approximation.

| Feature | Bad (statistical proxy) | Good (mechanism) |
|---------|------------------------|------------------|
| Prefix caching | Accept a hit_rate parameter | Implement trie tracking cached token sequences |
| Speculative decoding | Multiply throughput by a speedup factor | Implement draft-verify-accept loop |
| KV cache eviction | Assume infinite cache or fixed eviction rate | Track per-block allocation with LRU eviction |
| Chunked prefill | Apply fixed prefill latency | Budget max_num_batched_tokens across the batch |

Statistical proxy is a last resort. Discuss with the user first, and document clearly.

---

## 6. New Model Support

When the user wants to simulate a model not already in the codebase:

1. **Check if it exists** in `crates/col-core/src/models/`.
2. **Get the real config** from HuggingFace or official documentation. You need all architecture parameters.
3. **Implement it properly** using `Model` trait and `register_model!` macro. Do not alias another model.
4. **Verify profiling data exists** for this model+hardware combination. Do not use data from a different model.

If the model is already implemented but the task depends on shape-sensitive quantities (KV bytes, attention widths), re-read the real config and verify explicit fields like `head_dim`.

**Do not:**
- Alias models: `type Qwen3Model = Llama31Model` is wrong even if architectures look similar.
- Guess model parameters.
- Use profiling data from a different model.

---

## Red Flags

Stop and reconsider if you are about to:

| Instead of this | Do this |
|---|---|
| Use theoretical PCIe bandwidth in `col.toml` | Ask for measured or effective bandwidth |
| Start coding without GPU interconnect topology | Ask the user for missing hardware details |
| Assume the workload schema matches an old example | Parse defensively and confirm actual artifact semantics |
| Assume `head_dim = hidden_size / num_attention_heads` | Read the real model config and use explicit fields when present |
| Build a new module from scratch | Check whether an existing module can be extended |
| Accept a "rate" or "ratio" parameter instead of the mechanism | Implement the mechanism directly |
| Blame profiling data for sim-real mismatch first | Verify `col.toml` and launch config first |
| Export only internal residency state when the real metric is usage | Preserve internal state, but export the metric view that matches external semantics |
| Compare a full simulation against a timeout-truncated reference | Confirm completion policy and compare like-for-like |
| Proceed with incomplete system info | Stop and ask the user |

---

## Pre-Implementation Checklist

Before writing any simulator code, verify:

- [ ] Target system detail is complete — concrete values for all hardware and software parameters.
- [ ] Artifact and metric semantics are explicit — actual schema, field meanings, scoring method, reference completion policy.
- [ ] Actual files were inspected — real model config and at least one real sample/header for each artifact.
- [ ] Hardware topology is modeled — `col.toml` reflects actual interconnect, not theoretical peak.
- [ ] Internal state vs exported metric semantics are explicit — if they differ, both are documented.
- [ ] Existing code is surveyed — read relevant implementations before writing new code.
- [ ] Implementation follows the fallback loop — reuse → compose → create.
- [ ] Mechanism is implemented, not approximated.
- [ ] Underspecified costs are surfaced — config knobs or documented assumptions, not hidden constants.
- [ ] Model support is proper — real config, own implementation, matching profiling data.
- [ ] `ARCHITECTURE.md` invariants are respected.
