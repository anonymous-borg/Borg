# Borg Framework

## Prerequisites

- dependencies
  - `uv`
  - `cargo`

## Project Structure

- `crates/borg-core/`: foundation simulator code
- `crates/borg-cli/`: foundation simulator CLI interface
- `.agents/`: harness code for synthesizer agent
- `scenarios/`: example system configurations
- `workloads/`: example traces

## Examples

- profile (requires a single GPU)
```bash
uv run python scripts/vllm-profile.py \
  --receipt profile_receipts/llama3_1_8b.yaml \
  --output-csv data/llama3_1_8b.csv
```

- run the simulator
```bash
cargo run --release -- \
  --config scenarios/multi-gpu/borg.toml \
  --workload workloads/sharegpt.jsonl \
  --output-jsonl output/borg-sharegpt.jsonl
```

