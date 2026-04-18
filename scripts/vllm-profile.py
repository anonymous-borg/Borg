import csv
import gc
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def install_typeshed_shim() -> None:
    if "_typeshed" in sys.modules:
        return
    module = types.ModuleType("_typeshed")
    module.DataclassInstance = object  # ty: ignore
    sys.modules["_typeshed"] = module


# NOTE: vLLM bug
install_typeshed_shim()


import torch
import typer
import yaml
from borg.perf import RawRow, RequestSpec
from deepmerge import always_merger
from pydantic import BaseModel
from tqdm import tqdm
from vllm import LLM
from vllm.transformers_utils.config import get_config


class LayerModes(BaseModel):
    general: list[str]
    tp_agnostic: list[str]
    lm_head: list[str]
    sampler: list[str]
    attention: list[str]
    moe: list[str] = []


class Receipts(BaseModel):
    vllm_config: dict[str, Any]
    profile_tp: list[int]
    tp_modifications: list[str]
    layers: dict[str, Any]
    layer_modes: LayerModes


@dataclass
class Row:
    name: str
    tp: int
    key_0: int
    key_1: int
    latency: float


def build_power_of_two_grid(max_value: int) -> list[int]:
    values = []
    current = 1
    while current <= max_value:
        values.append(current)
        current *= 2
    return values


def build_token_grid_specs(max_num_batched_tokens: int) -> list[list[RequestSpec]]:
    num_token_grids = (
        list(range(1, 16))
        + list(range(16, 64, 4))
        + list(range(64, max_num_batched_tokens + 1, 16))
    )
    return [[RequestSpec(i, 0)] for i in num_token_grids]


def build_lm_head_specs(max_num_seqs: int) -> list[list[RequestSpec]]:
    num_token_grids = (
        list(range(1, 16))
        + list(range(16, 64, 4))
        + list(range(64, max_num_seqs + 1, 16))
    )
    return [[RequestSpec(1, 0)] * i for i in num_token_grids]


def build_attention_specs(
    max_num_batched_tokens: int, num_cache_tokens: int, max_model_len: int
) -> list[list[RequestSpec]]:
    specs = []
    total_kv_len = 1
    while total_kv_len <= min(num_cache_tokens, max_model_len * 32):
        kv_len = min(total_kv_len, max_model_len // 2)
        num_reqs = total_kv_len // kv_len

        for i in range(1, 9):
            specs.append([RequestSpec(i, kv_len)] * num_reqs)

        q_len = 16
        while q_len <= max_num_batched_tokens:
            spec = [RequestSpec(1, kv_len) for _ in range(num_reqs)]
            spec[-1].q_len = min(q_len, max_num_batched_tokens - num_reqs + 1)
            specs.append(spec)
            q_len *= 2

        total_kv_len *= 2

    return specs


def build_moe_specs(
    max_num_batched_tokens: int,
    num_experts: int,
    top_k: int,
) -> list[tuple[list[RequestSpec], int]]:
    specs = []
    for num_tokens in build_power_of_two_grid(max_num_batched_tokens):
        for activated_experts in build_power_of_two_grid(num_experts):
            if activated_experts < top_k:
                continue
            if activated_experts > min(num_experts, num_tokens * top_k):
                continue
            specs.append(([RequestSpec(num_tokens, 0)], activated_experts))
    return specs


def aggregate_rows(rows: list[Row]) -> list[Row]:
    aggregates: dict[tuple[str, int, int, int], tuple[float, int]] = {}
    for row in rows:
        key = (row.name, row.tp, row.key_0, row.key_1)
        if key in aggregates:
            total_latency, count = aggregates[key]
            aggregates[key] = (total_latency + row.latency, count + 1)
        else:
            aggregates[key] = (row.latency, 1)

    return [
        Row(name, tp, key_0, key_1, total_latency / count)
        for (name, tp, key_0, key_1), (total_latency, count) in aggregates.items()
    ]


def main(
    receipt: Path = typer.Option(),
    output_csv: Path = typer.Option(),
):
    receipt_file = receipt.resolve()
    with receipt_file.open() as f:
        cfg = yaml.safe_load(f)
    receipts = Receipts(**cfg)

    layer_modes: dict[str, str] = {}
    for mode, mode_layers in (
        ("general", receipts.layer_modes.general),
        ("tp_agnostic", receipts.layer_modes.tp_agnostic),
        ("lm_head", receipts.layer_modes.lm_head),
        ("sampler", receipts.layer_modes.sampler),
        ("attention", receipts.layer_modes.attention),
        ("moe", receipts.layer_modes.moe),
    ):
        for layer in mode_layers:
            layer_modes[layer] = mode

    if 1 not in receipts.profile_tp:
        raise RuntimeError("profile_tp should include 1")

    res = []
    for tp in receipts.profile_tp:
        model_config = get_config(receipts.vllm_config["model"], True).to_dict()
        hf_overrides = {m: model_config[m] // tp for m in receipts.tp_modifications}

        llm = LLM(
            **always_merger.merge(receipts.vllm_config, {"hf_overrides": hf_overrides}),  # ty: ignore
            worker_extension_cls="borg.perf.Perf",
        )

        max_num_seqs = llm.llm_engine.vllm_config.scheduler_config.max_num_seqs
        max_num_batched_tokens = (
            llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens
        )
        num_cache_blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
        num_cache_tokens = (
            num_cache_blocks * llm.llm_engine.vllm_config.cache_config.block_size
        )  # ty: ignore
        max_model_len = llm.llm_engine.vllm_config.model_config.max_model_len

        assert isinstance(num_cache_tokens, int)
        assert isinstance(max_model_len, int)

        specs = []
        specs.extend(build_token_grid_specs(max_num_batched_tokens))
        specs.extend(build_lm_head_specs(max_num_seqs))
        specs.extend(
            build_attention_specs(
                max_num_batched_tokens, num_cache_tokens, max_model_len
            )
        )
        moe_specs = []
        if receipts.layer_modes.moe and tp == 1:
            moe_specs = build_moe_specs(
                max_num_batched_tokens=max_num_batched_tokens,
                num_experts=model_config["num_experts"],
                top_k=model_config["num_experts_per_tok"],
            )

        for spec in tqdm(specs):
            raw_rows = llm.collective_rpc("run_specs", args=(spec, receipts.layers))[0]
            raw_rows = [RawRow(**raw_row) for raw_row in raw_rows]

            for raw_row in raw_rows:
                mode = layer_modes.get(raw_row.name)
                if mode is None:
                    continue
                match mode:
                    case "general":
                        key_0 = 0
                        key_1 = sum(req.q_len for req in spec)

                        res.append(Row(raw_row.name, tp, key_0, key_1, raw_row.latency))
                    case "tp_agnostic":
                        key_0 = 0
                        key_1 = sum(req.q_len for req in spec)

                        if tp == 1:
                            for output_tp in receipts.profile_tp:
                                res.append(
                                    Row(
                                        raw_row.name,
                                        output_tp,
                                        key_0,
                                        key_1,
                                        raw_row.latency,
                                    )
                                )
                    case "lm_head":
                        key_0 = 0
                        key_1 = len(spec)

                        res.append(Row(raw_row.name, tp, key_0, key_1, raw_row.latency))
                    case "sampler":
                        key_0 = 0
                        key_1 = len(spec)

                        if tp == 1:
                            for output_tp in receipts.profile_tp:
                                res.append(
                                    Row(
                                        raw_row.name,
                                        output_tp,
                                        key_0,
                                        key_1,
                                        raw_row.latency,
                                    )
                                )
                    case "attention":
                        key_0 = sum(req.kv_len for req in spec)
                        key_1 = sum(
                            req.q_len * (req.q_len // 2 + req.kv_len) for req in spec
                        )

                        res.append(Row(raw_row.name, tp, key_0, key_1, raw_row.latency))
                    case "moe":
                        continue

        for spec, activated_experts in tqdm(moe_specs):
            moe_rows = llm.collective_rpc(
                "run_specs_moe",
                args=(spec, receipts.layers, activated_experts),
            )[0]
            moe_rows = [RawRow(**raw_row) for raw_row in moe_rows]
            total_tokens = sum(req.q_len for req in spec)
            for raw_row in moe_rows:
                res.append(
                    Row(
                        raw_row.name,
                        tp,
                        total_tokens,
                        activated_experts,
                        raw_row.latency,
                    )
                )

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    output_csv = output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["name", "tp", "key_0", "key_1", "latency"],
        )
        writer.writeheader()
        for row in aggregate_rows(res):
            writer.writerow(asdict(row))


if __name__ == "__main__":
    typer.run(main)
