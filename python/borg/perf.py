import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps

import torch
from vllm import SamplingParams
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.profiler.layerwise_profile import layerwise_profile
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput


@dataclass
class RawRow:
    name: str
    latency: float


@dataclass
class RequestSpec:
    q_len: int
    kv_len: int


@dataclass
class ForcedRoutingTensors:
    layer_name: str
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


def parse_entries(res: list[dict], layers: dict) -> list[RawRow]:
    parsed = []

    for entry in res:
        parsed.extend(parse_entry(entry, layers))

    return parsed


def create_batch(
    requests: list[RequestSpec], block_size: int
) -> tuple[SchedulerOutput, set[str]]:
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=1,
    )
    num_kv_cache_groups = 1

    scheduled = []
    num_scheduled_tokens: dict[str, int] = {}
    total_num_scheduled_tokens = 0
    block_start = 0
    req_ids = []
    for idx, request in enumerate(requests):
        req_id = str(idx)
        total_len = request.q_len + request.kv_len
        num_blocks = math.ceil(total_len / block_size)
        block_ids = list(range(block_start, block_start + num_blocks))
        block_start += num_blocks

        scheduled.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=[1] * total_len,
                mm_features=[],
                sampling_params=sampling_params,
                pooling_params=None,
                block_ids=tuple(block_ids for _ in range(num_kv_cache_groups)),
                num_computed_tokens=request.kv_len,
                lora_request=None,
            )
        )
        num_scheduled_tokens[req_id] = request.q_len
        total_num_scheduled_tokens += request.q_len
        req_ids.append(req_id)

    return (
        SchedulerOutput(
            scheduled_new_reqs=scheduled,
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * num_kv_cache_groups,
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        ),
        set(req_ids),
    )


def parse_entry(res: dict, layers: dict) -> list[RawRow]:
    parsed = []

    name = str(res["entry"]["name"]).split("(", 1)[0]  # )
    if name in layers.keys():
        if isinstance(layers[name], str):
            total_latency = float(res["entry"]["cuda_time_us"]) / 1000.0 / 1000.0
            invocations = int(res["entry"]["invocations"])
            latency = total_latency / invocations
            parsed.append(RawRow(layers[name], latency))
        elif isinstance(layers[name], dict):
            parsed.extend(parse_entries(res["children"], layers[name]))

    return parsed


def prepare_forced_routing_tensors(
    layer: FusedMoE,
    topk_ids_rows: list[list[int]],
    num_tokens: int,
) -> ForcedRoutingTensors:
    indices_type = layer.router._get_indices_type()
    device = next(layer.parameters()).device
    topk_ids = torch.tensor(
        topk_ids_rows,
        device=device,
        dtype=torch.int32 if indices_type is None else indices_type,
    )
    if topk_ids.shape != (num_tokens, layer.top_k):
        raise ValueError(
            f"Forced routing shape mismatch for {layer.layer_name}: "
            f"expected {(num_tokens, layer.top_k)}, got {tuple(topk_ids.shape)}"
        )

    topk_weights = torch.full(
        (num_tokens, layer.top_k),
        1.0 / layer.top_k,
        device=device,
        dtype=torch.float32,
    )
    return ForcedRoutingTensors(layer.layer_name, topk_weights, topk_ids)


@contextmanager
def patch_moe_routing(forced: ForcedRoutingTensors | None = None):
    original_forward_impl = FusedMoE.forward_impl

    @wraps(original_forward_impl)
    def hooked_forward_impl(self, hidden_states, router_logits):
        if forced is None or self.layer_name != forced.layer_name:
            return original_forward_impl(self, hidden_states, router_logits)
        if forced.topk_ids.shape != (hidden_states.shape[0], self.top_k):
            raise ValueError(
                f"Forced routing shape mismatch for {self.layer_name}: "
                f"expected {(hidden_states.shape[0], self.top_k)}, got {tuple(forced.topk_ids.shape)}"
            )

        original_select_experts = self.router.select_experts
        original_compute_routing = self.router._compute_routing

        @wraps(original_select_experts)
        def hooked_select_experts(*args, **kwargs):
            @wraps(original_compute_routing)
            def forced_compute_routing(
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                indices_type: torch.dtype | None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return forced.topk_weights, forced.topk_ids

            self.router._compute_routing = forced_compute_routing

            try:
                topk_weights, topk_ids = original_select_experts(*args, **kwargs)
            finally:
                self.router._compute_routing = original_compute_routing

            return topk_weights, topk_ids

        self.router.select_experts = hooked_select_experts
        try:
            return original_forward_impl(self, hidden_states, router_logits)
        finally:
            self.router.select_experts = original_select_experts

    FusedMoE.forward_impl = hooked_forward_impl
    try:
        yield
    finally:
        FusedMoE.forward_impl = original_forward_impl


def only_rows(raw_rows: list[RawRow], names: set[str]) -> list[RawRow]:
    return [row for row in raw_rows if row.name in names]


def single_moe_layer(model_runner) -> FusedMoE:
    model = model_runner.get_model()
    moe_layers = [module for module in model.modules() if isinstance(module, FusedMoE)]
    if len(moe_layers) != 1:
        raise ValueError(f"Expected exactly one FusedMoE layer, got {len(moe_layers)}")
    return moe_layers[0]


def build_moe_topk_ids(
    num_tokens: int,
    top_k: int,
    activated_experts: int,
) -> list[list[int]]:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if activated_experts < top_k:
        raise ValueError("activated_experts must be at least top_k")
    if activated_experts > num_tokens * top_k:
        raise ValueError("activated_experts must not exceed num_tokens * top_k")

    return [
        [((token_idx * top_k) + offset) % activated_experts for offset in range(top_k)]
        for token_idx in range(num_tokens)
    ]


class Perf:
    def profile_specs(
        self,
        requests: list[RequestSpec],
        layers: dict,
        forced: ForcedRoutingTensors | None = None,
    ) -> dict[str, list[dict]]:
        batch, _ = create_batch(requests, self.cache_config.block_size)  # ty: ignore

        output = self.model_runner.execute_model(batch)  # ty: ignore
        if output is None:
            self.model_runner.sample_tokens(None)  # ty: ignore

        with patch_moe_routing(forced):
            with layerwise_profile() as profiler:
                self.model_runner.execute_model(batch)  # ty: ignore
                if output is None:
                    self.model_runner.sample_tokens(None)  # ty: ignore

        res = profiler.results.convert_stats_to_dict()
        raw_rows = parse_entries(res["summary_stats"], layers)
        return {
            "raw_rows": [asdict(row) for row in raw_rows],
        }

    def run_specs(self, requests: list[dict], layers: dict):
        requests = [RequestSpec(**req) for req in requests]  # ty: ignore

        return self.profile_specs(requests, layers)["raw_rows"]

    def run_specs_moe(
        self,
        requests: list[dict],
        layers: dict,
        activated_experts: int,
    ):
        requests = [RequestSpec(**req) for req in requests]  # ty: ignore
        moe_layer = single_moe_layer(self.model_runner)  # ty: ignore
        num_tokens = sum(request.q_len for request in requests)
        topk_ids = build_moe_topk_ids(num_tokens, moe_layer.top_k, activated_experts)
        forced = prepare_forced_routing_tensors(moe_layer, topk_ids, num_tokens)
        res = self.profile_specs(requests, layers, forced)
        moe_rows = only_rows(
            [RawRow(**row) for row in res["raw_rows"]],
            {"moe"},
        )
        if len(moe_rows) != 1:
            raise ValueError(f"Expected exactly one moe row, got {len(moe_rows)}")
        return [asdict(row) for row in moe_rows]
