use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};

pub type NodeID = u64;
pub type NetworkDeviceID = u64;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub enum Layer {
    Embedding {
        name: String,
        num_embeddings: u64,
        embedding_dim: u64,
    },
    LayerNorm {
        name: String,
        hidden_size: u64,
    },
    Gemm {
        name: String,
        m: u64,
        n: u64,
    },
    Rope {
        name: String,
        head_dim: u64,
        num_heads: u64,
    },
    Attention {
        name: String,
        num_q_heads: u64,
        num_kv_heads: u64,
        head_dim: u64,
    },
    Act {
        name: String,
        width: u64,
    },
    Sampler {
        name: String,
        vocab_size: u64,
    },
    AllReduce {
        name: String,
        bytes_per_token: u64,
    },
    Moe {
        name: String,
        num_experts: u64,
        num_experts_per_token: u32,
        num_shared_experts_per_token: u32,
        dispatch_bytes_per_token: u64,
        combine_bytes_per_token: u64,
    },
}

impl Layer {
    pub fn name(&self) -> &str {
        match self {
            Self::Embedding { name, .. }
            | Self::LayerNorm { name, .. }
            | Self::Gemm { name, .. }
            | Self::Rope { name, .. }
            | Self::Attention { name, .. }
            | Self::Act { name, .. }
            | Self::Sampler { name, .. }
            | Self::AllReduce { name, .. }
            | Self::Moe { name, .. } => name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub request_id: u64,
    pub arrival_time: f64,
    pub initial: Vec<usize>,
    pub sub_requests: Vec<SubRequest>,
}

impl Request {
    pub fn total_input_tokens(&self) -> u32 {
        self.sub_requests.iter().map(SubRequest::input_tokens).sum()
    }

    pub fn total_output_tokens(&self) -> u32 {
        self.sub_requests
            .iter()
            .map(SubRequest::output_tokens)
            .sum()
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.sub_requests.is_empty(),
            "requests must contain at least one sub-request"
        );
        for (subrequest_index, sub_request) in self.sub_requests.iter().enumerate() {
            if let SubRequest::Llm {
                input_tokens,
                output_tokens,
                known_tokens,
                kv_tokens,
                ..
            } = sub_request
            {
                match (known_tokens, kv_tokens) {
                    (Some(known_tokens), Some(kv_tokens)) => {
                        let total_tokens = input_tokens.saturating_add(*output_tokens);
                        ensure!(
                            *known_tokens >= *input_tokens,
                            "llm sub-request {subrequest_index} has known_tokens smaller than input_tokens"
                        );
                        ensure!(
                            *known_tokens <= total_tokens,
                            "llm sub-request {subrequest_index} has known_tokens larger than input_tokens + output_tokens"
                        );
                        ensure!(
                            *kv_tokens <= *known_tokens,
                            "llm sub-request {subrequest_index} has kv_tokens larger than known_tokens"
                        );
                    }
                    (None, None) => {}
                    _ => {
                        bail!(
                            "llm sub-request {subrequest_index} must set both known_tokens and kv_tokens together"
                        );
                    }
                }
            }
        }
        validate_initial_indices(&self.initial, &self.sub_requests)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SubRequestKind {
    Llm,
    ToolCall,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SubRequest {
    Llm {
        input_tokens: u32,
        output_tokens: u32,
        known_tokens: Option<u32>,
        kv_tokens: Option<u32>,
        next: Vec<usize>,
        interval: Option<f64>,
        model: Option<String>,
        input_token_ids: Option<Vec<u32>>,
        output_token_ids: Option<Vec<u32>>,
    },
    ToolCall {
        input_tokens: u32,
        output_tokens: u32,
        next: Vec<usize>,
        duration: Option<f64>,
        interval: Option<f64>,
    },
}

impl SubRequest {
    pub fn kind(&self) -> SubRequestKind {
        match self {
            Self::Llm { .. } => SubRequestKind::Llm,
            Self::ToolCall { .. } => SubRequestKind::ToolCall,
        }
    }

    pub fn input_tokens(&self) -> u32 {
        match self {
            Self::Llm { input_tokens, .. } | Self::ToolCall { input_tokens, .. } => *input_tokens,
        }
    }

    pub fn output_tokens(&self) -> u32 {
        match self {
            Self::Llm { output_tokens, .. } | Self::ToolCall { output_tokens, .. } => {
                *output_tokens
            }
        }
    }

    pub fn next(&self) -> &[usize] {
        match self {
            Self::Llm { next, .. } | Self::ToolCall { next, .. } => next,
        }
    }

    pub fn duration(&self) -> Option<f64> {
        match self {
            Self::Llm { .. } => None,
            Self::ToolCall { duration, .. } => *duration,
        }
    }

    pub fn interval(&self) -> Option<f64> {
        match self {
            Self::Llm { interval, .. } | Self::ToolCall { interval, .. } => *interval,
        }
    }

    pub fn model(&self) -> Option<&str> {
        match self {
            Self::Llm { model, .. } => model.as_deref(),
            Self::ToolCall { .. } => None,
        }
    }

    pub fn input_token_ids(&self) -> Option<&[u32]> {
        match self {
            Self::Llm {
                input_token_ids, ..
            } => input_token_ids.as_deref(),
            Self::ToolCall { .. } => None,
        }
    }

    pub fn output_token_ids(&self) -> Option<&[u32]> {
        match self {
            Self::Llm {
                output_token_ids, ..
            } => output_token_ids.as_deref(),
            Self::ToolCall { .. } => None,
        }
    }

    pub fn known_tokens(&self) -> Option<u32> {
        match self {
            Self::Llm { known_tokens, .. } => *known_tokens,
            Self::ToolCall { .. } => None,
        }
    }

    pub fn kv_tokens(&self) -> Option<u32> {
        match self {
            Self::Llm { kv_tokens, .. } => *kv_tokens,
            Self::ToolCall { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReadySubRequest {
    pub request_id: u64,
    pub subrequest_index: usize,
    pub completion_node_id: NodeID,
    pub arrival_time: f64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub known_tokens: u32,
    pub kv_tokens: u32,
    pub input_token_ids: Option<Vec<u32>>,
    pub output_token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SubRequestCompletion {
    pub request_id: u64,
    pub subrequest_index: usize,
    pub completion_node_id: NodeID,
    pub token_latency: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestState {
    pub q_len: u32,
    pub kv_len: u32,
    pub lm_head_len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelStrategy {
    pub tp: u32,
    pub pp: u32,
    pub ep: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InstanceLoad {
    pub waiting_requests: usize,
    pub running_requests: usize,
    pub waiting_tokens: u64,
    pub running_tokens: u64,
}

impl InstanceLoad {
    pub fn total_requests(self) -> usize {
        self.waiting_requests + self.running_requests
    }

    pub fn total_tokens(self) -> u64 {
        self.waiting_tokens + self.running_tokens
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ComputeContext {
    pub moe_activated_experts: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResultSubRequest {
    Llm {
        input_tokens: u32,
        output_tokens: u32,
        next: Vec<usize>,
        token_latency: Vec<f64>,
        interval: Option<f64>,
        model: Option<String>,
    },
    ToolCall {
        input_tokens: u32,
        output_tokens: u32,
        next: Vec<usize>,
        duration: f64,
        interval: Option<f64>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RequestResult {
    pub request_id: u64,
    pub arrival_time: f64,
    pub initial: Vec<usize>,
    pub sub_requests: Vec<ResultSubRequest>,
}

fn validate_initial_indices(initial: &[usize], sub_requests: &[SubRequest]) -> Result<()> {
    let mut seen_initial = vec![false; sub_requests.len()];
    for &index in initial {
        if index >= sub_requests.len() {
            bail!(
                "initial sub-request index {index} is out of range for {} sub-requests",
                sub_requests.len()
            );
        }
        ensure!(
            !seen_initial[index],
            "initial sub-request index {index} is duplicated"
        );
        seen_initial[index] = true;
    }

    let mut indegree = vec![0usize; sub_requests.len()];
    for (subrequest_index, sub_request) in sub_requests.iter().enumerate() {
        for &next_index in sub_request.next() {
            if next_index >= sub_requests.len() {
                bail!("sub-request {subrequest_index} references invalid successor {next_index}");
            }
            indegree[next_index] += 1;
        }
    }

    let derived_initial = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, &degree)| (degree == 0).then_some(index))
        .collect::<Vec<_>>();
    ensure!(
        initial == derived_initial,
        "initial sub-request indices {initial:?} do not match DAG roots {derived_initial:?}"
    );

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PollResult {
    Pending { latency: f64 },
    Complete { latency: f64 },
}
