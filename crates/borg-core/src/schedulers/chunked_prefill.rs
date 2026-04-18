use std::collections::LinkedList;

use anyhow::{anyhow, bail, Result};
use serde::Deserialize;

use crate::module::Module;
use crate::scheduler::Scheduler;
use crate::schedulers::prefix_cache::PrefixCache;
use crate::types::{
    InstanceLoad, NodeID, ParallelStrategy, ReadySubRequest, RequestState, SubRequestCompletion,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkedPrefillScheduler {
    block_size: u32,
    max_batch: usize,
    max_num_batched_tokens: u32,
    device_mem: u64,
    device_mem_utilization: f64,
    enable_prefix_caching: bool,

    #[serde(skip_deserializing, default)]
    queue: LinkedList<RunningRequest>,

    #[serde(skip_deserializing, default)]
    running: bool,

    #[serde(skip_deserializing, default)]
    current_batch: Vec<RequestState>,

    #[serde(skip_deserializing, default)]
    model_weight_bytes_per_device: u64,

    #[serde(skip_deserializing, default)]
    kv_bytes_per_token_per_device: u64,

    #[serde(skip_deserializing, default)]
    usable_device_mem: u64,

    #[serde(skip_deserializing, default)]
    prefix_cache: PrefixCache,

    #[serde(skip_deserializing, default)]
    effective_max_batch: usize,

    #[serde(skip_deserializing, default)]
    effective_max_num_batched_tokens: u32,
}

crate::register_scheduler!(ChunkedPrefillScheduler, "chunked_prefill");

impl Module for ChunkedPrefillScheduler {}

#[derive(Debug, Clone)]
struct RunningRequest {
    request_id: u64,
    subrequest_index: usize,
    completion_node_id: NodeID,
    arrival_time: f64,
    input_tokens: u32,
    output_tokens: u32,

    known_tokens: u32,
    shared_kv_tokens: u32,
    private_kv_tokens: u32,
    last_time: f64,
    token_latency: Vec<f64>,

    input_token_ids: Option<Vec<u32>>,
    output_token_ids: Option<Vec<u32>>,
}

fn block_count(num_tokens: u32, block_size: u32) -> u32 {
    num_tokens.div_ceil(block_size)
}

impl RunningRequest {
    fn new(request: ReadySubRequest) -> Self {
        Self {
            request_id: request.request_id,
            subrequest_index: request.subrequest_index,
            completion_node_id: request.completion_node_id,
            arrival_time: request.arrival_time,
            input_tokens: request.input_tokens,
            output_tokens: request.output_tokens,
            known_tokens: request.known_tokens,
            shared_kv_tokens: 0,
            private_kv_tokens: request.kv_tokens,
            last_time: request.arrival_time,
            token_latency: Vec::new(),
            input_token_ids: request.input_token_ids,
            output_token_ids: request.output_token_ids,
        }
    }

    fn effective_kv_tokens(&self) -> u32 {
        self.shared_kv_tokens + self.private_kv_tokens
    }

    fn private_kv_bytes(&self, block_size: u32, kv_bytes_per_token_per_device: u64) -> u64 {
        u64::from(block_count(self.private_kv_tokens, block_size))
            * u64::from(block_size)
            * kv_bytes_per_token_per_device
    }

    fn private_kv_bytes_after(
        &self,
        q_len: u32,
        block_size: u32,
        kv_bytes_per_token_per_device: u64,
    ) -> u64 {
        u64::from(block_count(self.private_kv_tokens + q_len, block_size))
            * u64::from(block_size)
            * kv_bytes_per_token_per_device
    }

    fn available_token_prefix(&self, max_tokens: u32) -> Option<Vec<u32>> {
        let input_token_ids = self.input_token_ids.as_ref()?;
        let max_tokens = max_tokens as usize;
        let mut prefix = Vec::with_capacity(max_tokens.min(input_token_ids.len()));

        let prompt_len = input_token_ids.len().min(max_tokens);
        prefix.extend_from_slice(&input_token_ids[..prompt_len]);
        if prompt_len == max_tokens {
            return Some(prefix);
        }

        let output_token_ids = self.output_token_ids.as_ref()?;
        let remaining = max_tokens - prompt_len;
        let output_len = output_token_ids.len().min(remaining);
        prefix.extend_from_slice(&output_token_ids[..output_len]);
        Some(prefix)
    }

    fn exact_token_prefix(&self, tokens: u32) -> Option<Vec<u32>> {
        let prefix = self.available_token_prefix(tokens)?;
        (prefix.len() == tokens as usize).then_some(prefix)
    }

    fn longest_cache_hit(&self, prefix_cache: &mut PrefixCache) -> u32 {
        if self.effective_kv_tokens() != 0 {
            return self.shared_kv_tokens;
        }

        let Some(tokens) = self.available_token_prefix(self.known_tokens) else {
            return 0;
        };
        prefix_cache.longest_hit(&tokens)
    }

    fn publishable_shared_tokens(&self, block_size: u32) -> u32 {
        let available = self
            .available_token_prefix(self.effective_kv_tokens())
            .map(|tokens| tokens.len() as u32)
            .unwrap_or(0);
        available
            .min(self.effective_kv_tokens())
            .div_euclid(block_size)
            * block_size
    }

    fn publish_shared_prefixes(
        &self,
        prefix_cache: &mut PrefixCache,
        block_size: u32,
        publishable_shared_tokens: u32,
    ) {
        let mut boundary = self.shared_kv_tokens + block_size;
        while boundary <= publishable_shared_tokens {
            if let Some(prefix) = self.exact_token_prefix(boundary) {
                prefix_cache.insert_prefix(&prefix);
            }
            boundary += block_size;
        }
    }

    fn relock_shared_prefix(&mut self, prefix_cache: &mut PrefixCache, new_shared_kv_tokens: u32) {
        if self.shared_kv_tokens == new_shared_kv_tokens {
            return;
        }

        if self.shared_kv_tokens > 0 {
            if let Some(prefix) = self.exact_token_prefix(self.shared_kv_tokens) {
                prefix_cache.unlock_prefix(&prefix);
            }
            self.shared_kv_tokens = 0;
        }

        if new_shared_kv_tokens > 0 {
            if let Some(prefix) = self.exact_token_prefix(new_shared_kv_tokens) {
                prefix_cache.lock_prefix(&prefix);
                self.shared_kv_tokens = new_shared_kv_tokens;
            }
        }
    }

    fn preempt(&mut self, prefix_cache: &mut PrefixCache) {
        self.relock_shared_prefix(prefix_cache, 0);
        self.private_kv_tokens = 0;
    }
}

impl ChunkedPrefillScheduler {
    fn cache_block_bytes(&self) -> u64 {
        u64::from(self.block_size) * self.kv_bytes_per_token_per_device
    }
}

impl Scheduler for ChunkedPrefillScheduler {
    fn init(
        &mut self,
        model_weight_bytes_per_device: u64,
        kv_bytes_per_token_per_device: u64,
        parallel: ParallelStrategy,
    ) -> Result<()> {
        if !self.device_mem_utilization.is_finite()
            || self.device_mem_utilization <= 0.0
            || self.device_mem_utilization > 1.0
        {
            bail!(
                "device_mem_utilization must be in (0, 1], got {}",
                self.device_mem_utilization
            );
        }

        self.model_weight_bytes_per_device = model_weight_bytes_per_device;
        self.kv_bytes_per_token_per_device = kv_bytes_per_token_per_device;
        self.usable_device_mem =
            (self.device_mem as f64 * self.device_mem_utilization).floor() as u64;
        self.effective_max_batch = self
            .max_batch
            .checked_mul(parallel.ep as usize)
            .ok_or_else(|| anyhow!("effective max_batch overflowed"))?;
        self.effective_max_num_batched_tokens = self
            .max_num_batched_tokens
            .checked_mul(parallel.ep)
            .ok_or_else(|| anyhow!("effective max_num_batched_tokens overflowed"))?;
        if self.usable_device_mem == 0 {
            bail!(
                "device_mem_utilization leaves no usable device memory from device_mem={}",
                self.device_mem
            );
        }
        Ok(())
    }

    fn schedule(&mut self) -> Result<Vec<RequestState>> {
        if self.running || self.queue.is_empty() {
            return Ok(Vec::new());
        }

        let mut batch_requests = Vec::new();
        let mut total_q_len = 0_u32;
        let block_size = self.block_size;
        let kv_bytes_per_token_per_device = self.kv_bytes_per_token_per_device;
        let usable_device_mem = self.usable_device_mem;
        let enable_prefix_caching = self.enable_prefix_caching;
        let cache_block_bytes = self.cache_block_bytes();
        let prefix_cache = &mut self.prefix_cache;
        let mut gpu_bytes = self.model_weight_bytes_per_device
            + prefix_cache.bytes_used(block_size, kv_bytes_per_token_per_device);

        for request in self.queue.iter_mut() {
            if batch_requests.len() >= self.effective_max_batch {
                break;
            }

            let remaining_batch_tokens = self
                .effective_max_num_batched_tokens
                .saturating_sub(total_q_len);
            if remaining_batch_tokens == 0 {
                break;
            }

            let original_shared = request.shared_kv_tokens;
            let matched_shared = if enable_prefix_caching {
                request.longest_cache_hit(prefix_cache)
            } else {
                0
            };
            if matched_shared > original_shared {
                request.relock_shared_prefix(prefix_cache, matched_shared);
            }

            let effective_kv_tokens = request.effective_kv_tokens();
            let num_tokens = request
                .known_tokens
                .saturating_sub(effective_kv_tokens)
                .clamp(1, remaining_batch_tokens);
            let needed_private_bytes = request.private_kv_bytes_after(
                num_tokens,
                block_size,
                kv_bytes_per_token_per_device,
            );

            if enable_prefix_caching {
                while gpu_bytes.saturating_add(needed_private_bytes) > usable_device_mem {
                    if cache_block_bytes == 0 || !prefix_cache.evict_one_leaf() {
                        request.relock_shared_prefix(prefix_cache, original_shared);
                        break;
                    }
                    gpu_bytes = gpu_bytes.saturating_sub(cache_block_bytes);
                }
                if gpu_bytes.saturating_add(needed_private_bytes) > usable_device_mem {
                    break;
                }
            } else if gpu_bytes.saturating_add(needed_private_bytes) > usable_device_mem {
                request.relock_shared_prefix(prefix_cache, original_shared);
                break;
            }

            total_q_len += num_tokens;
            gpu_bytes += needed_private_bytes;

            batch_requests.push(RequestState {
                q_len: num_tokens,
                kv_len: effective_kv_tokens,
                lm_head_len: 1,
            });
        }

        if batch_requests.is_empty() {
            bail!("system cannot handle a request!");
        }

        for request in self.queue.iter_mut().skip(batch_requests.len()) {
            if request.effective_kv_tokens() == 0 {
                break;
            }

            let private_bytes = request.private_kv_bytes(block_size, kv_bytes_per_token_per_device);
            if gpu_bytes.saturating_add(private_bytes) > usable_device_mem {
                request.preempt(prefix_cache);
            } else {
                gpu_bytes += private_bytes;
            }
        }

        self.running = true;
        self.current_batch = batch_requests.clone();
        Ok(batch_requests)
    }

    fn enqueue_sub_request(&mut self, request: ReadySubRequest) {
        self.queue.push_back(RunningRequest::new(request));
    }

    fn done_iteration(&mut self, now: f64) -> Vec<SubRequestCompletion> {
        let current_batch = std::mem::take(&mut self.current_batch);

        let mut tail = self.queue.split_off(current_batch.len());
        for (request, scheduled) in self.queue.iter_mut().zip(&current_batch) {
            if request.known_tokens == request.effective_kv_tokens() {
                let latency = if request.token_latency.is_empty() {
                    now - request.arrival_time
                } else {
                    now - request.last_time
                };
                request.token_latency.push(latency);
                request.last_time = now;
                request.known_tokens += 1;
            }
            request.private_kv_tokens += scheduled.q_len;

            if self.enable_prefix_caching {
                let total_effective_kv_tokens = request.effective_kv_tokens();
                let publishable_shared_tokens = request.publishable_shared_tokens(self.block_size);
                if publishable_shared_tokens > request.shared_kv_tokens {
                    request.publish_shared_prefixes(
                        &mut self.prefix_cache,
                        self.block_size,
                        publishable_shared_tokens,
                    );
                    request.relock_shared_prefix(&mut self.prefix_cache, publishable_shared_tokens);
                }
                request.private_kv_tokens =
                    total_effective_kv_tokens.saturating_sub(request.shared_kv_tokens);
            }
        }

        let mut completed = Vec::new();
        let mut next_queue = LinkedList::new();
        while let Some(mut request) = self.queue.pop_front() {
            let total_tokens = request.input_tokens + request.output_tokens;
            if request.known_tokens >= total_tokens {
                if self.enable_prefix_caching {
                    request.relock_shared_prefix(&mut self.prefix_cache, 0);
                }
                completed.push(SubRequestCompletion {
                    request_id: request.request_id,
                    subrequest_index: request.subrequest_index,
                    completion_node_id: request.completion_node_id,
                    token_latency: request.token_latency,
                });
            } else {
                next_queue.push_back(request);
            }
        }

        next_queue.append(&mut tail);
        self.queue = next_queue;
        self.running = false;
        completed
    }

    fn instance_load(&self) -> InstanceLoad {
        let running_requests = self.current_batch.len();
        let running_tokens = self
            .current_batch
            .iter()
            .map(|request| u64::from(request.q_len))
            .sum();
        let mut waiting_requests = 0usize;
        let mut waiting_tokens = 0u64;
        for request in self.queue.iter().skip(running_requests) {
            waiting_requests += 1;
            waiting_tokens += u64::from(
                request
                    .known_tokens
                    .saturating_sub(request.effective_kv_tokens())
                    .max(1),
            );
        }
        InstanceLoad {
            waiting_requests,
            running_requests,
            waiting_tokens,
            running_tokens,
        }
    }
}
