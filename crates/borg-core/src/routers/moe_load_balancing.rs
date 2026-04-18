use anyhow::{bail, Result};
use serde::Deserialize;

use crate::module::Module;
use crate::router::Router;
use crate::types::{InstanceLoad, Request};

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoeLoadBalancingRouter {
    #[serde(skip_deserializing, default)]
    next_tiebreak: usize,
}

crate::register_router!(MoeLoadBalancingRouter, "moe_load_balancing");

impl Module for MoeLoadBalancingRouter {}

impl Router for MoeLoadBalancingRouter {
    fn select_instance(&mut self, _request: &Request, instances: &[InstanceLoad]) -> Result<usize> {
        if instances.is_empty() {
            bail!("moe_load_balancing router requires at least one instance");
        }

        let len = instances.len();
        let start = self.next_tiebreak % len;
        let mut best_offset = 0usize;
        let mut best_score = score(instances[start]);

        for offset in 1..len {
            let index = (start + offset) % len;
            let candidate = score(instances[index]);
            if candidate < best_score {
                best_offset = offset;
                best_score = candidate;
            }
        }

        let selected = (start + best_offset) % len;
        self.next_tiebreak = (selected + 1) % len;
        Ok(selected)
    }
}

fn score(load: InstanceLoad) -> (u64, usize, u64, usize) {
    (
        load.total_tokens(),
        load.total_requests(),
        load.running_tokens,
        load.running_requests,
    )
}
