use anyhow::{bail, Result};
use serde::Deserialize;

use crate::module::Module;
use crate::router::Router;
use crate::types::{InstanceLoad, Request};

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundRobinRouter {
    #[serde(skip_deserializing, default)]
    next_instance: usize,
}

crate::register_router!(RoundRobinRouter, "round_robin");

impl Module for RoundRobinRouter {}

impl Router for RoundRobinRouter {
    fn select_instance(&mut self, _request: &Request, instances: &[InstanceLoad]) -> Result<usize> {
        if instances.is_empty() {
            bail!("round_robin router requires at least one instance");
        }

        let index = self.next_instance % instances.len();
        self.next_instance = (index + 1) % instances.len();
        Ok(index)
    }
}
