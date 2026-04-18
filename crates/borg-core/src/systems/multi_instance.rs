use std::any::Any;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use serde::Deserialize;

use crate::engine::Graph;
use crate::module::{LogicalHandler, Module};
use crate::router::Router;
use crate::system::System;
use crate::types::{InstanceLoad, NodeID, Request, RequestResult};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiInstanceSystem {
    instances: Vec<Box<dyn System>>,
    #[serde(default = "default_router")]
    router: Box<dyn Router>,
}

crate::register_system!(MultiInstanceSystem, "multi_instance");

fn default_router() -> Box<dyn Router> {
    Box::new(crate::routers::round_robin::RoundRobinRouter::default())
}

impl LogicalHandler for MultiInstanceSystem {
    fn handle(
        &mut self,
        _payload: Box<dyn Any>,
        _now: f64,
        _current: NodeID,
        _graph: &mut Graph,
    ) -> Result<f64> {
        Err(anyhow!(
            "multi_instance does not accept logical payloads directly"
        ))
    }
}

impl Module for MultiInstanceSystem {
    fn resolve_path(&mut self, base_dir: &Path) -> Result<()> {
        self.router.resolve_path(base_dir)?;
        for instance in &mut self.instances {
            instance.resolve_path(base_dir)?;
        }
        Ok(())
    }
}

impl System for MultiInstanceSystem {
    fn init(&mut self) -> Result<()> {
        if self.instances.is_empty() {
            bail!("multi_instance requires at least one child instance");
        }

        for instance in &mut self.instances {
            instance.init()?;
        }
        Ok(())
    }

    fn add_request_arrival(
        &mut self,
        request: Request,
        graph: &mut Graph,
        parent: Option<NodeID>,
        sink: Option<*mut dyn LogicalHandler>,
    ) -> Result<()> {
        let len = self.instances.len();
        if len == 0 {
            bail!("multi_instance requires at least one child instance");
        }

        let loads = vec![InstanceLoad::default(); len];
        let index = self.router.select_instance(&request, &loads)?;
        if index >= len {
            bail!("router selected invalid instance index {index} for {len} instances");
        }
        self.instances[index].add_request_arrival(request, graph, parent, sink)
    }

    fn into_results(self: Box<Self>) -> Result<Vec<RequestResult>> {
        let MultiInstanceSystem { instances, .. } = *self;
        let mut results = Vec::new();
        for instance in instances {
            results.extend(instance.into_results()?);
        }
        Ok(results)
    }
}
