use std::any::Any;
use std::path::Path;

use anyhow::Result;

use crate::engine::Graph;
use crate::types::NodeID;

pub trait Module {
    fn resolve_path(&mut self, _base_dir: &Path) -> Result<()> {
        Ok(())
    }
}

pub trait LogicalHandler {
    fn handle(
        &mut self,
        payload: Box<dyn Any>,
        now: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<f64>;
}
