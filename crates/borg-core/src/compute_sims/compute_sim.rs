use anyhow::Result;

use crate::module::Module;
use crate::types::{ComputeContext, Layer, ParallelStrategy, RequestState};

pub trait ComputeSimulator: Module {
    fn init(&mut self, _parallel: ParallelStrategy) -> Result<()> {
        Ok(())
    }

    fn simulate(
        &self,
        layer: &Layer,
        batch: &[RequestState],
        context: ComputeContext,
    ) -> Result<f64>;
}

crate::declare_registry! {
    registration = ComputeSimulatorRegistration,
    trait = crate::compute_sim::ComputeSimulator,
    deserialize_fn = deserialize_compute_simulator_impl,
    registry_name = "compute_sim",
}

crate::impl_registry_deserialize! {
    trait = crate::compute_sim::ComputeSimulator,
    deserialize_fn = crate::compute_sim::deserialize_compute_simulator_impl,
}

#[macro_export]
macro_rules! register_compute_simulator {
    ($ty:ty, $kind:expr $(,)?) => {
        $crate::submit_registration!(
            registration = $crate::compute_sim::ComputeSimulatorRegistration,
            trait = $crate::compute_sim::ComputeSimulator,
            type = $ty,
            kind = $kind,
            registry_name = "compute_sim",
        );
    };
}
