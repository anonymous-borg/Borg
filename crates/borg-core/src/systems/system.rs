use anyhow::Result;

use crate::engine::Graph;
use crate::module::{LogicalHandler, Module};
use crate::types::{NodeID, Request, RequestResult};

pub enum SystemEvent {
    RequestCompleted {
        request: Request,
        result: RequestResult,
    },
}

pub trait System: Module + LogicalHandler {
    fn init(&mut self) -> Result<()>;

    fn add_request_arrival(
        &mut self,
        request: Request,
        graph: &mut Graph,
        parent: Option<NodeID>,
        sink: Option<*mut dyn LogicalHandler>,
    ) -> Result<()>;

    fn into_results(self: Box<Self>) -> Result<Vec<RequestResult>>;
}

crate::declare_registry! {
    registration = SystemRegistration,
    trait = crate::system::System,
    deserialize_fn = deserialize_system_impl,
    registry_name = "system",
}

crate::impl_registry_deserialize! {
    trait = crate::system::System,
    deserialize_fn = crate::system::deserialize_system_impl,
}

#[macro_export]
macro_rules! register_system {
    ($ty:ty, $kind:expr $(,)?) => {
        $crate::submit_registration!(
            registration = $crate::system::SystemRegistration,
            trait = $crate::system::System,
            type = $ty,
            kind = $kind,
            registry_name = "system",
        );
    };
}
