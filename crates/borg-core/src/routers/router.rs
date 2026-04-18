use anyhow::Result;

use crate::module::Module;
use crate::types::{InstanceLoad, Request};

pub trait Router: Module {
    fn select_instance(&mut self, request: &Request, instances: &[InstanceLoad]) -> Result<usize>;
}

crate::declare_registry! {
    registration = RouterRegistration,
    trait = crate::router::Router,
    deserialize_fn = deserialize_router_impl,
    registry_name = "router",
}

crate::impl_registry_deserialize! {
    trait = crate::router::Router,
    deserialize_fn = crate::router::deserialize_router_impl,
}

#[macro_export]
macro_rules! register_router {
    ($ty:ty, $kind:expr $(,)?) => {
        $crate::submit_registration!(
            registration = $crate::router::RouterRegistration,
            trait = $crate::router::Router,
            type = $ty,
            kind = $kind,
            registry_name = "router",
        );
    };
}
