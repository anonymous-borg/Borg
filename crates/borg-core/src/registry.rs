use std::fmt;

use serde_json::Value;

#[derive(Debug)]
pub enum RegistryError {
    MissingKind {
        registry: &'static str,
    },
    UnknownKind {
        registry: &'static str,
        kind: String,
    },
    DuplicateKind {
        registry: &'static str,
        kind: String,
    },
    Deserialize {
        registry: &'static str,
        error: String,
    },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingKind { registry } => write!(f, "missing {registry} kind field"),
            Self::UnknownKind { registry, kind } => {
                write!(f, "unknown {registry} kind '{kind}'")
            }
            Self::DuplicateKind { registry, kind } => {
                write!(f, "duplicate {registry} kind '{kind}'")
            }
            Self::Deserialize { registry, error } => {
                write!(f, "failed to deserialize {registry}: {error}")
            }
        }
    }
}

impl std::error::Error for RegistryError {}

impl RegistryError {
    pub fn deserialize(registry: &'static str, error: impl ToString) -> Self {
        Self::Deserialize {
            registry,
            error: error.to_string(),
        }
    }
}

pub fn extract_kind<'a>(
    value: &'a Value,
    registry: &'static str,
) -> std::result::Result<&'a str, RegistryError> {
    value
        .get("kind")
        .and_then(Value::as_str)
        .ok_or(RegistryError::MissingKind { registry })
}

pub fn strip_kind(mut value: Value) -> Value {
    if let Value::Object(ref mut object) = value {
        object.remove("kind");
    }
    value
}

pub fn find_registration<'a, R>(
    kind: &str,
    registry: &'static str,
    registrations: impl IntoIterator<Item = &'a R>,
    registration_kind: impl Fn(&R) -> &'static str,
) -> std::result::Result<&'a R, RegistryError>
where
{
    let mut found = None;
    for registration in registrations {
        if registration_kind(registration) != kind {
            continue;
        }
        if found.is_some() {
            return Err(RegistryError::DuplicateKind {
                registry,
                kind: kind.to_string(),
            });
        }
        found = Some(registration);
    }

    found.ok_or_else(|| RegistryError::UnknownKind {
        registry,
        kind: kind.to_string(),
    })
}

#[macro_export]
macro_rules! declare_registry {
    (
        registration = $registration:ident,
        trait = $trait:path,
        deserialize_fn = $deserialize_fn:ident,
        registry_name = $registry_name:expr $(,)?
    ) => {
        pub struct $registration {
            pub kind: &'static str,
            pub deserialize: fn(
                $crate::__private::serde_json::Value,
            ) -> ::std::result::Result<
                ::std::boxed::Box<dyn $trait>,
                $crate::registry::RegistryError,
            >,
        }

        $crate::__private::inventory::collect!($registration);

        pub fn $deserialize_fn(
            value: $crate::__private::serde_json::Value,
        ) -> ::std::result::Result<::std::boxed::Box<dyn $trait>, $crate::registry::RegistryError> {
            let kind = $crate::registry::extract_kind(&value, $registry_name)?;
            let registration = $crate::registry::find_registration(
                kind,
                $registry_name,
                $crate::__private::inventory::iter::<$registration>,
                |registration| registration.kind,
            )?;
            (registration.deserialize)(value)
        }
    };
}

#[macro_export]
macro_rules! impl_registry_deserialize {
    (
        trait = $trait:path,
        deserialize_fn = $deserialize_fn:path $(,)?
    ) => {
        impl<'de> ::serde::Deserialize<'de> for ::std::boxed::Box<dyn $trait> {
            fn deserialize<D>(deserializer: D) -> ::std::result::Result<Self, D::Error>
            where
                D: ::serde::Deserializer<'de>,
            {
                let value =
                    <$crate::__private::serde_json::Value as ::serde::Deserialize>::deserialize(
                        deserializer,
                    )?;
                $deserialize_fn(value).map_err(::serde::de::Error::custom)
            }
        }
    };
}

#[macro_export]
macro_rules! submit_registration {
    (
        registration = $registration:path,
        trait = $trait:path,
        type = $ty:ty,
        kind = $kind:expr,
        registry_name = $registry_name:expr $(,)?
    ) => {
        const _: () = {
            fn deserialize(
                value: $crate::__private::serde_json::Value,
            ) -> ::std::result::Result<
                ::std::boxed::Box<dyn $trait>,
                $crate::registry::RegistryError,
            > {
                let value = $crate::registry::strip_kind(value);
                $crate::__private::serde_json::from_value::<$ty>(value)
                    .map(|value| ::std::boxed::Box::new(value) as ::std::boxed::Box<dyn $trait>)
                    .map_err(|err| {
                        $crate::registry::RegistryError::deserialize($registry_name, err)
                    })
            }

            $crate::__private::inventory::submit! {
                $registration {
                    kind: $kind,
                    deserialize,
                }
            }
        };
    };
}
