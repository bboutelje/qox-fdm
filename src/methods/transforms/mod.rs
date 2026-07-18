pub mod identity;
pub mod log;
pub mod sinh;
pub mod unified;

pub trait Transform<T> {
    fn to_transform(&self, physical: T) -> T;
    fn to_physical(&self, mesh: T) -> T;
    fn jacobian(&self, xi: T) -> T;
    fn hessian(&self, xi: T) -> T;
}

pub enum TransformConfig {
    Log,
    Sinh { alpha: f64 },
}
