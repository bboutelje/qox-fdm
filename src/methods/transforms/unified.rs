use crate::types::Real;

use crate::methods::transforms::{Transform, log::LogTransform, sinh::SinhTransform};

#[derive(Copy, Clone)]
pub enum UnifiedTransform<T> {
    Log(LogTransform<T>),
    Sinh(SinhTransform<T>),
}

impl<T: Real> Transform<T> for UnifiedTransform<T> {
    #[inline]
    fn to_transform(&self, physical: T) -> T {
        match self {
            UnifiedTransform::Log(t) => t.to_transform(physical),
            UnifiedTransform::Sinh(t) => t.to_transform(physical),
        }
    }

    #[inline]
    fn to_physical(&self, mesh: T) -> T {
        match self {
            UnifiedTransform::Log(t) => t.to_physical(mesh),
            UnifiedTransform::Sinh(t) => t.to_physical(mesh),
        }
    }

    #[inline]
    fn jacobian(&self, xi: T) -> T {
        match self {
            UnifiedTransform::Log(t) => t.jacobian(xi),
            UnifiedTransform::Sinh(t) => t.jacobian(xi),
        }
    }

    #[inline]
    fn hessian(&self, xi: T) -> T {
        match self {
            UnifiedTransform::Log(t) => t.hessian(xi),
            UnifiedTransform::Sinh(t) => t.hessian(xi),
        }
    }
}
