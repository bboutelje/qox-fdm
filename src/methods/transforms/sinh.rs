use crate::types::Real;

use crate::methods::transforms::Transform;

#[derive(Copy, Clone)]
pub struct SinhTransform<T> {
    k: T,
    alpha: T,
}

impl<T: Real> SinhTransform<T> {
    pub fn new(k: T, alpha: T) -> Self {
        Self { k, alpha }
    }
}

impl<T: Real> Transform<T> for SinhTransform<T> {
    fn to_transform(&self, physical: T) -> T {
        let x = (physical / self.k).ln();
        (x / self.alpha).asinh()
    }

    fn to_physical(&self, transformed: T) -> T {
        self.k * (self.alpha * transformed.sinh()).exp()
    }

    fn jacobian(&self, transformed: T) -> T {
        let physical = self.to_physical(transformed);
        self.alpha * transformed.cosh() * physical
    }

    fn hessian(&self, transformed: T) -> T {
        let physical = self.to_physical(transformed);
        let jacobian = self.jacobian(transformed);

        (self.alpha * transformed.sinh() * physical) + (self.alpha * transformed.cosh() * jacobian)
    }
}
