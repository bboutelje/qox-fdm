use libm;
use crate::types::Real;

impl Real for f64 {
    fn from_f64(val: f64) -> Self {
        val
    }

    fn from_usize(val: usize) -> Self {
        Self::from_f64(val as f64)
    }

    fn scalar(self) -> f64 {
        self
    }

    fn gradient(&self, _k: usize) -> Option<f64> {
        None
    }

    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }

    fn powf(self, n: Self) -> Self {
        f64::powf(self, n)
    }

    fn norm_cdf(self) -> Self {
        0.5 * (1.0 + libm::erf(self / std::f64::consts::SQRT_2))
    }

    fn norm_pdf(self) -> Self {
        if self.is_infinite() {
            return Self::zero();
        }

        let pi = Self::pi();
        let two = Self::one() + Self::one();
        (-self * self / two).exp() / (two * pi).sqrt()
    }

    fn sinh(self) -> Self {
        self.sinh()
    }

    fn cosh(self) -> Self {
        self.cosh()
    }

    fn asinh(self) -> Self {
        self.asinh()
    }

    fn cos(self) -> Self {
        self.cos()
    }
}
