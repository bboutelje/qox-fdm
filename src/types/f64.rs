use crate::types::Real;
use libm;

impl Real for f64 {
    #[inline]
    fn from_f64(val: f64) -> Self {
        val
    }

    #[inline]
    fn from_usize(val: usize) -> Self {
        Self::from_f64(val as f64)
    }

    #[inline]
    fn scalar(self) -> f64 {
        self
    }

    #[inline]
    fn gradient(&self, _k: usize) -> Option<f64> {
        None
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }

    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        f64::powf(self, n)
    }

    #[inline]
    fn norm_cdf(self) -> Self {
        0.5 * (1.0 + libm::erf(self / std::f64::consts::SQRT_2))
    }

    #[inline]
    fn norm_pdf(self) -> Self {
        if self.is_infinite() {
            return Self::zero();
        }

        let pi = Self::pi();
        let two = Self::one() + Self::one();
        (-self * self / two).exp() / (two * pi).sqrt()
    }

    #[inline]
    fn sinh(self) -> Self {
        self.sinh()
    }

    #[inline]
    fn cosh(self) -> Self {
        self.cosh()
    }

    #[inline]
    fn asinh(self) -> Self {
        self.asinh()
    }

    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }

    #[inline]
    fn acos(self) -> Self {
        self.acos()
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }
}
