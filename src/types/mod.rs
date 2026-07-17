pub mod f64;

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait Real:
    Sized
    + Copy
    + Debug
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Div<Self, Output = Self>
    + Neg<Output = Self>
    + PartialOrd
    + PartialEq
    + From<f64>
{
    fn from_f64(v: f64) -> Self;
    fn from_usize(v: usize) -> Self;
    fn scalar(self) -> f64;

    fn gradient(&self, k: usize) -> Option<f64>;

    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;

    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn asinh(self) -> Self;
    fn cos(self) -> Self;

    fn abs(self) -> Self;

    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, n: Self) -> Self;
    fn norm_cdf(self) -> Self;
    fn norm_pdf(self) -> Self;

    #[inline(always)]
    fn zero() -> Self {
        Self::from_f64(0.0)
    }

    #[inline(always)]
    fn half() -> Self {
        Self::from_f64(0.5)
    }

    #[inline(always)]
    fn one() -> Self {
        Self::from_f64(1.0)
    }

    #[inline(always)]
    fn two() -> Self {
        Self::from_f64(2.0)
    }

    fn pi() -> Self {
        Self::from_f64(std::f64::consts::PI)
    }

    fn epsilon() -> Self {
        Self::from_f64(std::f64::EPSILON)
    }
}
