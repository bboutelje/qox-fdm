use crate::types::Real;

use crate::methods::linear_operators::LinearOperator;
use std::cell::RefCell;

// Helper to store the factorization results
struct TridiagonalCache<T> {
    coeff: T,
    a_prime: Vec<T>,
    c_prime: Vec<T>,
    m_inv: Vec<T>,
}

pub struct TridiagonalOperator<T> {
    pub lower: Vec<T>,
    pub diag: Vec<T>,
    pub upper: Vec<T>,
    cache: RefCell<Option<TridiagonalCache<T>>>,
}

impl<T: Real> TridiagonalOperator<T> {
    pub fn new(lower: Vec<T>, diag: Vec<T>, upper: Vec<T>) -> Self {
        let n = diag.len();
        if n < 3 {
            panic!(
                "TridiagonalOperator requires at least 3 elements, found {}",
                n
            );
        }

        TridiagonalOperator {
            lower,
            diag,
            upper,
            cache: RefCell::new(None),
        }
    }
}

impl<T: Real> LinearOperator<T> for TridiagonalOperator<T> {
    fn size(&self) -> usize {
        self.diag.len()
    }

    fn apply_into(&self, v: &[T], out: &mut [T]) {
        let n = self.size();

        out[0] = self.diag[0] * v[0] + self.upper[0] * v[1];

        for i in 1..n - 1 {
            out[i] = self.lower[i] * v[i - 1] + self.diag[i] * v[i] + self.upper[i] * v[i + 1];
        }

        let last = n - 1;
        out[last] = self.lower[last] * v[last - 1] + self.diag[last] * v[last];
    }

    fn setup_coeff(&self, coeff: T) {
        let mut cache = self.cache.borrow_mut();

        // Skip if already computed for this coeff
        if let Some(ref c) = *cache {
            if (c.coeff - coeff).abs() < T::from_f64(1e-12) {
                return;
            }
        }

        let n = self.size();
        let mut a_prime = vec![T::zero(); n];
        let mut c_prime = vec![T::zero(); n];
        let mut m_inv = vec![T::zero(); n];

        // Solve (I - coeff * L)x = b
        // Diagonal: d_i' = 1 - coeff * diag[i]
        // Off-diagonals: a_i = -coeff * lower[i], c_i = -coeff * upper[i]

        a_prime[0] = T::zero();

        let d0 = T::one() - coeff * self.diag[0];
        m_inv[0] = T::one() / d0;
        c_prime[0] = (-coeff * self.upper[0]) * m_inv[0];

        for i in 1..n {
            a_prime[i] = -coeff * self.lower[i];
            let d = T::one() - coeff * self.diag[i];
            let c = if i < n - 1 {
                -coeff * self.upper[i]
            } else {
                T::zero()
            };

            let m = d - a_prime[i] * c_prime[i - 1];
            m_inv[i] = T::one() / m;
            c_prime[i] = c * m_inv[i];
        }

        *cache = Some(TridiagonalCache {
            coeff,
            a_prime,
            c_prime,
            m_inv,
        });
    }

    fn solve_inverse_into(&self, coeff: T, b: &[T], x: &mut [T], z_buffer: &mut [T]) {
        // 1. Ensure the cache is synchronized with the provided coeff
        self.setup_coeff(coeff);

        // 2. Proceed with the solve using the (now guaranteed) cached values
        let cache_ref = self.cache.borrow();
        let c = cache_ref
            .as_ref()
            .expect("Cache should be populated by setup_coeff");
        let n = self.size();

        z_buffer[0] = b[0] * c.m_inv[0];
        for i in 1..n {
            z_buffer[i] = (b[i] - c.a_prime[i] * z_buffer[i - 1]) * c.m_inv[i];
        }

        x[n - 1] = z_buffer[n - 1];
        for i in (0..n - 1).rev() {
            x[i] = z_buffer[i] - c.c_prime[i] * x[i + 1];
        }
    }

    fn set_boundary_row(&mut self, row_idx: usize, weights: &[T]) {
        let n = self.size();

        // 1. Clear the cache since matrix coefficients have altered
        *self.cache.get_mut() = None;

        if row_idx == 0 {
            // Lower boundary: Only touches the first row.
            // weights[0] maps to diag[0], weights[1] maps to upper[0]
            self.diag[0] = weights[0];
            if weights.len() > 1 {
                self.upper[0] = weights[1];
            }
            // lower[0] is structurally outside the matrix boundary, but keep it clean
            self.lower[0] = T::zero();

            // println!("Diag: {:?}", self.diag[0]);
            // println!("Upper: {:?}", self.upper[0]);
        } else if row_idx == n - 1 {
            // Upper boundary: Only touches the final row.
            // For a 2-point relation layout passed from your BC:
            // weights[0] is the weight on the inner neighbor (lower[n-1])
            // weights[1] is the weight on the boundary itself (diag[n-1])
            if weights.len() > 1 {
                self.lower[row_idx] = weights[0];
                self.diag[row_idx] = weights[1];
            } else {
                self.diag[row_idx] = weights[0];
            }
            // upper[n-1] is structurally outside the matrix boundary, keep it clean
            self.upper[row_idx] = T::zero();
            // println!("Diag: {:?}", self.diag[row_idx]);
            // println!("Lower: {:?}", self.lower[row_idx]);
        } else {
            // Optional fallback: handling generic interior rows if ever needed
            if weights.len() >= 3 {
                self.lower[row_idx] = weights[0];
                self.diag[row_idx] = weights[1];
                self.upper[row_idx] = weights[2];
            }
        }
    }
}
