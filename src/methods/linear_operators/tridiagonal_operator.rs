use crate::types::Real;

use crate::methods::linear_operators::LinearOperator;
use std::cell::RefCell;

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
        if n == 0 {
            return;
        }

        // --- AVX-512 Path (Compile-Time Gated) ---
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "fma"
        ))]
        if std::mem::size_of::<T>() == std::mem::size_of::<f64>() {
            unsafe {
                let lower = std::slice::from_raw_parts(self.lower.as_ptr() as *const f64, n);
                let diag = std::slice::from_raw_parts(self.diag.as_ptr() as *const f64, n);
                let upper = std::slice::from_raw_parts(self.upper.as_ptr() as *const f64, n);
                let v = std::slice::from_raw_parts(v.as_ptr() as *const f64, v.len());
                let out = std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f64, out.len());

                TridiagonalOperator::<f64>::apply_into_avx512_f64(lower, diag, upper, v, out);
            }
            return;
        }

        // --- AVX2 Path (Compile-Time Gated) ---
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ))]
        if std::mem::size_of::<T>() == std::mem::size_of::<f64>() {
            unsafe {
                let lower = std::slice::from_raw_parts(self.lower.as_ptr() as *const f64, n);
                let diag = std::slice::from_raw_parts(self.diag.as_ptr() as *const f64, n);
                let upper = std::slice::from_raw_parts(self.upper.as_ptr() as *const f64, n);
                let v = std::slice::from_raw_parts(v.as_ptr() as *const f64, v.len());
                let out = std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f64, out.len());

                TridiagonalOperator::<f64>::apply_into_avx2_f64(lower, diag, upper, v, out);
            }
            return;
        }

        out[0] = self.diag[0] * v[0] + self.upper[0] * v[1];

        for i in 1..n - 1 {
            out[i] = self.lower[i] * v[i - 1] + self.diag[i] * v[i] + self.upper[i] * v[i + 1];
        }

        let last = n - 1;
        out[last] = self.lower[last] * v[last - 1] + self.diag[last] * v[last];
    }

    fn setup_coeff(&self, coeff: T) {
        let mut cache = self.cache.borrow_mut();

        if let Some(ref c) = *cache {
            if (c.coeff - coeff).abs() < T::from_f64(1e-12) {
                return;
            }
        }

        let n = self.size();
        let mut a_prime = vec![T::zero(); n];
        let mut c_prime = vec![T::zero(); n];
        let mut m_inv = vec![T::zero(); n];

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
        self.setup_coeff(coeff);

        let cache_ref = self.cache.borrow();
        let c = cache_ref
            .as_ref()
            .expect("Cache should be populated by setup_coeff");

        let a_prime = &c.a_prime;
        let c_prime = &c.c_prime;
        let m_inv = &c.m_inv;

        let n = b.len();
        if n == 0 {
            return;
        }

        // --- Step 1: Forward Substitution (Ly = b) ---
        let (z_first, z_rest) = z_buffer.split_first_mut().unwrap();
        let (b_first, b_rest) = b.split_first().unwrap();

        *z_first = *b_first * m_inv[0];

        let mut z_prev = *z_first;
        for (((z_out, &b_val), &a_val), &m_val) in z_rest
            .iter_mut()
            .zip(b_rest)
            .zip(&a_prime[1..])
            .zip(&m_inv[1..])
        {
            let z_curr = (-a_val).mul_add(z_prev, b_val) * m_val;
            *z_out = z_curr;
            z_prev = z_curr;
        }

        // --- Step 2: Backward Substitution (Ux = z) ---
        let last_idx = n - 1;
        x[last_idx] = z_buffer[last_idx];

        if n > 1 {
            let mut x_next = x[last_idx];

            let x_rev = &mut x[..last_idx];
            let z_rev = &z_buffer[..last_idx];
            let c_rev = &c_prime[..last_idx];

            for ((x_out, &z_val), &c_val) in x_rev.iter_mut().zip(z_rev).zip(c_rev).rev() {
                let x_curr = (-c_val).mul_add(x_next, z_val);
                *x_out = x_curr;
                x_next = x_curr;
            }
        }
    }

    fn set_boundary_row(&mut self, row_idx: usize, weights: &[T]) {
        let n = self.size();

        *self.cache.get_mut() = None;

        if row_idx == 0 {
            self.diag[0] = weights[0];
            if weights.len() > 1 {
                self.upper[0] = weights[1];
            }
            self.lower[0] = T::zero();
        } else if row_idx == n - 1 {
            if weights.len() > 1 {
                self.lower[row_idx] = weights[0];
                self.diag[row_idx] = weights[1];
            } else {
                self.diag[row_idx] = weights[0];
            }
            self.upper[row_idx] = T::zero();
        } else {
            if weights.len() >= 3 {
                self.lower[row_idx] = weights[0];
                self.diag[row_idx] = weights[1];
                self.upper[row_idx] = weights[2];
            }
        }
    }
}

impl TridiagonalOperator<f64> {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    #[inline(always)]
    unsafe fn apply_into_avx2_f64(
        lower: &[f64],
        diag: &[f64],
        upper: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) {
        use std::arch::x86_64::*;

        let n = diag.len();
        out[0] = diag[0].mul_add(v[0], upper[0] * v[1]);

        if n <= 2 {
            if n == 2 {
                out[1] = lower[1].mul_add(v[0], diag[1] * v[1]);
            }
            return;
        }

        let mut i = 1;

        // Process in batches of 4 x double precision (256-bit registers)
        while i + 4 <= n - 1 {
            unsafe {
                let l_vec = _mm256_loadu_pd(lower.as_ptr().add(i));
                let d_vec = _mm256_loadu_pd(diag.as_ptr().add(i));
                let u_vec = _mm256_loadu_pd(upper.as_ptr().add(i));

                let v_prev = _mm256_loadu_pd(v.as_ptr().add(i - 1));
                let v_curr = _mm256_loadu_pd(v.as_ptr().add(i));
                let v_next = _mm256_loadu_pd(v.as_ptr().add(i + 1));

                let res = _mm256_fmadd_pd(
                    l_vec,
                    v_prev,
                    _mm256_fmadd_pd(d_vec, v_curr, _mm256_mul_pd(u_vec, v_next)),
                );

                _mm256_storeu_pd(out.as_mut_ptr().add(i), res);
            }
            i += 4;
        }

        // Scalar tail handling for remaining elements (< 4)
        while i < n - 1 {
            out[i] = lower[i].mul_add(v[i - 1], diag[i].mul_add(v[i], upper[i] * v[i + 1]));
            i += 1;
        }

        let last = n - 1;
        out[last] = lower[last].mul_add(v[last - 1], diag[last] * v[last]);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "fma"
    ))]
    #[inline(always)]
    unsafe fn apply_into_avx512_f64(
        lower: &[f64],
        diag: &[f64],
        upper: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) {
        use std::arch::x86_64::*;

        let n = diag.len();
        out[0] = diag[0].mul_add(v[0], upper[0] * v[1]);

        if n <= 2 {
            if n == 2 {
                out[1] = lower[1].mul_add(v[0], diag[1] * v[1]);
            }
            return;
        }

        let mut i = 1;

        while i + 8 <= n - 1 {
            unsafe {
                let l_vec = _mm512_loadu_pd(lower.as_ptr().add(i));
                let d_vec = _mm512_loadu_pd(diag.as_ptr().add(i));
                let u_vec = _mm512_loadu_pd(upper.as_ptr().add(i));

                let v_prev = _mm512_loadu_pd(v.as_ptr().add(i - 1));
                let v_curr = _mm512_loadu_pd(v.as_ptr().add(i));
                let v_next = _mm512_loadu_pd(v.as_ptr().add(i + 1));

                let res = _mm512_fmadd_pd(
                    l_vec,
                    v_prev,
                    _mm512_fmadd_pd(d_vec, v_curr, _mm512_mul_pd(u_vec, v_next)),
                );

                _mm512_storeu_pd(out.as_mut_ptr().add(i), res);
            }
            i += 8;
        }

        let rem = (n - 1) - i;
        if rem > 0 {
            let mask: __mmask8 = (1u8 << rem) - 1;

            unsafe {
                let l_vec = _mm512_maskz_loadu_pd(mask, lower.as_ptr().add(i));
                let d_vec = _mm512_maskz_loadu_pd(mask, diag.as_ptr().add(i));
                let u_vec = _mm512_maskz_loadu_pd(mask, upper.as_ptr().add(i));

                let v_prev = _mm512_maskz_loadu_pd(mask, v.as_ptr().add(i - 1));
                let v_curr = _mm512_maskz_loadu_pd(mask, v.as_ptr().add(i));
                let v_next = _mm512_maskz_loadu_pd(mask, v.as_ptr().add(i + 1));

                let res = _mm512_fmadd_pd(
                    l_vec,
                    v_prev,
                    _mm512_fmadd_pd(d_vec, v_curr, _mm512_mul_pd(u_vec, v_next)),
                );

                _mm512_mask_storeu_pd(out.as_mut_ptr().add(i), mask, res);
            }
        }

        let last = n - 1;
        out[last] = lower[last].mul_add(v[last - 1], diag[last] * v[last]);
    }
}
