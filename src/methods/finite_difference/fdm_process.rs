use crate::types::Real;

use crate::{
    methods::{
        finite_difference::{
            FdmProcess,
            grids::{Grid1d, grid_type::GridType},
        },
        linear_operators::tridiagonal_operator::TridiagonalOperator,
        transforms::Transform,
    },
    pdes::ParabolicPde1d,
};

impl<T, Tr, P> FdmProcess<T, Tr, TridiagonalOperator<T>> for P
where
    T: Real,
    Tr: Transform<T> + Copy,
    P: ParabolicPde1d<T>,
{
    fn build_operator(&self, grid: &GridType<T, Tr>, t: T) -> TridiagonalOperator<T> {
        let n = grid.size();
        let centers = grid.centers();
        let h_minus = grid.h_minus();
        let h_plus = grid.h_plus();

        let mut lower = vec![T::zero(); n];
        let mut diag = vec![T::zero(); n];
        let mut upper = vec![T::zero(); n];

        for i in 1..n - 1 {
            let xi = centers[i];
            let hm = h_minus[i];
            let hp = h_plus[i];

            // 1. Map back to physical space to get PDE coefficients
            let s = grid.get_transform().to_physical(xi);
            let pde_a = self.a(s, t);
            let pde_b = self.b(s, t);
            let pde_c = self.c(s, t);

            // 2. Apply Chain Rule for the transformation: x = f(s)
            // L = a(s) ∂²/∂s² + b(s) ∂/∂s + c(s)
            // In transform space ξ:
            // a_new = a(s) / (J²)
            // b_new = b(s) / J - (a(s) * H) / J³
            let j = grid.get_transform().jacobian(xi);
            let h = grid.get_transform().hessian(xi);
            let j2 = j * j;

            let a_xi = pde_a / j2;
            let b_xi = (pde_b / j) - (pde_a * h) / (j2 * j);
            let c_xi = pde_c;

            // 3. Finite Difference Weights (Non-uniform)
            let denom = hm * hp * (hm + hp);

            let d2_lower = (T::from_f64(2.0) * hp) / denom;
            let d2_diag = (T::from_f64(-2.0) * (hm + hp)) / denom;
            let d2_upper = (T::from_f64(2.0) * hm) / denom;

            let d1_lower = -(hp * hp) / denom;
            let d1_diag = (hp * hp - hm * hm) / denom;
            let d1_upper = (hm * hm) / denom;

            lower[i] = a_xi * d2_lower + b_xi * d1_lower;
            diag[i] = a_xi * d2_diag + b_xi * d1_diag + c_xi;
            upper[i] = a_xi * d2_upper + b_xi * d1_upper;
        }

        TridiagonalOperator::<T>::new(lower, diag, upper)
    }
}
