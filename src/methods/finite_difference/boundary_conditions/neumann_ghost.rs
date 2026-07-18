use crate::types::Real;

use crate::{
    methods::{
        finite_difference::{
            boundary_conditions::{BoundaryCondition, BoundarySide},
            grids::{Grid1d, grid_type::GridType},
        },
        linear_operators::{LinearOperator, tridiagonal_operator::TridiagonalOperator},
        transforms::Transform,
    },
    pdes::ParabolicPde1d,
};

/// A constant-slope physical boundary condition (∂V/∂S = constant).
/// Automatically maps a target physical slope into transform space using the Jacobian.
pub struct NeumannGhost2ndOrder<T> {
    physical_slope: T,
}

impl<T> NeumannGhost2ndOrder<T> {
    pub fn new(physical_slope: T) -> Self {
        Self { physical_slope }
    }
}

impl<T, Tr, Pde> BoundaryCondition<T, Tr, TridiagonalOperator<T>, Pde> for NeumannGhost2ndOrder<T>
where
    T: Real,
    Tr: Transform<T> + Copy,
    Pde: ParabolicPde1d<T> + Copy,
{
    fn apply(
        &self,
        side: BoundarySide,
        grid: &GridType<T, Tr>,
        pde: &Pde,
        t: T,
        operator: &mut TridiagonalOperator<T>,
    ) {
        match side {
            BoundarySide::Lower => {
                let h = grid.h_plus()[0];
                let u_first = grid.centers()[0];

                // Evaluate PDE coefficients at the boundary point
                let a_first = pde.a(u_first, t);
                let c_first = pde.c(u_first, t);

                let diag = c_first - T::two() * a_first / (h * h);
                let super_diag = T::two() * a_first / (h * h);

                operator.set_boundary_row(0, &[diag, super_diag]);
            }
            BoundarySide::Upper => {
                let n = operator.size();
                let h = grid.h_minus()[n - 1];
                let u_last = grid.centers()[n - 1];

                // Evaluate PDE coefficients at the boundary point
                let a_last = pde.a(u_last, t);
                let c_last = pde.c(u_last, t);

                // Ghost-cell elimination (upper): u_{n+1} replaced via Neumann condition
                // ∂u/∂x ≈ (u_{n+1} - u_{n-1}) / (2h) = g  =>  u_{n+1} = u_{n-1} + 2hg
                // The 2hg term goes to the RHS (handled in update_rhs).
                // Sub-diagonal gets 2a/h², diagonal gets c - 2a/h².
                let sub_diag = T::two() * a_last / (h * h);
                let diag = c_last - T::two() * a_last / (h * h);

                operator.set_boundary_row(n - 1, &[sub_diag, diag]);
            }
        }
    }

    fn update_rhs(&self, side: BoundarySide, grid: &GridType<T, Tr>, pde: &Pde, t: T, b: &mut [T]) {
        let n = b.len();
        let transform = grid.get_transform();
        let centers = grid.centers();

        match side {
            BoundarySide::Lower => {
                let h = grid.h_plus()[0];
                let u_first = centers[0];
                let jacobian = transform.jacobian(u_first);

                let g = self.physical_slope / jacobian;

                let a_first = pde.a(u_first, t);
                let b_first = pde.b(u_first, t);

                b[0] += (b_first - T::two() * a_first / h) * g;
            }
            BoundarySide::Upper => {
                let h = grid.h_minus()[n - 1];
                let u_last = centers[n - 1];
                let jacobian = transform.jacobian(u_last);

                let g = self.physical_slope / jacobian;

                let a_last = pde.a(u_last, t);
                let b_last = pde.b(u_last, t);

                b[n - 1] += (b_last + T::two() * a_last / h) * g;
            }
        }
    }
}
