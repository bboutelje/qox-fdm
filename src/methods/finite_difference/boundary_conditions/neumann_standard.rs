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
/// Employs a pure 1st-order substitution directly onto the boundary rows (QuantLib style).
pub struct NeumannStandard<T> {
    physical_slope: T,
}

impl<T> NeumannStandard<T> {
    pub fn new(physical_slope: T) -> Self {
        Self { physical_slope }
    }
}

impl<T, Tr, Pde> BoundaryCondition<T, Tr, TridiagonalOperator<T>, Pde> for NeumannStandard<T>
where
    T: Real,
    Tr: Transform<T> + Copy,
    Pde: ParabolicPde1d<T> + Copy,
{
    fn apply(
        &self,
        side: BoundarySide,
        _grid: &GridType<T, Tr>,
        _pde: &Pde, // Unused because we don't evaluate PDE at the boundary row
        _t: T,
        operator: &mut TridiagonalOperator<T>,
    ) {
        match side {
            BoundarySide::Lower => {
                // Equation: (u_1 - u_0) / h = g  =>  -1/h * u_0 + 1/h * u_1 = g
                // QuantLib often normalizes this to: -1 * u_0 + 1 * u_1 = h * g
                let diag = -T::one();
                let super_diag = T::one();

                operator.set_boundary_row(0, &[diag, super_diag]);
            }
            BoundarySide::Upper => {
                let n = operator.size();
                // Equation: (u_n - u_{n-1}) / h = g => -1 * u_{n-1} + 1 * u_n = h * g
                let sub_diag = -T::one();
                let diag = T::one();

                operator.set_boundary_row(n - 1, &[sub_diag, diag]);
            }
        }
    }

    fn update_rhs(
        &self,
        side: BoundarySide,
        grid: &GridType<T, Tr>,
        _pde: &Pde,
        _t: T,
        b: &mut [T],
    ) {
        let n = b.len();
        let transform = grid.get_transform();
        let centers = grid.centers();

        match side {
            BoundarySide::Lower => {
                let h = grid.h_plus()[0];
                let u_first = centers[0];
                let jacobian = transform.jacobian(u_first);
                let g = self.physical_slope / jacobian;

                // Set the RHS value directly to match the normalized matrix row: h * g
                b[0] = h * g;
            }
            BoundarySide::Upper => {
                let h = grid.h_minus()[n - 1];
                let u_last = centers[n - 1];
                let jacobian = transform.jacobian(u_last);
                let g = self.physical_slope / jacobian;

                // Set the RHS value directly to match the normalized matrix row: h * g
                b[n - 1] = h * g;
            }
        }
    }
}
