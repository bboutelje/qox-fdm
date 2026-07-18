use crate::types::Real;

use crate::methods::{
    finite_difference::grids::grid_type::GridType, linear_operators::LinearOperator,
};

pub mod neumann_ghost;
pub mod neumann_standard;
pub mod unified;

pub enum BoundarySide {
    Lower,
    Upper,
}

pub trait BoundaryCondition<T, Tr, L, Pde>
where
    T: Real,
    L: LinearOperator<T>,
{
    /// Applies the boundary condition by modifying the specific row coefficients
    /// and accounting for any coordinate transformation.
    fn apply(&self, side: BoundarySide, grid: &GridType<T, Tr>, pde: &Pde, t: T, operator: &mut L);

    /// If the boundary changes the right-hand-side vector (e.g., non-zero Dirichlet),
    /// modify the vector $b$ prior to solving the implicit system.
    fn update_rhs(&self, side: BoundarySide, grid: &GridType<T, Tr>, pde: &Pde, t: T, b: &mut [T]);
}
