use crate::types::Real;

use crate::methods::{
    finite_difference::{
        boundary_conditions::{
            BoundaryCondition, BoundarySide, neumann_ghost::NeumannGhost2ndOrder,
            neumann_standard::NeumannStandard,
        },
        grids::grid_type::GridType,
    },
    linear_operators::LinearOperator,
};

pub enum UnifiedBoundaryCondition<T> {
    NeumannStandard(NeumannStandard<T>),
    NeumannGhost2ndOrder(NeumannGhost2ndOrder<T>),
}

impl<T, Tr, L, Pde> BoundaryCondition<T, Tr, L, Pde> for UnifiedBoundaryCondition<T>
where
    T: Real,
    L: LinearOperator<T>,
    NeumannStandard<T>: BoundaryCondition<T, Tr, L, Pde>,
    NeumannGhost2ndOrder<T>: BoundaryCondition<T, Tr, L, Pde>,
{
    fn apply(&self, side: BoundarySide, grid: &GridType<T, Tr>, pde: &Pde, t: T, operator: &mut L) {
        match self {
            Self::NeumannStandard(bc) => bc.apply(side, grid, pde, t, operator),
            Self::NeumannGhost2ndOrder(bc) => bc.apply(side, grid, pde, t, operator),
        }
    }

    fn update_rhs(&self, side: BoundarySide, grid: &GridType<T, Tr>, pde: &Pde, t: T, b: &mut [T]) {
        match self {
            Self::NeumannStandard(bc) => bc.update_rhs(side, grid, pde, t, b),
            Self::NeumannGhost2ndOrder(bc) => bc.update_rhs(side, grid, pde, t, b),
        }
    }
}
