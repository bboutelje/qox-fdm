pub mod boundary_conditions;
pub mod fdm_process;
pub mod grids;
pub mod initial_conditions;
pub mod obstacle;
pub mod solver;

use crate::{
    methods::{
        finite_difference::grids::grid_type::GridType, linear_operators::LinearOperator,
        transforms::Transform,
    },
    types::Real,
};

pub trait FdmProcess<T: Real, Tr: Transform<T> + Copy, L: LinearOperator<T>> {
    fn build_operator(&self, grid: &GridType<T, Tr>, t: T) -> L;
}
