pub mod linear;

use crate::methods::{
    finite_difference::{grids::grid_type::GridType, initial_conditions::InitialConditions},
    time_stepping::{GlmWorkspace, input_vectors::jet_vector::JetVector},
};

pub trait SteppingScheme<T, Tr, O, P, Ubc, Lbc> {
    fn initialize_payoff<IC>(&self, intitial_conditions: IC) -> JetVector<T>
    where
        IC: InitialConditions<T> + Copy;

    fn get_grid(&self) -> &GridType<T, Tr>;

    fn apply_jump_into(&mut self, vector: &mut JetVector<T>, amount: T);

    fn solve_vector_into(&self, stage_idx: usize, coeff: T, dest: &mut GlmWorkspace<T>);

    fn operate_on_stage(&self, stage_slice: &[T], operated_on_stage_slice: &mut [T]);
}
