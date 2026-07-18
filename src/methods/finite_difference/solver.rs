use crate::methods::time_stepping::GlmWorkspace;
use crate::types::Real;

use crate::methods::{
    finite_difference::{FdmProcess, boundary_conditions::BoundaryCondition, obstacle::Obstacle},
    linear_operators::LinearOperator,
    stepping_scheme::SteppingScheme,
    time_stepping::{TimeStepper, input_vectors::jet_vector::JetVector},
    transforms::Transform,
};

pub struct Solver;
impl Solver {
    pub fn solve_into<T, Tr, L, P, Lbc, Ubc, Step, O, SP, const S: usize>(
        &self,
        stepper: &Step,
        external_vector: &mut JetVector<T>,
        dt: T,
        step_policy: &SP,
        workspace: &mut GlmWorkspace<T>,
        time_steps: usize,
    ) where
        T: Real,
        Tr: Transform<T> + Copy,
        Step: TimeStepper<T, JetVector<T>, S, 2>,
        L: LinearOperator<T>,
        O: Obstacle<T> + Copy,
        P: FdmProcess<T, Tr, L>,
        Lbc: BoundaryCondition<T, Tr, L, P>,
        Ubc: BoundaryCondition<T, Tr, L, P>,
        SP: SteppingScheme<T, Tr, O, P, Lbc, Ubc>,
    {
        for _ in 0..time_steps {
            for stage_idx in 0..S {
                stepper.prepare_stage_rhs(
                    stage_idx,
                    external_vector,
                    &workspace.stage_derivatives,
                    dt,
                    &mut workspace.rhs_buffer,
                );

                let stage_coeff = stepper.tableau().a[stage_idx][stage_idx] * dt;
                step_policy.solve_vector_into(stage_idx, stage_coeff, workspace);
            }

            stepper.finalize_step(external_vector, dt, &workspace);
        }
    }
}
