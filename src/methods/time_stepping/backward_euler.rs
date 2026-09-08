use crate::{
    methods::time_stepping::{
        GlmTableau, GlmWorkspace, TimeStepper,
        input_vectors::{InputVector, jet_vector::JetVector},
    },
    types::Real,
};

pub struct BackwardEuler<T: Real> {
    tableau: GlmTableau<T, 1, 2>,
}

impl<T: Real> BackwardEuler<T> {
    pub fn new() -> Self {
        let one = T::one();
        let zero = T::zero();
        Self {
            tableau: GlmTableau {
                a: [[one]],
                u: [[one, zero]],
                b: [[one], [one]],
                v: [[one, zero], [zero, zero]],
                c: [one],
            },
        }
    }
}

impl<T: Real> TimeStepper<T, JetVector<T>, 1, 2> for BackwardEuler<T> {
    fn tableau(&self) -> &GlmTableau<T, 1, 2> {
        &self.tableau
    }

    fn prepare_stage_rhs(
        &self,
        _stage_idx: usize,
        state: &JetVector<T>,
        _l_stages: &[T],
        _dt: T,
        rhs_out: &mut [T],
    ) {
        rhs_out.copy_from_slice(state.step_slice(0));
    }

    fn finalize_step(&self, state: &mut JetVector<T>, _dt: T, ws: &GlmWorkspace<T>) {
        let n = state.n;

        state.step_slice_mut(0).copy_from_slice(&ws.stages[0..n]);

        state
            .step_slice_mut(1)
            .copy_from_slice(&ws.stage_derivatives[0..n]);
    }
}
