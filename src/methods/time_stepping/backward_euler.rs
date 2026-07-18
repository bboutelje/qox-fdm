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
                // Y_1 = y_n + h*f(Y_1)
                a: [[one]],
                u: [[one, zero]],
                // y_{n+1} = y_n + h*f(Y_1)
                // h*y'_{n+1} = h*f(Y_1)
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
        // Backward Euler stage Y_1 starts from the previous state y_n
        rhs_out.copy_from_slice(state.step_slice(0));
    }

    fn finalize_step(&self, state: &mut JetVector<T>, _dt: T, ws: &GlmWorkspace<T>) {
        let n = state.n;

        // Update y_{n+1} with the computed stage value
        state.step_slice_mut(0).copy_from_slice(&ws.stages[0..n]);

        // Update h*y'_{n+1} with the derivative at the stage
        state
            .step_slice_mut(1)
            .copy_from_slice(&ws.stage_derivatives[0..n]);
    }
}
