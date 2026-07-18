pub mod backward_euler;
pub mod input_vectors;

use crate::{methods::time_stepping::input_vectors::InputVector, types::Real};

pub struct GlmTableau<T, const S: usize, const R: usize> {
    pub a: [[T; S]; S],
    pub u: [[T; R]; S],
    pub b: [[T; S]; R],
    pub v: [[T; R]; R],
    pub c: [T; S],
}

pub struct GlmWorkspace<T> {
    pub stages: Vec<T>,
    pub stage_derivatives: Vec<T>,
    pub rhs_buffer: Vec<T>,
    pub z_buffer: Vec<T>,
}

impl<T: Real> GlmWorkspace<T> {
    pub fn new(s: usize, n: usize) -> Self {
        let zero = T::zero();
        Self {
            stages: vec![zero; s * n],
            stage_derivatives: vec![zero; s * n],
            rhs_buffer: vec![zero; n],
            z_buffer: vec![zero; n],
        }
    }
}

pub trait TimeStepper<T: Real, IV: InputVector<T>, const S: usize, const R: usize> {
    fn tableau(&self) -> &GlmTableau<T, S, R>;
    fn external_stages(&self) -> usize {
        R
    }
    fn prepare_stage_rhs(
        &self,
        stage_idx: usize,
        state: &IV,
        scaled_stage_derivatives: &[T],
        dt: T,
        rhs_out: &mut [T],
    );

    fn finalize_step(&self, state: &mut IV, dt: T, ws: &GlmWorkspace<T>);
}
