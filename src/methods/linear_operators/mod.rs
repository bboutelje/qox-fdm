use crate::types::Real;

pub mod tridiagonal_operator;

pub trait LinearOperator<T: Real> {
    /// Returns the number of grid nodes (N)
    fn size(&self) -> usize;

    /// Computes out = L(t) * v
    /// Equivalent to the 'function evaluation' f(t, y) in ODE solvers.
    fn apply_into(&self, v: &[T], out: &mut [T]);

    fn setup_coeff(&self, coeff: T);

    /// Solves (I - coeff * L(t)) * x = b
    /// Writes the result into 'dest'.
    /// This is where the Thomas Algorithm (TDMA) lives.
    fn solve_inverse_into(&self, coeff: T, b: &[T], out: &mut [T], z_buffer: &mut [T]);

    fn set_boundary_row(&mut self, row_idx: usize, weights: &[T]);
}
