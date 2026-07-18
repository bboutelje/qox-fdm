use crate::methods::finite_difference::grids::fitted::FittedGrid1d;

pub mod fitted;
pub mod grid_type;
pub mod uniform;

pub trait Grid1d<T, Tr> {
    fn centers(&self) -> &[T];
    fn h_plus(&self) -> &[T];
    fn h_minus(&self) -> &[T];

    fn size(&self) -> usize;
    fn location(&self, index: usize) -> T;
    fn apply_physical_jump(&self, jump: T) -> FittedGrid1d<T, Tr>;

    fn get_transform(&self) -> Tr;
}
