use crate::types::Real;

use crate::methods::{
    finite_difference::grids::{Grid1d, fitted::FittedGrid1d, uniform::UniformGrid1d},
    transforms::Transform,
};

#[derive(Clone)]
pub enum GridType<T, Tr> {
    Uniform(UniformGrid1d<T, Tr>),
    Fitted(FittedGrid1d<T, Tr>),
}

impl<T: Real, Tr: Transform<T> + Copy> Grid1d<T, Tr> for GridType<T, Tr> {
    fn location(&self, index: usize) -> T {
        match self {
            Self::Uniform(g) => g.location(index),
            Self::Fitted(g) => g.location(index),
        }
    }

    fn centers(&self) -> &[T] {
        match self {
            Self::Uniform(g) => g.centers(),
            Self::Fitted(g) => g.centers(),
        }
    }

    fn h_plus(&self) -> &[T] {
        match self {
            Self::Uniform(g) => g.h_plus(),
            Self::Fitted(g) => g.h_plus(),
        }
    }

    fn h_minus(&self) -> &[T] {
        match self {
            Self::Uniform(g) => g.h_minus(),
            Self::Fitted(g) => g.h_minus(),
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::Uniform(g) => g.size(),
            Self::Fitted(g) => g.size(),
        }
    }

    fn apply_physical_jump(&self, jump: T) -> FittedGrid1d<T, Tr> {
        match self {
            Self::Uniform(g) => g.apply_physical_jump(jump),
            Self::Fitted(g) => g.apply_physical_jump(jump),
        }
    }

    fn get_transform(&self) -> Tr {
        match self {
            Self::Uniform(g) => g.get_transform(),
            Self::Fitted(g) => g.get_transform(),
        }
    }
}
