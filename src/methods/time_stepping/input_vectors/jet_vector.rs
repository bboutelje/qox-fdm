use crate::{methods::time_stepping::input_vectors::InputVector, types::Real};

#[derive(Clone)]
pub struct JetVector<T> {
    pub items: Vec<T>,
    pub r: usize,
    pub n: usize,
}

impl<T: Real> JetVector<T> {
    pub fn new(r: usize, nodes: usize, time: T) -> Self {
        Self {
            items: vec![time; r * nodes],
            r,
            n: nodes,
        }
    }
}

impl<T: Real> InputVector<T> for JetVector<T> {
    fn r(&self) -> usize {
        self.r
    }
    fn n(&self) -> usize {
        self.n
    }

    fn get_items(&self) -> &[T] {
        &self.items
    }
    fn get_items_mut(&mut self) -> &mut [T] {
        &mut self.items
    }
}
