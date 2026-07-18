pub mod jet_vector;

pub trait InputVector<T> {
    fn r(&self) -> usize;

    fn n(&self) -> usize;
    fn get_items(&self) -> &[T];
    fn get_items_mut(&mut self) -> &mut [T];

    #[inline(always)]
    fn step_slice(&self, j: usize) -> &[T] {
        let start = j * self.n();
        &self.get_items()[start..start + self.n()]
    }

    #[inline(always)]
    fn step_slice_mut(&mut self, j: usize) -> &mut [T] {
        let start = j * self.n();
        let end = start + self.n();
        &mut self.get_items_mut()[start..end]
    }
}
