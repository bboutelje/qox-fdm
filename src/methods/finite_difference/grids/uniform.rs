use crate::types::Real;

use crate::methods::{
    finite_difference::grids::{Grid1d, fitted::FittedGrid1d},
    transforms::Transform,
};

#[derive(Clone)]
pub struct UniformGrid1d<T, Tr> {
    pub transform: Tr,
    pub centers: Vec<T>,
    pub h_plus: Vec<T>,
    pub h_minus: Vec<T>,
    pub locations: Vec<T>,
}

impl<T: Real, Tr: Transform<T>> UniformGrid1d<T, Tr> {
    pub fn new(start: T, end: T, size: usize, transform: Tr) -> Self {
        let n_minus_1 = T::from_f64((size - 1) as f64);
        let dx = (end - start) / n_minus_1;

        let centers: Vec<T> = (0..size)
            .map(|i| start + (T::from_f64(i as f64) * dx))
            .collect();

        let locations: Vec<T> = centers
            .iter()
            .map(|&c| transform.to_physical(c)) // Assumes Transform trait has a 'map' method
            .collect();

        let (h_plus, h_minus) = Self::build_distances(&centers);

        Self {
            transform,
            centers,
            h_plus,
            h_minus,
            locations,
        }
    }

    fn build_distances(centers: &[T]) -> (Vec<T>, Vec<T>) {
        let n = centers.len();
        let mut hp = vec![T::zero(); n];
        let mut hm = vec![T::zero(); n];
        for (i, window) in centers.windows(2).enumerate() {
            let diff = window[1] - window[0];
            hp[i] = diff;
            hm[i + 1] = diff;
        }
        (hp, hm)
    }
}

impl<T: Real, Tr: Transform<T> + Copy> Grid1d<T, Tr> for UniformGrid1d<T, Tr> {
    fn centers(&self) -> &[T] {
        &self.centers
    }

    fn h_plus(&self) -> &[T] {
        &self.h_plus
    }
    fn h_minus(&self) -> &[T] {
        &self.h_minus
    }

    fn size(&self) -> usize {
        self.centers.len()
    }

    // Restore the mapping here
    fn location(&self, index: usize) -> T {
        self.locations[index]
        // self.transform.to_physical(self.centers[index])
    }

    fn apply_physical_jump(&self, jump: T) -> FittedGrid1d<T, Tr> {
        let new_locations: Vec<T> = self.locations.iter().map(|&s| s + jump).collect();

        FittedGrid1d::from_physical(new_locations, self.transform)
    }

    fn get_transform(&self) -> Tr {
        self.transform
    }
}
