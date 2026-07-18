use crate::types::Real;

use crate::methods::{finite_difference::grids::Grid1d, transforms::Transform};

#[derive(Clone)]
pub struct FittedGrid1d<T, Tr> {
    pub transform: Tr,
    pub centers: Vec<T>,
    pub h_plus: Vec<T>,
    pub h_minus: Vec<T>,
    pub locations: Vec<T>,
}

impl<T: Real, Tr: Transform<T>> FittedGrid1d<T, Tr> {
    pub fn new(centers: Vec<T>, transform: Tr) -> Self {
        // Calculate physical locations based on the provided non-uniform centers
        let locations: Vec<T> = centers.iter().map(|&c| transform.to_physical(c)).collect();

        // reuse your logic for calculating distances between points
        let (h_plus, h_minus) = Self::build_distances(&centers);

        Self {
            transform,
            centers,
            h_plus,
            h_minus,
            locations,
        }
    }

    pub fn from_physical(locations: Vec<T>, transform: Tr) -> Self {
        // Map physical -> mesh space (centers)
        let centers: Vec<T> = locations
            .iter()
            .map(|&s| transform.to_transform(s))
            .collect();

        // Calculate distances in mesh space
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
            hp[i] = diff; // Distance to the next point
            hm[i + 1] = diff; // Distance from the previous point
        }
        (hp, hm)
    }
}

impl<T: Real, Tr: Transform<T> + Copy> Grid1d<T, Tr> for FittedGrid1d<T, Tr> {
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

    fn location(&self, index: usize) -> T {
        self.locations[index]
    }

    fn apply_physical_jump(&self, jump: T) -> FittedGrid1d<T, Tr> {
        let new_locations: Vec<T> = self.locations.iter().map(|&s| s + jump).collect();

        FittedGrid1d::from_physical(new_locations, self.transform)
    }

    fn get_transform(&self) -> Tr {
        self.transform
    }
}
