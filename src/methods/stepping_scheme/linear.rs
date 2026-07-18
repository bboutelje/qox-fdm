use crate::methods::finite_difference::FdmProcess;
use crate::methods::finite_difference::boundary_conditions::BoundaryCondition;
use crate::methods::finite_difference::grids::Grid1d;
use crate::methods::finite_difference::intitial_conditions::InitialConditions;
use crate::methods::linear_operators::LinearOperator;
use crate::methods::stepping_scheme::SteppingScheme;
use crate::methods::time_stepping::GlmWorkspace;
use crate::methods::time_stepping::input_vectors::jet_vector::JetVector;
use crate::methods::{
    finite_difference::boundary_conditions::BoundarySide,
    linear_operators::tridiagonal_operator::TridiagonalOperator, transforms::Transform,
};
use crate::pdes::ParabolicPde1d;
use crate::{methods::finite_difference::grids::grid_type::GridType, types::Real};
use std::marker::PhantomData;

pub struct LinearScheme<T, Tr, O, P, Lbc, Ubc> {
    pub operator: TridiagonalOperator<T>,
    pub grid: GridType<T, Tr>,
    pub pde: P,
    pub lower_bc: Lbc,
    pub upper_bc: Ubc,
    _marker: PhantomData<(T, Tr, O, Lbc, Ubc)>,
}

impl<T, Tr, O, P, Lbc, Ubc> LinearScheme<T, Tr, O, P, Lbc, Ubc>
where
    T: Real,
    Tr: Transform<T> + Copy,
    P: ParabolicPde1d<T> + Copy,
    Lbc: BoundaryCondition<T, Tr, TridiagonalOperator<T>, P>,
    Ubc: BoundaryCondition<T, Tr, TridiagonalOperator<T>, P>,
{
    pub fn new(grid: GridType<T, Tr>, pde: P, lower_bc: Lbc, upper_bc: Ubc) -> Self {
        let mut operator = pde.build_operator(&grid, T::zero());

        lower_bc.apply(BoundarySide::Lower, &grid, &pde, T::zero(), &mut operator);
        upper_bc.apply(BoundarySide::Upper, &grid, &pde, T::zero(), &mut operator);
        Self {
            operator,
            grid,
            pde,
            lower_bc,
            upper_bc,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, Tr, O, P, Lbc, Ubc> SteppingScheme<T, Tr, O, P, Lbc, Ubc>
    for LinearScheme<T, Tr, O, P, Lbc, Ubc>
where
    T: Real,
    Tr: Transform<T> + Copy,
    P: FdmProcess<T, Tr, TridiagonalOperator<T>>,
    Lbc: BoundaryCondition<T, Tr, TridiagonalOperator<T>, P>,
    Ubc: BoundaryCondition<T, Tr, TridiagonalOperator<T>, P>,
{
    fn initialize_payoff<IC>(&self, initial_conditions: IC) -> JetVector<T>
    where
        IC: InitialConditions<T> + Copy,
    {
        let y: Vec<T> = (0..self.grid.size())
            .map(|i| {
                let s = self.grid.location(i);
                initial_conditions.get_value(s)
            })
            .collect();

        let mut dy_dt = vec![T::zero(); self.grid.size()];
        self.operator.apply_into(&y, &mut dy_dt);

        let mut items = y; // Start with the first n items
        items.extend(dy_dt); // Append the second n items (h * y')

        let vector = JetVector {
            items,
            r: 2,
            n: self.grid.size(),
        };
        vector
    }

    fn apply_jump_into(&mut self, external_vector: &mut JetVector<T>, amount: T) {
        let n = self.grid.size();

        self.grid = GridType::Fitted(self.grid.apply_physical_jump(amount));

        let mut new_op = self.pde.build_operator(&self.grid, T::zero());

        self.lower_bc.apply(
            BoundarySide::Lower,
            &self.grid,
            &self.pde,
            T::zero(),
            &mut new_op,
        );
        self.upper_bc.apply(
            BoundarySide::Upper,
            &self.grid,
            &self.pde,
            T::zero(),
            &mut new_op,
        );

        self.operator = new_op;

        let (y_slice, f_slice) = external_vector.items.split_at_mut(n);
        self.operator.apply_into(y_slice, f_slice);
    }

    fn operate_on_stage(&self, stage_slice: &[T], operated_on_stage_slice: &mut [T]) {
        self.operator
            .apply_into(stage_slice, operated_on_stage_slice);
    }

    fn solve_vector_into(&self, stage_idx: usize, coeff: T, workspace: &mut GlmWorkspace<T>) {
        let n = self.grid.size();
        let start = stage_idx * n;
        let end = (stage_idx + 1) * n;

        let current_time = T::zero();

        self.lower_bc.update_rhs(
            BoundarySide::Lower,
            &self.grid,
            &self.pde,
            current_time,
            &mut workspace.rhs_buffer,
        );

        self.upper_bc.update_rhs(
            BoundarySide::Upper,
            &self.grid,
            &self.pde,
            current_time,
            &mut workspace.rhs_buffer,
        );

        self.operator.solve_inverse_into(
            coeff,
            &workspace.rhs_buffer,
            &mut workspace.stages[start..end],
            &mut workspace.z_buffer,
        );

        self.operator.apply_into(
            &mut workspace.stages[start..end],
            &mut workspace.stage_derivatives[start..end],
        );
    }

    fn get_grid(&self) -> &GridType<T, Tr> {
        &self.grid
    }
}
