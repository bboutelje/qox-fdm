pub trait InitialConditions<T> {
    fn get_value(self, x: T) -> T;
}
