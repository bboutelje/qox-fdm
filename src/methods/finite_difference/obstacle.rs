pub trait Obstacle<T> {
    fn get_value(self, x: T) -> T;
}
