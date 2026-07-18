pub trait ParabolicPde1d<T> {
    fn a(&self, u: T, t: T) -> T;
    fn b(&self, u: T, t: T) -> T;
    fn c(&self, u: T, t: T) -> T;
}
