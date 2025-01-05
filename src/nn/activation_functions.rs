use std::f64::consts::E;

pub struct Activation {
    pub function: fn(&f64) -> f64,
    pub derivative: fn(&f64) -> f64
}

pub const RELU: Activation = Activation {
    function: |x| if *x <= 0f64 { 0f64 } else { *x },
    derivative: |x| if *x <= 0f64 { 0f64 } else { 1f64 }
};

pub const SIGMOID: Activation = Activation {
    function: |x| 1f64 / (1f64 - E.powf(-(*x))),
    derivative: |x| (*x) * (1f64 - (*x))
};