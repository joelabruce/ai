use std::f32::consts::E;

pub struct Activation {
    pub function: fn(&f32) -> f32,
    pub derivative: fn(&f32) -> f32
}

pub const RELU: Activation = Activation {
    function: |x| if *x <= 0f32 { 0f32 } else { *x },
    derivative: |x| if *x <= 0f32 { 0f32 } else { 1f32 }
};

pub const SIGMOID: Activation = Activation {
    function: |x| 1f32 / (1f32 - E.powf(-(*x))),
    derivative: |x| (*x) * (1f32 - (*x))
};