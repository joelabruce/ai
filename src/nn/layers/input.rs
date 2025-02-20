use crate::geoalg::f32_math::{shape::Shape, tensor::Tensor};

pub enum InputTypes {
    Flattened { features: usize },
    Image{ height: usize, width: usize, channels: usize}
}


/// Allows for creation of succeeding layers based on initial data passed in.
#[derive(Clone)]
pub struct Input {
    pub input_tensor: Tensor
}

impl Input {
    pub fn load(input_type: &InputTypes, batch_size: usize, values: Vec<f32>) -> Self {
        match input_type {
            InputTypes::Flattened { features } => {
                Self { input_tensor: Tensor::matrix(batch_size, *features, values) }
            },
            InputTypes::Image { height, width, channels } => {
                Self { input_tensor: Tensor::new(Shape::d4(batch_size, *height,*width, *channels), values) }
            }
        }
    }
}