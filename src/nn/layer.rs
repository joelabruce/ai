use matrix::Matrix;

use crate::geoalg::f64_math::*;

pub struct InputLayer_ {

}

impl InputLayer_ {
    pub fn forward<'a>(&self, layer: &'a mut Layer) -> &'a Layer {
        layer
    }
}

pub struct Layer {
    // pub neurons: usize,
    // pub weights: Matrix,
    // pub biases: Matrix
}

impl Layer {
    pub fn forward<'a>(&self, activation: &'a mut Activation) -> &'a Activation {
        activation
    }

    pub fn backward<'a>(&self, activation: &'a mut Activation) -> &'a Activation {
        activation 
    }
}

pub struct Activation {

}

impl Activation {
    pub fn forward<'a>(&self, layer: &'a mut Layer) -> &'a Layer {
        layer
    }

    pub fn backward<'a>(&self, layer: &'a mut Layer) -> &'a Layer {
        layer
    }

    pub fn loss(&self, loss: &Loss) -> Loss {
        Loss {}
    }
}

pub struct Loss {
    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chaining() {
        let input_layer = InputLayer_ {};
        let mut hidden1 = Layer {};
        let mut activation1 = Activation {};
        let mut hidden2 = Layer {};
        let mut activation2 = Activation {};
        let loss = Loss {};

        let x = input_layer
            .forward(&mut hidden1)
            .forward(&mut activation1)
            .forward(&mut hidden2)
            .forward(&mut activation2);
    }
}