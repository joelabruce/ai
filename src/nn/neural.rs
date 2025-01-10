use crate::nn::layers::*;
use crate::nn::activation_functions::*;
use crate::geoalg::f64_math::matrix::*;
use crate::input_csv_reader::*;

pub struct Neural {

}


pub enum Node {
    HiddenLayer(HiddenLayer),
    Activation(Activation)
}

impl Neural {
    pub fn import_inputs(file_path: &str) -> (Vec<Vec<f64>>, usize, Matrix) {
        // Open file for reading to create training data.
        let mut reader = InputCsvReader::new(file_path);
        let _ = reader.read_and_skip_header_line();     // Discard header

        let mut target_ohes = vec![];         // One-hot encoding of the targets
        let mut normalized_inputs = vec![];     // Normalized data 
        let input_count = 100;

        for _sample in 0..input_count {
            let (v, label) = reader.read_and_parse_data_line(784);
            normalized_inputs.push(v);
            target_ohes.extend(one_hot_encode(label, 10));
        }

        let targets = Matrix {
            rows: input_count,
            columns: 10,
            values: target_ohes
        };

        (normalized_inputs, input_count, targets)
    }

    pub fn forward(il: &InputLayer, nodes: &Vec<Node>) -> Vec<Matrix> {
        let mut forward_stack = Vec::with_capacity(nodes.len() + 1);

        let inputs = forward_stack.push(il.forward());
        for node in nodes.iter() {
            match node {
                Node::Activation(n) => forward_stack.push(n.forward(forward_stack.last().unwrap())),
                Node::HiddenLayer(n) => forward_stack.push(n.forward(forward_stack.last().unwrap()))
            }
        } 

        forward_stack
    }

    pub fn backward(nodes: &mut Vec<Node>, dz: &Matrix, fcalcs: &mut Vec<Matrix>) {
        let dvalues = dz;
        
        for node in nodes.iter_mut().rev() {
            let dvalues = match node {
                Node::Activation(n) => {
                    let fcalc = fcalcs.pop().unwrap();
                    n.backward(&dvalues, &fcalc)
                },
                Node::HiddenLayer(n) => {
                    let fcalc = fcalcs.pop().unwrap();

                    println!("unwrapped fcalc!");
                    n.backward(&dvalues, &fcalc)
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "Needs mnist_train.csv to train on."]
    #[test]
    fn test() {
        // Read from specified file for creating neural network
        let (normalized_inputs, input_count, targets) = Neural::import_inputs("./training/mnist_train.csv");

        // Create Layers in network
        let il = InputLayer::from_vec(normalized_inputs);
        assert_eq!(il.values.rows, input_count);
        assert_eq!(il.values.columns, 784);
        assert_eq!(il.values.get_element_count(), 784 * input_count);

        // Hidden layer 1
        let mut dense1 = il.new_hidden_layer(128);
        assert_eq!(dense1.weights.rows, 784);
        assert_eq!(dense1.weights.columns, 128);
        assert_eq!(dense1.weights.get_element_count(), 100352);

        // Hidden layer 2
        let mut dense2 = dense1.new_hidden_layer(64);
        assert_eq!(dense2.weights.rows, 128);
        assert_eq!(dense2.weights.columns, 64);
        assert_eq!(dense2.weights.get_element_count(), 8192);

        // Output layer
        let mut dense3 = dense2.new_hidden_layer(10);

        let mut nodes: Vec<Node> = Vec::new();
        // nodes.push(Node::HiddenLayer(dense1));
        // nodes.push(Node::Activation(RELU));
        // nodes.push(Node::HiddenLayer(dense2));
        // nodes.push(Node::Activation(RELU));
        // nodes.push(Node::HiddenLayer(dense3)); 

        // Begin training
        for epoch in 0..400 {
            // Sanity check to make sure the output shapes are correct.
            // Forward propagation should not be able to mutate anything, only backpropagation can do that.
            // let inputs = il.forward();                                  // input layer -> 
            // let fcalc1 = dense1.forward(&inputs);                       // dense1 ->
            // let fcalc2 = RELU.forward(&fcalc1);                 // RELU ->
            // let fcalc3 = dense2.forward(&fcalc2);               // dense2 -> 
            // let fcalc4 = RELU.forward(&fcalc3);                 // RELU ->
            // let fcalc5 = dense3.forward(&fcalc4);               // dense3
            // let predictions = (SOFTMAX.f)(&fcalc5);                     // softmax ->

            let mut forward_stack = Vec::new();//Neural::forward(&il, &nodes);// Vec::new();
            forward_stack.push(il.forward());
            forward_stack.push(dense1.forward(forward_stack.last().unwrap()));
            forward_stack.push(RELU.forward(forward_stack.last().unwrap()));
            forward_stack.push(dense2.forward(forward_stack.last().unwrap()));
            forward_stack.push(RELU.forward(forward_stack.last().unwrap()));
            forward_stack.push(dense3.forward(forward_stack.last().unwrap()));

            let predictions = &(SOFTMAX.f)(&forward_stack.pop().unwrap());
            //println!("Predictions: {:?}", predictions);
            assert_eq!(predictions.columns, 10);
            assert_eq!(predictions.rows, input_count);

            // Used to calculate accuracy and if progress is being made.
            // Not being used yet.
            let sample_losses = forward_categorical_cross_entropy_loss(&predictions, &targets);
            let data_loss = sample_losses.values.iter().copied().sum::<f64>() / sample_losses.get_element_count() as f64;
            //let x = argmax(predictions);
            
            if epoch % 10 == 0 || epoch < 100 {
                println!("Epoch: {epoch} | Data Loss: {data_loss}");
            }

            // Start backpropagating.
            let dvalues6 = backward_categorical_cross_entropy_loss_wrt_softmax(&predictions, &targets); // loss ->
            let dvalues6 = dvalues6.div_by_scalar(input_count as f64);   // Don't forget to scale by batch size

            // Can use simple gradient descent now! Woohoo!
            // Will implement adam optimization later. Whew
            // let dvalues5 = dense3.backward(&dvalues6, &fcalc4);
            // let dvalues4 = RELU.backward(&dvalues5, &fcalc3);
            // let dvalues3 = dense2.backward(&dvalues4, &fcalc2);
            // let dvalues2 = RELU.backward(&dvalues3, &fcalc1);
            // let _dvalues1 = dense1.backward(&dvalues2, &inputs);

            //let _ = Neural::backward(&mut nodes, &dvalues6, &mut forward_stack);

            let dvalues5 = dense3.backward(&dvalues6, &forward_stack.pop().unwrap());
            let dvalues4 = RELU.backward(&dvalues5, &forward_stack.pop().unwrap());
            let dvalues3 = dense2.backward(&dvalues4, &forward_stack.pop().unwrap());
            let dvalues2 = RELU.backward(&dvalues3, &forward_stack.pop().unwrap());
            let dvalues1 = dense1.backward(&dvalues2, &forward_stack.pop().unwrap());
        }
    }
}