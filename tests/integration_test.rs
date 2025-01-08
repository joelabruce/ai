use ai::nn::network::*;
use ai::input_csv_reader::*;
use ai::geoalg::f64_math::matrix::*;
use ai::nn::activation_functions::*;

fn one_hot_encode(digit: f64) -> Vec<f64> {
    let mut result = vec![0.0f64; 10];
    result[digit as usize] = 1.0f64;

    println!("{:?}", result);

    result
}

#[ignore = "reason"]
#[test]
fn test_nn() {
    let mut reader = InputCsvReader::new("./training/mnist_train.csv");
    let _ = reader.read_and_skip_header_line();

    let mut inputs = vec![];
    let mut targets = vec![];

    for _sample in 0..400 {
        let (v, tag) = reader.read_and_parse_data_line(784);
        inputs.push(v);
        targets.push(one_hot_encode(tag));
    }

    let mut network = Network::new(vec![784, 128, 10], SIGMOID, 0.001f64);
    network.train(inputs, targets, 15);


    // Reload training data and see if it correctly predicts what it was trained on.
    let mut reader = InputCsvReader::new("./training/mnist_train.csv");
    let _ = reader.read_and_skip_header_line();

    for _ in 0..6 {
        let (v, tag) = reader.read_and_parse_data_line(784);
        let prediction = network.feed_forward(Matrix::from(v));

        print!("expected: {tag}, ");
        println!("{:?}", prediction);

        //println!("predicted: {prediction}, actual: {tag}");
    }
}