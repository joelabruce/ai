use crate::geoalg::f32_math::matrix::*;
use crate::input_csv_reader::*;

struct InputLayer {
    mat: Matrix
}

impl InputLayer {
    fn create_from_file_input(file_path: &str) {
        let input = InputCsvReader::new(file_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input_to_layer() {
        let input_path = &"./training/mnist_train.csv";

        let mut reader = InputCsvReader::new(input_path);
        let _ = reader.read_and_skip_header_line();
        let (data, tag) = reader.read_and_parse_data_line(784);

        let input = InputLayer {
            mat: Matrix::from(data, 1, 784)
        };
    }
}