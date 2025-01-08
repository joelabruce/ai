use std::fs::File;
use std::io::{self, prelude::*, BufReader};

pub struct InputCsvReader {
    reader: BufReader<File>
}

impl InputCsvReader {
    pub fn new(file_path: &str) -> InputCsvReader {
        let file = File::open(file_path).unwrap();
        let reader = BufReader::new(file);

        InputCsvReader {
            reader
        }
    }

    pub fn read_and_skip_header_line(&mut self) -> io::Result<()>{
        let mut line = String::new();
        let _ = self.reader.read_line(&mut line);
        Ok(())
    }

    pub fn read_and_parse_data_line(&mut self, vec_size: usize) -> (Vec<f64>, f64) {
        let mut result_vector = Vec::with_capacity(vec_size);
        let mut tag = 0f64;
        let mut line = String::new();

        let _len = self.reader.read_line(&mut line);
        let pre_processed  = line.split(",");

        for (i, element) in pre_processed.enumerate() {
            let retrieved_value = element.trim();

            if i == 0 {
                tag = retrieved_value.parse::<f64>().unwrap();
            }
            else if i > vec_size {
                println!("Line contained more data than expected, possible error");
            }
            else {
                let float_value = retrieved_value.parse::<f64>();
                let value_to_push = match float_value {
                    Ok(r) => r / 255.0f64,
                    Err(e) => { // This should only error for header.
                        println!("ERROR!!!!!!!!!!!!!!!!!! {e} -> {retrieved_value} (will assume 0.0f64)");
                        0.0f64
                    }
                };

                result_vector.push(value_to_push);
            }
        }

        (result_vector, tag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut reader = InputCsvReader::new("./tests/test.csv");
        let _ = reader.read_and_skip_header_line();

        let (v, tag) = reader.read_and_parse_data_line(784);
        assert_eq!(v.len(), 784);
        assert_eq!(tag, 7.0);

        let (v, tag) = reader.read_and_parse_data_line(784);
        assert_eq!(v.len(), 784);
        assert_eq!(tag, 2.0);
    }
}