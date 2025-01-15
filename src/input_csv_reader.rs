use std::fs::File;
use std::io::{self, prelude::*, BufReader};

use crate::digit_image::*;

pub struct InputCsvReader {
    reader: BufReader<File>
}

impl InputCsvReader {
    /// Opens file and creates reader.
    pub fn new(file_path: &str) -> InputCsvReader {
        let file = File::open(file_path).expect("File should exist for training and testing network.");
        let reader = BufReader::new(file);

        InputCsvReader {
            reader
        }
    }

    /// Reads and skips the header line.
    pub fn read_and_skip_header_line(&mut self) -> io::Result<()>{
        let mut line = String::new();
        let _ = self.reader.read_line(&mut line);
        Ok(())
    }

    /// Reads a single line after 
    pub fn read_and_parse_data_line(&mut self, vec_size: usize) -> DigitImage {
        let mut pixels = Vec::with_capacity(vec_size);
        let mut label = 0f64;
        let mut line = String::new();

        let _len = self.reader.read_line(&mut line);
        let pre_processed  = line.split(",");

        for (i, element) in pre_processed.enumerate() {
            let retrieved_value = element.trim();

            if i == 0 {
                label = retrieved_value.parse::<f64>()
                    .expect(format!("Retrieved value should have been a tag, found: [{retrieved_value}].").as_str());
            }
            else if i > vec_size {
                println!("Line contained more data than expected, possible error");
            }
            else {
                let float_value = retrieved_value.parse::<f64>();
                let value_to_push = match float_value {
                    Ok(r) => r / 255.0,
                    Err(e) => { // This should only error for header.
                        println!("ERROR!!!!!!!!!!!!!!!!!! {e} -> {retrieved_value} (will assume 0.0f64)");
                        0.0f64
                    }
                };

                pixels.push(value_to_push);
            }
        }

        DigitImage {
            label,
            pixels
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut reader = InputCsvReader::new("./tests/test.csv");
        let _ = reader.read_and_skip_header_line();

        let digit_image = reader.read_and_parse_data_line(784);
        assert_eq!(digit_image.pixels.len(), 784);
        assert_eq!(digit_image.label, 7.0);

        let digit_image = reader.read_and_parse_data_line(784);
        assert_eq!(digit_image.pixels.len(), 784);
        assert_eq!(digit_image.label, 2.0);
    }
}