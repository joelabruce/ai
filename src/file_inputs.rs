use std::fs::File;
use std::io::{self, prelude::*, BufReader};
use std::num::ParseFloatError;

struct InputCsvReader {
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

    pub fn read_and_parse_data_line(&mut self) -> Vec<f32> {
        let mut line = String::new();
        let _len = self.reader.read_line(&mut line);
        
        let x: Vec<_> = line.split(",").map(|s| {
            let retrieved_value = s.trim();
            let x = retrieved_value.parse::<f32>();
            match x {
                Ok(r) => r,
                Err(e) => { // This should only error for header, will fix soon.
                    println!("ERROR!!!!!!!!!!!!!!!!!! {e} -> {retrieved_value} (will assume 0.0f32)");
                    0.0f32
                }
            }
        }).collect();

        x
    }
}

#[cfg(test)]
mod tests {
    use std::fs::read;

    use super::*;

    #[test]
    fn test() {
        // This test shouldfail if you do not have comparable csv file to test.
        // The data is larger than 100 MB so could not store in repo.
        let mut reader = InputCsvReader::new("./tests/test.csv");
        let _ = reader.read_and_skip_header_line();
        let v = reader.read_and_parse_data_line();

        assert_eq!(v.len(), 785);
    }
}