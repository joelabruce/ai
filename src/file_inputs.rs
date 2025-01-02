use std::fs::File;
use std::io::{self, prelude::*, BufReader};
use std::num::ParseFloatError;

    pub fn read(file_path: &str, line_index: usize) -> io::Result<Vec<f32>>{
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);

        let mut line = String::new();
        let len = reader.read_line(&mut line);
        
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

        assert_eq!(x.len(), 785);

        Ok(x)
    }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        read("./training/mnist_test.csv", 1);
    }
}