use std::fs::File;
use std::io::Write;

pub struct OutputBinWriter {
    //writer: BufWriter<File>
    file: File
}

impl OutputBinWriter {
    pub fn new(file_path: &str) -> OutputBinWriter {
        let file = File::create(file_path).expect("File should be able to be created to save the neural network model.");
        //let mut writer = BufWriter::new(file);

        OutputBinWriter {
            file
        }
    }

    pub fn write_meta_legible(&mut self, data: &str) {
        self.file.write(data.as_bytes()).unwrap();
        self.file.flush().expect("Could not flush to file");
    }

    pub fn write_slice_f64(&mut self, data: &[f64]) {
        for &value in data {
            self.file.write_all(&value.to_le_bytes()).expect("Should be able to write file, please check permissions.");
            self.file.flush().expect("Could not flush to file, please check permissions.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut writer = OutputBinWriter::new("./tests/test_model.nn");
        writer.write_meta_legible("test meta");
        
        let vec = vec![1.3; 8];
        writer.write_slice_f64(&vec);
    }
}