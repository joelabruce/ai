mod tests {
    use ai::output_bin_writer::OutputBinWriter;

   #[test]
    fn try_nn() {
        let mut writer = OutputBinWriter::new("./tests/test_model.nn");
        writer.write_meta_legible("test meta");

        let vec = vec![0.3; 3];
        writer.write_slice_f32(&vec);
    }
}