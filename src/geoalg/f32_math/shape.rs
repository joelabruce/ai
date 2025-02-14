#[derive(Debug, Clone)]
pub struct Shape {
    dimensions: Vec<usize>,

    // Calculated on creation, and only privately accessible
    strides: Vec<usize>,
    size: usize,
}

impl Shape {
    /// Creates a 1-dimensional shape.
    pub fn d1(columns: usize) -> Self { Self::new(vec![columns]) }

    /// Creates a 2-dimensional shape (row-major).
    pub fn d2(rows: usize, columns: usize) -> Self { Self::new(vec![rows, columns]) } 

    /// Generalized shape creation
    pub fn new(dimensions: Vec<usize>) -> Self {
        let (strides, size) = Shape::compute_strides(&dimensions);

        Self { dimensions, strides, size }
    }

    /// Helper function to pre-compute strides and size of shape.
    fn compute_strides(dimensions: &Vec<usize>) -> (Vec<usize>, usize) {
        let mut strides = vec![0; dimensions.len()];
        let mut stride = 1;
        for axis in (0..dimensions.len()).rev() {
            strides[axis] = stride;
            stride *= dimensions[axis];
        }

        (strides, stride)
    }

    /// Gets total size of shape.
    pub fn size(&self) -> usize { self.size }

    /// Get length of dimension.
    /// Might consider renaming.
    pub fn axis_len(&self, axis: usize) -> usize {
        assert!(axis < self.dimensions.len(), "Cannot get axis_len for non-existant axis.");
        self.dimensions[axis]
    }

    /// Gets offset for specific axis.
    pub fn stride_for(&self, axis: usize) -> usize {
        assert!(axis < self.dimensions.len(), "Cannot get stride_for for non-existant axis.");
        self.strides[axis]
    }

    /// Useful for constructing slice start and end.
    pub fn index_at(&self, coordinate: &Shape) -> usize {
        assert_eq!(self.dimensions.len(), coordinate.dimensions.len(), "Coordinate outside of shape bounds.");
        let mut index = 0;
        for axis in 0..self.dimensions.len() {            
            assert!(self.axis_len(axis) > coordinate.dimensions[axis]);
            index += self.stride_for(axis) * coordinate.dimensions[axis];
        }

        index
    }
}

#[cfg(test)]
mod tests {
    use super::Shape;

    #[test]
    fn test_axis() {
        let shape = Shape::new(vec![5, 4, 3, 2]);

        let actual = shape.axis_len(0);
        assert_eq!(actual, 5);

        let actual = shape.axis_len(1);
        assert_eq!(actual, 4);

        let actual = shape.axis_len(2);
        assert_eq!(actual, 3);

        let actual = shape.axis_len(3);
        assert_eq!(actual, 2);
    }

    #[test]

    fn test_stride() {
        let shape = Shape::new(vec![5, 4, 3, 2]);

        let actual = shape.stride_for(3);
        assert_eq!(actual, 1);

        let actual = shape.stride_for(2);
        assert_eq!(actual, 2);

        let actual = shape.stride_for(1);
        assert_eq!(actual, 6);

        let actual = shape.stride_for(0);
        assert_eq!(actual, 24);
    }

    #[test]
    fn test_index_at() {
        let shape = Shape::new(vec![100, 30, 14, 500]);

        let coordinate = Shape::new(vec![1, 4, 7, 8]);
        let actual = shape.index_at(&coordinate);

        // 1* 8 + 500* 7 + 500*14* 4 + 500*14*30* 1 = 241508
        assert_eq!(actual, 241508);
    }
}