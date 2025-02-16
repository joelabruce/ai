use std::ops::Index;

#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    dimensions: Vec<usize>,

    // Calculated on creation, and only privately accessible
    strides: Vec<usize>,
    size: usize,
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output { &self.dimensions[index] }
}

impl Shape {
    /// Creates a 1-dimensional shape.
    pub fn d1(columns: usize) -> Self { Self::new(vec![columns]) }

    /// Creates a 2-dimensional shape (row-major).
    pub fn d2(rows: usize, columns: usize) -> Self { Self::new(vec![rows, columns]) } 

    /// Creates a 3-dimensional shape.
    pub fn d3(i: usize, j: usize, k: usize) -> Self { Self::new(vec![i, j, k]) }

    /// Creates a 4-dimensional shape.
    pub fn d4(i: usize, j: usize, k: usize, l: usize) -> Self { Self::new(vec![i, j, k, l]) }

    /// Generalized shape creation.
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

    /// Gets number of dimensions in shape.
    pub fn len(&self) -> usize { self.dimensions.len() }

    /// Gets offset for specific axis.
    pub fn stride_for(&self, axis: usize) -> usize {
        assert!(axis < self.dimensions.len(), "Cannot get stride_for for non-existant axis.");
        self.strides[axis]
    }

    /// Useful for constructing slice start and end.
    pub fn index_at(&self, coordinate: &Shape) -> usize {
        assert_eq!(self.dimensions.len(), coordinate.dimensions.len(), "Coordinate does not have same dimensionality as shape.");
        let mut index = 0;
        for axis in 0..self.dimensions.len() {            
            assert!(self[axis] > coordinate.dimensions[axis], "Coordinate outside of shape bounds.");
            index += self.stride_for(axis) * coordinate.dimensions[axis];
        }

        index
    }

    /// Grabs a slice of the dimensions
    pub fn dims(&self) -> &[usize] { &self.dimensions }
}

#[cfg(test)]
mod tests {
    use super::Shape;

    #[test]
    fn test_d4() {
        let shape = Shape::d4(5, 4, 3, 2);

        let actual = shape[0];
        assert_eq!(actual, 5);

        let actual = shape[1];
        assert_eq!(actual, 4);

        let actual = shape[2];
        assert_eq!(actual, 3);

        let actual = shape[3];
        assert_eq!(actual, 2);
    }

    #[test]

    fn test_stride_and_size() {
        let shape = Shape::new(vec![5, 4, 3, 2]);

        let actual = shape.stride_for(3);
        assert_eq!(actual, 1);

        let actual = shape.stride_for(2);
        assert_eq!(actual, 2);

        let actual = shape.stride_for(1);
        assert_eq!(actual, 6);

        let actual = shape.stride_for(0);
        assert_eq!(actual, 24);

        let actual = shape.size;
        assert_eq!(actual, 120);
    }

    #[test]
    #[should_panic]
    fn test_invalid_stride_for() {
        let shape = Shape::d1(3);
        shape.stride_for(1);
    }

    #[test]
    #[should_panic]
    fn test_invalid_dimension() {
        let shape = Shape::d2(3, 7);
        shape[3];
    }

    #[test]
    fn test_index_at() {
        let shape = Shape::new(vec![100, 30, 14, 500]);

        let coordinate = Shape::new(vec![1, 4, 7, 8]);
        let actual = shape.index_at(&coordinate);

        // 1* 8 + 500* 7 + 500*14* 4 + 500*14*30* 1 = 241508
        assert_eq!(actual, 241508);
    }

    #[test]
    #[should_panic]
    fn test_out_of_bounds_index_at() {
        let shape = Shape::d2(3, 5);
        let coordinate = Shape::d1(19);

        shape.index_at(&coordinate);
    }

    #[test]
    #[should_panic]
    fn test_invalid_index_at() {
        let shape = Shape::d2(3, 5);
        let coordinate = Shape::d2(1,9);

        shape.index_at(&coordinate);
    }    
}