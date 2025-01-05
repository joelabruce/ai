use std::ops::{Add, Mul};

use rand::{distributions::{Distribution, Uniform}, Error};

/// Calculates the Kronecker Delta given i and j that are equatable to eachother.
/// # Arguments
/// # Returns
pub fn kronecker_delta_f32<I:Eq>(i: I, j: I) -> f32 {
    if i == j { 1.0 } else { 0.0 } 
}

/// Used to create an identity matrix of size n.
pub struct IdentityMatrixIterator {
    i: usize,
    n: usize
}

impl Iterator for IdentityMatrixIterator {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.i += 1;

        let p = self.i % (self.n + 1);
        let r = kronecker_delta_f32(p, 1);
        let n2 = self.n * self.n;

        if self.i <= n2 { Some(r) } else { None }
    }
}

/// Matrix is implemented as a single dimensional vector of f32s.
/// This implementation of Matrix is row-major. 
/// Row-major is specified so certain optimizations and parallelization can be performed.
/// Column-major is not yet implemented.
#[derive(PartialEq, Debug, Clone)]
pub struct Matrix {
    columns: usize,
    rows: usize,
    pub values: Vec<f32>
}

impl Matrix {
    /// Create a matrix from a vector.
    /// # Arguments
    /// # Returns
    pub fn from(values: Vec<f32>, rows: usize, columns: usize) -> Self {
        assert_eq!(rows * columns, values.len());
        
        Matrix {
            rows,
            columns,
            values
        }
    }

    /// Creates a matrix with columns * size elements where every element is zero
    /// # Arguments
    /// # Returns
    pub fn new_zeroed(columns: usize, rows: usize) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let capacity = rows * columns;
        let values = vec![0.0f32; capacity];

        Self {
            columns,
            rows,
            values: values
        }
    }

    /// Constructor for defining an identity matrix with n dimensions.
    /// # Arguments
    /// # Returns
    pub fn new_identity(n: usize) -> Self {
        assert!(n > 0);

        let imi = IdentityMatrixIterator {
            i: 0,
            n
        };

        let values = imi.collect::<Vec<_>>();
        Self {
            columns: n,
            rows: n,
            values
        }
    }

    /// Possibly faster way to implement an identity matrix.
    pub fn new_identity_alt(n: usize) -> Self {
        assert!(n > 0);

        let values = (0..n).map(|i| kronecker_delta_f32(i % (n + 1), 0)).collect();

        Self {
            columns: n,
            rows: n,
            values
        }
    }

    /// Returns an ixj matrix filled with random values between -1.0 and 1.0 inclusive.
    /// # Arguments
    /// # Returns
    pub fn new_randomized(columns: usize, rows: usize) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let step = Uniform::new_inclusive(-1.0f32, 1.0f32);
        let element_counts = columns * rows;
        let mut rng = rand::thread_rng();
        let values = step.sample_iter(&mut rng).take(element_counts).collect();

        Self {
            columns,
            rows,
            values,
        }
    }

    /// Returns index in vec given row and column.
    /// # Arguments
    /// # Returns
    pub fn index_for(&self, row: usize, column: usize) -> usize {
        assert!(row < self.rows);
        assert!(column < self.columns);

        row * self.columns + column
    }

    /// Gets reference to value at specified row and column.
    /// # Arguments
    /// # Returns
    pub fn get(&self, row: usize, column: usize) -> Option<&f32> {
        self.values.get(self.index_for(row, column))
    }

    /// Returns slice of matrix that is a row of the matrix
    /// # Arguments
    /// # Returns
    pub fn get_row_vector_slice(&self, row: usize) -> &[f32] {
        assert!(row < self.rows);

        let start = row * self.columns;
        let end = start + self.columns;
        &self.values[start..end]
    }

    /// Returns a newly allocated matrix that is the transpose of the matrix operated on.
    /// # Arguments
    /// # Returns
    pub fn get_transpose(&self) -> Matrix {
        let capacity = self.rows * self.columns;
        let mut transposed = Vec::with_capacity(capacity);

        for i in 0..capacity {
            let index_to_push = self.columns * (i % self.rows) + i / self.rows;

            // Debug code to print that the transpose calcs are workinf correctly
            //if i % self.rows == 0 { println!() }
            //print!("{index_to_push} ");

            transposed.push(self.values[index_to_push]);
        }

        Matrix {
            columns: self.rows,
            rows: self.columns,
            values: transposed
        }
    }

    /// Multiplies two matrices using transpose operation for efficiency.
    /// # Arguments
    /// # Returns
    pub fn mul_using_transpose(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.columns, rhs.rows);
        let r_size = rhs.columns * self.rows;
        let mut floats = Vec::with_capacity(r_size);

        let t = rhs.get_transpose();

        for row in 0..self.rows {
            let ls = self.get_row_vector_slice(row);
            for t_row in 0..t.rows {
                let rs = t.get_row_vector_slice(t_row);

                let x = dot_product_of_vector_slices(ls, rs);
                floats.push(x);
            }
        }

        Matrix {
            columns: rhs.columns,
            rows: self.rows,
            values: floats
        }        
    }

    /// Returns size of underlying vector.
    pub fn get_element_count(&self) -> usize {
        self.columns * self.rows
    }

    // Getting column vectors proving to be tricky, 
    //  perhaps abandon for now and focus on transposing and only using slices for matrix rows since matrix is row-major?
    // fn column_vector<'a>(&'a mut self, column: usize) -> &[f32] {
    //    self.values.iter().skip(column).step_by(self.columns).cloned().collect()
    // }
}

/// Dot product of two Vec<f32> slices. Will always assume they are same length (not production ready).
/// How can this be effectively benchmarked and optimized?
/// # Arguments
/// # Returns
pub fn dot_product_of_vector_slices(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());
    let n = lhs.len();

    let (x, y) = (&lhs[..n], &rhs[..n]);
    let mut sum = 0f32;
    for i in 0..n {
        sum += x[i] * y[i];
    }

    sum
}

impl Add<&Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Self::Output {
        assert_eq!(self.columns, rhs.columns);
        assert_eq!(self.rows, rhs.rows);

        let values = self.values.iter().zip(rhs.values.iter()).map(|(x, y)| x + y).collect();

        Matrix {
            columns: rhs.columns,
            rows: rhs.rows,
            values
        }
    }
}

impl Mul<&Matrix> for Matrix {
    type Output = Self;

    /// Naive implementation of matrix multiplication.
    /// Does not take advantage of any optimizations.
    /// # Arguments
    /// # Returns
    fn mul(self, rhs: &Matrix) -> Self::Output {
        assert_eq!(self.columns, rhs.rows);

        let r_size = rhs.columns * rhs.rows;

        let mut floats = Vec::with_capacity(r_size);

        for c in 0..self.rows {
            for b in 0..rhs.columns {
                let mut accumulator = 0f32;
                for a in 0..self.columns {
                    let x = match self.get(c, a) {
                        Some(r) => r,
                        None => panic!("Out of bounds for matrix lhs")
                    };
                    let y = match rhs.get(a, b) {
                        Some(r)  => r,
                        None => panic!("Out of bounds for matrix rhs")
                        
                    };
                    let v = x * y;
                    accumulator += v;
                }
                floats.push(accumulator);
            }
        }
        
        Matrix {
            columns: rhs.columns,
            rows: self.rows,
            values: floats
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_add() {
        let lhs = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                1f32, 2f32, 3f32,
                4f32, 5f32, 6f32,
                7f32, 8f32, 9f32
            ]
        };

        let rhs = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                -1f32, -2f32, -3f32,
                -4f32, -5f32, -6f32,
                -7f32, -8f32, -9f32
            ]
        };

        let actual = lhs + &rhs;

        let expected = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                0f32, 0f32, 0f32,
                0f32, 0f32, 0f32,
                0f32, 0f32, 0f32
            ]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn transpose_test() {
        let m = Matrix {
            rows: 5,
            columns: 4,
            values: vec![
                0f32, 1f32, 2f32, 3f32,
                4f32, 5f32, 6f32, 7f32,
                8f32, 9f32, 10f32, 11f32,
                12f32, 13f32, 14f32, 15f32,
                16f32, 17f32, 18f32, 19f32
            ]
        };

        let expected = Matrix {
            rows: 4,
            columns: 5,
            values: vec![
                0f32, 4f32, 8f32, 12f32, 16f32,
                1f32, 5f32, 9f32, 13f32, 17f32,
                2f32, 6f32, 10f32, 14f32, 18f32,
                3f32, 7f32, 11f32, 15f32, 19f32
            ]
        };
        
        let actual = m.get_transpose();
        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_1() {
        let imi = IdentityMatrixIterator { i: 0, n: 1 };

        let actual = imi.collect::<Vec<_>>();
        let expected = vec![1.0f32];

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_2() {
        let imi = IdentityMatrixIterator { i: 0, n: 2 };

        let actual = imi.collect::<Vec<_>>();
        let expected = vec![
            1.0f32, 0.0f32,
            0.0f32, 1.0f32];

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_3() {
        let imi = IdentityMatrixIterator { i: 0, n: 3 };

        let actual = imi.collect::<Vec<_>>();
        let expected = vec![
            1.0f32, 0.0f32, 0.0f32, 
            0.0f32, 1.0f32, 0.0f32, 
            0.0f32, 0.0f32, 1.0f32];

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_4() {
        let imi = IdentityMatrixIterator { i: 0, n: 4 };

        let actual = imi.collect::<Vec<_>>();
        let expected = vec![
            1.0f32, 0.0f32, 0.0f32, 0.0f32,
            0.0f32, 1.0f32, 0.0f32, 0.0f32,
            0.0f32, 0.0f32, 1.0f32, 0.0f32,
            0.0f32, 0.0f32, 0.0f32, 1.0f32
            ];

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_1_alt() {
        let actual = Matrix::new_identity(1);
        let expected = Matrix{
            rows: 1,
            columns: 1,
            values: vec![1.0f32]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_2_alt() {
        let actual = Matrix::new_identity(2);
        let expected = Matrix {
            rows: 2,
            columns: 2,
            values: vec![
                1.0f32, 0.0f32,
                0.0f32, 1.0f32]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_3_alt() {
        let actual = Matrix::new_identity(3);
        let expected = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                1.0f32, 0.0f32, 0.0f32, 
                0.0f32, 1.0f32, 0.0f32, 
                0.0f32, 0.0f32, 1.0f32]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_4_alt() {
        let actual = Matrix::new_identity(4);
        let expected = Matrix {
            rows: 4,
            columns: 4,
            values: vec![
                1.0f32, 0.0f32, 0.0f32, 0.0f32,
                0.0f32, 1.0f32, 0.0f32, 0.0f32,
                0.0f32, 0.0f32, 1.0f32, 0.0f32,
                0.0f32, 0.0f32, 0.0f32, 1.0f32]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn random_matrix() {
        let m28x28 = Matrix::new_randomized(28, 28);

        let r = m28x28.values.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");

        println!("{r}");
    }

    #[test]
    fn matrix_index() {
        let mat = Matrix::new_randomized(4, 7);

        let mut expected: usize = 0;
        for row in 0..mat.rows {
            for col in 0..mat.columns {
                let actual = mat.index_for(row, col);
                assert_eq!(actual, expected);
                expected += 1;
            }
        }
    }

    #[test]
    fn matrix_mult() {
        let lhs = Matrix {
            rows: 2,
            columns: 4,
            values: vec![
                1.0f32, 2.0f32, 3.0f32, 4.0f32,
                -1.0f32, -2.0f32, -3.0f32, -4.0f32
            ]
        };

        let rhs = Matrix {
            rows: 4,
            columns: 2,
            values: vec![
                3.0f32, 8.0f32,
                1.0f32, 1.2f32,
                0.8f32, 1.9f32,
                5.0f32, 6.0f32
            ]
        };

        let expected = Matrix {
            rows: 2,
            columns: 2,
            values: vec! [
                27.4f32, 40.1f32,
                -27.4f32, -40.1f32
            ]
        };

        let actual = lhs * &rhs;

        assert!(actual == expected);        
    }

    #[test]
    fn matrix_mult_with_transpose() {
        let lhs = Matrix {
            rows: 4,
            columns: 3,
            values: vec![
                1f32, 2f32, 3f32,
                4f32, 5f32, 6f32,
                7f32, 8f32, 9f32,
                10f32, 11f32, 12f32]
        };

        let rhs = Matrix {
            rows: 3,
            columns: 5,
            values: vec![
                1f32, 2f32, 3f32, 4f32, 5f32,
                6f32, 7f32, 8f32, 9f32, 10f32,
                11f32, 12f32, 13f32, 14f32, 15f32
            ]
        };

        // Resultant matrix needs to have aas many rows as lhs, and as many columns as rhs.
        let expected = Matrix {
            rows: 4,
            columns: 5,
            values: vec! [
                46f32, 52f32, 58f32, 64f32, 70f32,
                100f32, 115f32, 130f32, 145f32, 160f32,
                154f32, 178f32, 202f32, 226f32, 250f32,
                208f32, 241f32, 274f32, 307f32, 340f32
            ]
        };

        let actual = Matrix::mul_using_transpose(&lhs, &rhs);

        assert!(actual == expected);        
    }
}
