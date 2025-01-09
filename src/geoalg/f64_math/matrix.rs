use rand::distributions::{Distribution, Uniform};

/// Calculates the Kronecker Delta given i and j that are equatable to eachother.
/// # Arguments
/// # Returns
pub fn kronecker_delta_f64<I:PartialEq>(i: I, j: I) -> f64 {
    if i == j { 1.0 } else { 0.0 } 
}

pub fn one_hot_encode(label: f64, bounds: usize) -> Vec<f64> {
    (0..bounds)
        .map(|x| kronecker_delta_f64(label, x as f64))
        .collect()
}

/// Matrix is implemented as a single dimensional vector of f64s.
/// This implementation of Matrix is row-major. 
/// Row-major is specified so certain optimizations and parallelization can be performed.
/// Column-major is not yet implemented.
#[derive(PartialEq, Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f64>
}

impl Matrix {
    /// Create a matrix from a vector.
    /// # Arguments
    /// # Returns
    pub fn from_vec(values: Vec<f64>, rows: usize, columns: usize) -> Self {
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
    pub fn new_zeroed(rows: usize, columns: usize) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let capacity = rows * columns;
        let values = vec![0.0f64; capacity];

        Self {
            columns,
            rows,
            values: values
        }
    }

    /// Possibly faster way to implement an identity matrix.
    pub fn new_identity(n: usize) -> Self {
        assert!(n > 0);

        let values = (0..n*n).map(|i| kronecker_delta_f64(i % (n + 1), 0)).collect();

        Self {
            columns: n,
            rows: n,
            values
        }
    }

    /// Returns an ixj matrix filled with random values between -1.0 and 1.0 inclusive.
    /// # Arguments
    /// # Returns
    pub fn new_randomized_z(columns: usize, rows: usize) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let mut rng = rand::thread_rng();
        let step = Uniform::new_inclusive(-1.0, 1.0);
        let element_count = columns * rows;
        let values = step.sample_iter(&mut rng).take(element_count).collect();

        Self {
            columns,
            rows,
            values,
        }
    }

    /// Returns an ixj matrix filled with random values specified by uniform distribution.
    /// # Arguments
    /// # Returns
    pub fn new_randomized_uniform(rows: usize, columns: usize, uniform: Uniform<f64>) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let mut rng = rand::thread_rng();
        let step = uniform;// Uniform::new_inclusive(-0.15f64, 0.15f64);
        let element_count = columns * rows;
        let values = step.sample_iter(&mut rng).take(element_count).collect();

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
    pub fn get(&self, row: usize, column: usize) -> Option<&f64> {
        self.values.get(self.index_for(row, column))
    }

    /// Returns slice of matrix that is a row of the matrix
    /// # Arguments
    /// # Returns
    pub fn get_row_vector_slice(&self, row: usize) -> &[f64] {
        assert!(row < self.rows, "Tried to get a row that was out of bounds.");

        let start = row * self.columns;
        let end = start + self.columns;
        &self.values[start..end]
    }

    /// Given a specified row, return it transposed as a column.
    /// # Arguments
    /// # Returns
    pub fn get_row_transposed_as_column(&self, row: usize) -> Matrix {
        assert!(row < self.rows);

        let start = row * self.columns;
        let end = start + self.columns;

        Matrix {
            columns: 1,
            rows: self.columns,
            values: self.values[start..end].to_vec()
        }
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

    pub fn elementwise_multiply(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.columns, rhs.columns, "Columns must match for elementwise multiply.");
        assert_eq!(self.rows, rhs.rows, "Rows must match for elementwise multiply.");

        let values = self.values.iter()
            .zip(rhs.values.iter())
            .map(|(x, y)| x * y)
            .collect();

        Matrix {
            rows: self.rows,
            columns: self.columns,
            values
        }
    }

    /// Multiplies two matrices using transpose operation for efficiency.
    /// # Arguments
    /// # Returns
    pub fn mul(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.columns, rhs.rows, "When multiplying matrices, lhs columns must equal rhs rows.");

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

    /// Useful for applying an activation function to the entire matrix.
    /// *Allows to map capturing a variable outside of the closure.
    /// # Arguments
    /// # Returns
    pub fn map_with_capture(&self, func: impl Fn(&f64) -> f64) -> Matrix {
        let values = self.values.iter().map(|val| func(val)).collect();
        
        Matrix {
            rows: self.rows,
            columns: self.columns,
            values: values
        } 
    }

    /// Useful for applying an activation function to the entire matrix.
    /// # Arguments
    /// # Returns
    pub fn map(&self, func: fn(&f64) -> f64) -> Matrix {
        let values = self.values.iter().map(|&val| func(&val)).collect();
        
        Matrix {
            rows: self.rows,
            columns: self.columns,
            values: values
        } 
    }

    /// Elementwise difference of two matrices.
    /// # Arguments
    /// # Returns
    pub fn sub(&self, rhs: &Matrix) -> Matrix {
        assert!(self.columns == rhs.columns && self.rows == rhs.rows, "Cannot subtract matrices with different orders.");

        let values = self.values.iter().zip(rhs.values.iter()).map(|(x, y)| x - y).collect();

        Matrix {
            columns: rhs.columns,
            rows: rhs.rows,
            values
        }
    }

    /// Adds two matrices together. Efficient and easy because both matrices must have same order.
    pub fn add(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.columns, rhs.columns, "Columns of lhs and rhs must be equal when adding matrices.");
        assert_eq!(self.rows, rhs.rows, "Rows of lhs and rhs must be equal when adding matrices.");

        let values = self.values.iter().zip(rhs.values.iter()).map(|(x, y)| x + y).collect();

        Matrix {
            columns: rhs.columns,
            rows: rhs.rows,
            values
        }
    }

    /// Elementwise division of matrix by scalar.
    /// # Arguments
    /// # Returns
    pub fn div_by_scalar(&self, scalar: f64) -> Matrix {
        assert_ne!(scalar, 0.0, "Cannot divide matrix elements by zero.");
        let values = self.values.iter().map(|x| x / scalar).collect();

        Matrix {
            rows: self.rows,
            columns: self.columns,
            values
        }
    }

    /// Adds a Matrix of shape 1xn to every column. Each matrix must have same number of rows and rhs must have exactly 1 column.
    pub fn add_column_vector(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(rhs.columns, 1, "Rhs matrix must have 1 column.");
        assert_eq!(self.rows, rhs.rows, "Lhs and rhs must have equal number of rows.");

        let mut r = Vec::with_capacity(self.get_element_count());
        
        for row in 0..self.rows {
            let lhs_row = self.get_row_vector_slice(row);
            let new_row = lhs_row.iter().map(|f| f + rhs.values[row]);
            r.extend(new_row);
        }

        Matrix {
            columns: self.columns,
            rows: self.rows,
            values: r
        }
    }

    /// Adds a given row to each row in lhs matrix.
    pub fn add_row_vector(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(rhs.rows, 1, "Rhs matrix must have 1 row.");
        assert_eq!(self.columns, rhs.columns, "Lhs and rhs must have equal number of columns.");

        let mut r = Vec::with_capacity(self.get_element_count());
        for row in 0..self.rows{
            let x = self.get_row_vector_slice(row);
            let y = rhs.get_row_vector_slice(0);

            let xplusy = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect::<Vec<f64>>();

            r.extend(xplusy);
        }

        Matrix {
            rows: self.rows,
            columns: self.columns,
            values: r
        }
    }

    /// Sums each row in self, and outputs a new matrix that is 1 column.
    pub fn shrink_rows_by_add(&self) -> Matrix {
        let mut v = Vec::with_capacity(self.columns);

        let t = self.get_transpose();
        for row in 0..t.rows {
            let x = t.get_row_vector_slice(row).iter().sum();
            v.push(x);
        }

        Matrix {
            rows: 1,
            columns: self.columns,
            values: v
        }
    }

    /// Scales matrix by scalar.
    pub fn scale(&self, scalar: f64) -> Matrix {
        let values = self.values.iter().map(|x| x * scalar).collect();

        Matrix {
            rows: self.rows,
            columns: self.columns,
            values
        }
    }

    pub fn shape(&self) -> String {
        let rows = self.rows;
        let columns = self.columns;
        format!("{rows} x {columns}")
    }

    // Getting column vectors proving to be tricky, 
    //  perhaps abandon for now and focus on transposing and only using slices for matrix rows since matrix is row-major?
    // fn column_vector<'a>(&'a mut self, column: usize) -> &[f64] {
    //    self.values.iter().skip(column).step_by(self.columns).cloned().collect()
    // }
}

impl From<Vec<f64>> for Matrix {
    fn from(vec: Vec<f64>) -> Self {
        Matrix {
            rows: vec.len(),
            columns: 1,
            values: vec
        }
    }
}

/// Dot product of two Vec<f64> slices. Will always assume they are same length (not production ready).
/// How can this be effectively benchmarked and optimized?
/// # Arguments
/// # Returns
pub fn dot_product_of_vector_slices(lhs: &[f64], rhs: &[f64]) -> f64 {
    assert_eq!(lhs.len(), rhs.len());
    let n = lhs.len();

    let (x, y) = (&lhs[..n], &rhs[..n]);
    let mut sum = 0f64;
    for i in 0..n {
        sum += x[i] * y[i];
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_column() {
        let lhs = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                1f64, 2f64, 3f64,
                4f64, 5f64, 6f64,
                7f64, 8f64, 9f64
            ]
        };

        let rhs = Matrix {
            rows: 3,
            columns: 1,
            values: vec![
                10.0,
                20.0,
                30.0
            ]
        };

        let expected = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                11.0, 12.0, 13.0,
                24.0, 25.0, 26.0,
                37.0, 38.0, 39.0
            ]
        };

        let actual = lhs.add_column_vector(&rhs);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_matrix_add() {
        let lhs = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                1f64, 2f64, 3f64,
                4f64, 5f64, 6f64,
                7f64, 8f64, 9f64
            ]
        };

        let rhs = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                -1f64, -2f64, -3f64,
                -4f64, -5f64, -6f64,
                -7f64, -8f64, -9f64
            ]
        };

        let actual = lhs.add(&rhs);

        let expected = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                0f64, 0f64, 0f64,
                0f64, 0f64, 0f64,
                0f64, 0f64, 0f64
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
                0f64, 1f64, 2f64, 3f64,
                4f64, 5f64, 6f64, 7f64,
                8f64, 9f64, 10f64, 11f64,
                12f64, 13f64, 14f64, 15f64,
                16f64, 17f64, 18f64, 19f64
            ]
        };

        let expected = Matrix {
            rows: 4,
            columns: 5,
            values: vec![
                0f64, 4f64, 8f64, 12f64, 16f64,
                1f64, 5f64, 9f64, 13f64, 17f64,
                2f64, 6f64, 10f64, 14f64, 18f64,
                3f64, 7f64, 11f64, 15f64, 19f64
            ]
        };
        
        let actual = m.get_transpose();
        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_1() {
        let actual = Matrix::new_identity(1);
        let expected = Matrix{
            rows: 1,
            columns: 1,
            values: vec![1.0f64]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_2() {
        let actual = Matrix::new_identity(2);
        let expected = Matrix {
            rows: 2,
            columns: 2,
            values: vec![
                1.0f64, 0.0f64,
                0.0f64, 1.0f64]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_3() {
        let actual = Matrix::new_identity(3);
        let expected = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                1.0f64, 0.0f64, 0.0f64, 
                0.0f64, 1.0f64, 0.0f64, 
                0.0f64, 0.0f64, 1.0f64]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_4() {
        let actual = Matrix::new_identity(4);
        let expected = Matrix {
            rows: 4,
            columns: 4,
            values: vec![
                1.0f64, 0.0f64, 0.0f64, 0.0f64,
                0.0f64, 1.0f64, 0.0f64, 0.0f64,
                0.0f64, 0.0f64, 1.0f64, 0.0f64,
                0.0f64, 0.0f64, 0.0f64, 1.0f64]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn random_matrix() {
        let m28x28 = Matrix::new_randomized_z(28, 28);

        let _r =m28x28.values.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");

        //println!("{r}");
    }

    #[test]
    fn matrix_index() {
        let mat = Matrix::new_randomized_z(7, 4);

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
    fn matrix_mul() {
        let lhs = Matrix {
            rows: 4,
            columns: 3,
            values: vec![
                1f64, 2f64, 3f64,
                4f64, 5f64, 6f64,
                7f64, 8f64, 9f64,
                10f64, 11f64, 12f64]
        };

        let rhs = Matrix {
            rows: 3,
            columns: 5,
            values: vec![
                1f64, 2f64, 3f64, 4f64, 5f64,
                6f64, 7f64, 8f64, 9f64, 10f64,
                11f64, 12f64, 13f64, 14f64, 15f64
            ]
        };

        // Resultant matrix needs to have aas many rows as lhs, and as many columns as rhs.
        let expected = Matrix {
            rows: 4,
            columns: 5,
            values: vec! [
                46f64, 52f64, 58f64, 64f64, 70f64,
                100f64, 115f64, 130f64, 145f64, 160f64,
                154f64, 178f64, 202f64, 226f64, 250f64,
                208f64, 241f64, 274f64, 307f64, 340f64
            ]
        };

        let actual = Matrix::mul(&lhs, &rhs);

        assert!(actual == expected);        
    }
}
