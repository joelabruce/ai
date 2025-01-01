use std::ops::Mul;

use rand::distributions::{Distribution, Uniform};

/// Calculates the Kronecker Delta given i and j that are equatable to eachother.
/// # Arguments
/// # Returns
pub fn kronecker_delta_f32<I:Eq>(i: I, j: I) -> f32 {
    if i == j { 1.0 } else { 0.0 } 
}

/// Used to create an identity matrix of size n.
/// # Arguments
/// # Returns
pub struct IdentityMatrixIterator {
    i: u64,
    n: u64
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

/// Row-major matrix.
#[derive(PartialEq)]
#[derive(Debug)]
pub struct Matrix {
    cols: u64,
    rows: u64,
    pub values: Vec<f32>
}

impl Matrix {
    pub fn new_zeroed(i: u64, j: u64) -> Self {
        assert!(i > 0);
        assert!(j > 0);

        let capacity = i * j;
        let values = (1..=capacity).map(|_| 0.0f32).collect();

        Self {
            cols: i,
            rows: j,
            values
        }
    }

    pub fn new_identity(n: u64) -> Self {
        assert!(n > 0);

        let imi = IdentityMatrixIterator {
            i: 0,
            n
        };

        let values = imi.collect::<Vec<_>>();
        Self {
            cols: n,
            rows: n,
            values
        }
    }

    pub fn new_identity_alt(n: u64) -> Self {
        assert!(n > 0);

        let values = (0..n).map(|i| kronecker_delta_f32(i % (n + 1), 0)).collect();

        Self {
            cols: n,
            rows: n,
            values
        }
    }

    /// Returns an ixj matrix filled with random values between -1.0 and 1.0 inclusive.
    /// # Arguments
    /// # Returns
    pub fn new_randomized(i: u64, j: u64) -> Self {
        assert!(i > 0);
        assert!(j > 0);

        let step = Uniform::new_inclusive(-1.0f32, 1.0f32);
        let element_counts = i * j;
        let mut rng = rand::thread_rng();
        let values = step.sample_iter(&mut rng).take(element_counts as usize).collect();

        //let random_range = (1..=element_count).map(|_| )
        Self {
            cols: i,
            rows: j,
            values
        }
    }

    /// Returns index in vec given row and column.
    /// # Arguments
    /// # Returns
    pub fn index_for(&self, row: u64, column: u64) -> u64 {
        row * self.cols + column
    }

    /// Gets reference to value at specified row and column.
    /// # Arguments
    /// # Returns
    pub fn get(&self, row: u64, column: u64) -> Option<&f32> {
        self.values.get(self.index_for(row, column) as usize)
    }
}

impl Mul<&Matrix> for Matrix {
    type Output = Self;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        assert_eq!(self.cols, rhs.rows);

        let r_size = (rhs.cols * rhs.rows) as usize;

        let mut floats = Vec::with_capacity(r_size);

        for c in 0..self.rows {
            for b in 0..rhs.cols {
                let mut accumulator = 0f32;
                for a in 0..self.cols {
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
            cols: rhs.cols,
            rows: self.rows,
            values: floats
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            cols: 1,
            values: vec![1.0f32]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_2_alt() {
        let actual = Matrix::new_identity(2);
        let expected = Matrix {
            rows: 2,
            cols: 2,
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
            cols: 3,
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
            cols: 4,
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
        let mat = Matrix {
            cols: 3,
            rows: 4,
            values: Vec::with_capacity(12)
        };

        let mut expected: u64 = 0;
        for row in 0..mat.rows {
            for col in 0..mat.cols {
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
            cols: 4,
            values: vec![
                1.0f32, 2.0f32, 3.0f32, 4.0f32,
                -1.0f32, -2.0f32, -3.0f32, -4.0f32
            ]
        };

        let rhs = Matrix {
            rows: 4,
            cols: 2,
            values: vec![
                3.0f32, 8.0f32,
                1.0f32, 1.2f32,
                0.8f32, 1.9f32,
                5.0f32, 6.0f32
            ]
        };

        let expected = Matrix {
            rows: 2,
            cols: 2,
            values: vec! [
                27.4f32, 40.1f32,
                -27.4f32, -40.1f32
            ]
        };

        let actual = lhs * &rhs;

        assert!(actual == expected);        
    }
}