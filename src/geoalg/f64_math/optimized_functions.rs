const ROWS_DIFFEREMT_LENGTHS: &str = "Cannot take dot product of two &[f64] of unequal length.";

/// Dot product of two Vec<f64> slices. Will always assume they are same length (not production ready).
/// How can this be effectively benchmarked and optimized?
pub fn dot_product_of_vector_slices(lhs: &[f64], rhs: &[f64]) -> f64 {
    assert_eq!(lhs.len(), rhs.len(), "{}", ROWS_DIFFEREMT_LENGTHS);
    let n = lhs.len();

    let mut sum = 0f64;
    for i in 0..n {
        sum += lhs[i] * rhs[i];
    }

    sum
}

/// Calculates the Kronecker Delta given i and j that are equatable to eachother.
pub fn kronecker_delta_f64<I:PartialEq>(i: I, j: I) -> f64 {
    if i == j { 1.0 } else { 0.0 } 
}

/// Argmax. Returns index and value at the index.
pub fn vector_row_max(values: &[f64]) -> (usize, f64) {
    let mut max = values[0];
    let mut index = 0;
    for i in 1..values.len() {
        if values[i] > max { 
            max = values[i];
            index = i;
        }
    }

    (index, max)
}


/// Caclulates strict partitions of n into 3 distinct parts
/// # Arguments
/// # Returns
pub fn strict_partitions_n_into_3_fastest(n: i128) -> i128 {
    (n*n + 3)/12
}

/// Calculates strict partitions of n into 3 distinct parts using n choose
/// # Arguments
/// # Returns
pub fn strict_partitions_n_into_3_fast(n: i128) -> i128 {
    let n_choose = ((n - 1) * (n - 2)) / 2;
    let floor_term = 3 * ((n - 1) / 2);
    let adjustment = if n % 3 == 0 { 2 } else { 0 };

    (n_choose - floor_term + adjustment) / 6 
}

/// Attempting to find a simple closed form version of strict partitions of n into 4 distinct parts
/// # Arguments
/// # Returns
pub fn strict_partitions_n_into_4_experimental(n: i128) -> i128 {
    // really close, but gets 10th term incorrect
//     let n_choose = ((n - 4) * (n - 5) * (n - 6)) / 6;
//     let floor_term = 0;//4 * ((n - 2) / 6);
//     let adjustment = 0;//if n % 4 == 0 { 6 } else { 0 }; 
//     (n_choose - floor_term + adjustment) / 24
    let o = n - 4;
    (o*o*o - strict_partitions_n_into_3_fastest(n))/144 //- strict_partitions_n_into_3_fastest(n)
}

/// Gives exact result of strict partitions of n into 4 parts using a loop
/// # Arguments
/// # Returns
pub fn strict_partitions_n_into_4_recursive(n: i128) -> i128 {
    let terms = (n - 4) / 4;
    let offset = n % 4;

    let mut accumulator = 0;
    for i in 1..=terms {
        accumulator += strict_partitions_n_into_3_fastest(4*i + offset);
    }

    accumulator
}

/// Gives exact result of strict partitions of n into 5 parts using a loop
/// # Arguments
/// # Returns
fn _strict_partitions_n_into_5_recursive(n: i128) -> i128 {
    let terms = (n - 5) / 5;
    let offset = n % 5;

    let mut accumulator = 0;
    for i in 1..=terms {
        accumulator += strict_partitions_n_into_4_recursive(5*i + offset);
    }

    accumulator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_row_max() {
        let tc = vec![2., 5., 7., 3.2, 1.7];
        let (i, m) = vector_row_max(&tc);

        assert_eq!(i, 2);
        assert_eq!(m, 7.);
    }

    #[test]
    fn test_kronecker() {
        let actual = kronecker_delta_f64(3, 0);
        assert_eq!(actual, 0.);

        let actual = kronecker_delta_f64(0, 0);
        assert_eq!(actual, 1.);

        let actual = kronecker_delta_f64(5., 5.);
        assert_eq!(actual, 1.);
    }

    #[test]
    fn test_dot_product_of_slices() {
        let lhs = &[1., 2., 3., 4.];
        let rhs = &[10., 20., 30., 40.];

        let actual = dot_product_of_vector_slices(lhs, rhs);
        let expected = 300.;

        assert_eq!(actual, expected);
    }

    #[test]
    #[should_panic]
    fn test_invalid_dot_product() {
        let lhs = &[1., 2.];
        let rhs = &[10., 20., 30.];

        dot_product_of_vector_slices(lhs, rhs);
    }
}