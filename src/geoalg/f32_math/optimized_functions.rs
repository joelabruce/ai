const ROWS_DIFFEREMT_LENGTHS: &str = "Cannot take dot product of two &[f32] of unequal length.";

/// Dot product of two Vec<f32> slices. Will always assume they are same length (not production ready).
/// How can this be effectively benchmarked and optimized?
pub fn dot_product_of_vector_slices(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len(), "{}", ROWS_DIFFEREMT_LENGTHS);
    let n = lhs.len();

    let mut sum = 0f32;
    for i in 0..n {
        sum += lhs[i] * rhs[i];
    }

    sum
}

/// Calculates the Kronecker Delta given i and j that are equatable to eachother.
pub fn kronecker_delta_f32<I:PartialEq>(i: I, j: I) -> f32 {
    if i == j { 1.0 } else { 0.0 } 
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kronecker() {
        let actual = kronecker_delta_f32(3, 0);
        assert_eq!(actual, 0.);

        let actual = kronecker_delta_f32(0, 0);
        assert_eq!(actual, 1.);

        let actual = kronecker_delta_f32(5., 5.);
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