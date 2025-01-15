/// Calculates the Kronecker Delta given i and j that are equatable to eachother.
/// # Arguments
/// # Returns
pub fn kronecker_delta_f64<I:PartialEq>(i: I, j: I) -> f64 {
    if i == j { 1.0 } else { 0.0 } 
}
