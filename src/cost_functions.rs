pub fn mean_square_error(lhs: &Vec<f64>, rhs: &Vec<f64>) -> f64 {
    let item_count = lhs.len();
    assert_eq!(item_count, rhs.len());

    let mse = lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>() / (item_count as f64);

    mse
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_test() {
        let v1 = &vec![7.0f64, 12.0f64, 25.0f64, -2.0f64];
        let v2 = &vec![3.0f64, 9.0f64, 24.0f64, -1.0f64];

        let actual = mean_square_error(v1, v2);
        let expected = 6.75f64;

        assert_eq!(actual, expected);
    }
}
