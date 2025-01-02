pub fn mean_square_error(lhs: &Vec<f32>, rhs: &Vec<f32>) -> f32 {
    let item_count = lhs.len();
    assert_eq!(item_count, rhs.len());

    let mse = lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / (item_count as f32);

    mse
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_test() {
        let v1 = &vec![7.0f32, 12.0f32, 25.0f32, -2.0f32];
        let v2 = &vec![3.0f32, 9.0f32, 24.0f32, -1.0f32];

        let actual = mean_square_error(v1, v2);
        let expected = 6.75f32;

        assert_eq!(actual, expected);
    }
}
