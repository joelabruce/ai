use std::{fmt::Display, ops::*};

#[derive(PartialEq)]
#[derive(Debug)]
pub struct Complexf(pub f32, pub f32);

impl Display for Complexf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} + {}i)", self.0, self.1)
    } 
}

/// Calculates sum of two complex numbers.
/// # Arguments
/// # Returns
impl Add<Complexf> for Complexf {
    type Output = Self;

    fn add(self, rhs: Complexf) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

/// Calculates difference of two Complex numbers.
/// # Arguments
/// # Returns
impl Sub<Complexf> for Complexf {
    type Output = Self;

    fn sub(self, rhs: Complexf) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}

/// Calculates product of two Complex numbers.
/// # Arguments
/// # Returns
impl Mul<Complexf> for Complexf {
    type Output = Self;

    fn mul(self, rhs: Complexf) -> Self::Output {
        Self(self.0*rhs.0 - self.1*rhs.1, self.0*rhs.1 + self.1*rhs.0)
    }
}


mod tests {
    use std::thread::AccessError;
    
    use super::*;

    #[test]
    fn display_complexf() {
        let v = Complexf(3.3, -2.7);
        let actual = format!("{v}");
        let expected: &str = "(3.3 + -2.7i)";
        assert_eq!(actual, expected, "expected: {expected}, actual: {actual}");
    }

    #[test]
    fn adding_complexf() {
        let v1 = Complexf(1.0, 2.0);
        let v2 = Complexf(10.0, 23.0);
        let actual = v1 + v2;
        let expected = Complexf(11.0, 25.0);

        assert_eq!(actual, expected, "expected: {expected}, actual: {actual}");
    }

    #[test]
    fn subtracting_complexf() {
        let v1 = Complexf(28.0, 13.0);
        let v2 = Complexf(5.0, 7.0);
        let actual = v1 - v2;
        let expected = Complexf(23.0, 6.0);

        assert_eq!(actual, expected, "expected: {expected}, actual: {actual}");
    }

    #[test]
    fn multiplying_complexf() {
        let v1 = Complexf(2.0, 3.0);
        let v2 = Complexf(10.0, 20.0);
        let actual = v1*v2;
        let expected = Complexf(-40.0, 70.0);

        assert_eq!(actual, expected, "expected: {expected}, actual: {actual}");
    }
}