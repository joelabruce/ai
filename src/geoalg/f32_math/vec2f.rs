use std::{fmt::Display, ops::*};

#[derive(PartialEq)]
#[derive(Debug)]
pub struct Vec2f(pub f32, pub f32);

impl Display for Vec2f {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

/// Calculates sum of two 2d vectors.
/// # Arguments
/// # Returns
impl Add<Vec2f> for Vec2f {
    type Output = Self;
    
    fn add(self, rhs: Vec2f) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

/// Calculates the diference of two 2d vectors.
/// # Arguments
/// # Returns
impl Sub<Vec2f> for Vec2f {
    type Output =  Self;

    fn sub(self, rhs: Vec2f) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}

/// Calculates Outer Product of two 2d vectors.
/// # Arguments
/// # Returns
// impl BitXor<Vec2f> for Vec2f {
//     type Output = Self;

//     fn bitxor(self, rhs: Self) -> Self::Output {
//         self
//     }
// }

/// Calculates Inner (Dot) Product of two 2d vectors
/// # Arguments
/// # Returns
impl Mul<f32> for Vec2f {
    type Output = f32;

    fn mul(self, rhs: f32) -> Self::Output {
        rhs
    }
}

mod tests {
    use std::thread::AccessError;

    use super::*;

    #[test]
    fn display_vec2f() {
        let v = Vec2f(3.3, -2.7);
        let actual = format!("{v}");
        let expected: &str = "(3.3, -2.7)";
        assert_eq!(actual, expected, "expected: {expected}, actual: {actual}");
    }

    #[test]
    fn adding_vec2f() {
        let v1 = Vec2f(1.0, 2.0);
        let v2 = Vec2f(10.0, 23.0);
        let actual = v1 + v2;
        let expected = Vec2f(11.0, 25.0);

        assert_eq!(actual, expected, "expected: {expected}, actual: {actual}");
    }

    #[test]
    fn subtracting() {
        let v1 = Vec2f(28.0, 13.0);
        let v2 = Vec2f(5.0, 7.0);
        let actual = v1 - v2;
        let expected = Vec2f(23.0, 6.0);

        assert_eq!(actual, expected, "expected: {expected}, actual: {actual}");
    }
}