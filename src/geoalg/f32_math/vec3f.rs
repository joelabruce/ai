use std::{fmt::Display, ops::*};

#[derive(PartialEq)]
#[derive(Debug)]
pub struct Vec3f(pub f32, pub f32, pub f32);

impl Display for Vec3f {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.0, self.1, self.2)
    }
}

/// Calculates sum of two 3d vectors.
/// # Arguments
/// # Returns
impl Add<Vec3f> for Vec3f {
    type Output = Self;
    
    fn add(self, rhs: Vec3f) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

/// Calculates the diference of two 3d vectors.
/// # Arguments
/// # Returns
impl Sub<Vec3f> for Vec3f {
    type Output =  Self;

    fn sub(self, rhs: Vec3f) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

/// Calculates Outer Product of two 3d vectors.
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
impl Mul<f32> for Vec3f {
    type Output = f32;

    fn mul(self, rhs: f32) -> Self::Output {
        rhs
    }
}

mod tests {
    use std::thread::AccessError;

    use super::*;

    #[test]
    fn display_vec3f() {
        let v = Vec3f(3.3, -2.7, 6.1);
        let actual = format!("{v}");
        let expected: &str = "(3.3, -2.7, 6.1)";
        assert_eq!(actual, expected, "expected: {expected}, actual: {actual}");
    }

    #[test]
    fn adding_two_vec3f() {
        let v1 = Vec3f(1.0, 2.0, -5.0);
        let v2 = Vec3f(10.0, 23.0, 9.0);
        let actual = v1 + v2;
        let expected = Vec3f(11.0, 25.0, 4.0);

        assert_eq!(actual, expected, "expected: {expected}, actual: {actual}");
    }

    #[test]
    fn subtracting_two_vec3f() {
        let v1 = Vec3f(28.0, 13.0, 5.0);
        let v2 = Vec3f(5.0, 7.0, -3.0);
        let actual = v1 - v2;
        let expected = Vec3f(23.0, 6.0, 8.0);

        assert_eq!(actual, expected, "expected: {expected}, actual: {actual}");
    }
}