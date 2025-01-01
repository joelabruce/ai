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