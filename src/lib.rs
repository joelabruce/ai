pub mod geoalg;
pub mod partitions;

use crate::partitions::*;

//use std::io;
//use rand::Rng;
use colored::*;

fn _main() {
    //println!("n: ");
    // let _secret_number =  rand::thread_rng().gen_range(1..=100);
    
    //let mut prompt_response = String::new();
    //io::stdin()
    //    .read_line(&mut prompt_response)
    //    .expect("Oh nos! Problem with entering text from keyboard!");

    //let n: u128 = prompt_response.trim().parse().expect("Unsigned integer");

    for n in 1..=500 {
        //let p = strict_partitions_n_into_3_fast(n);
        let _p3 = strict_partitions_n_into_3_fast(n);
        let p4 = strict_partitions_n_into_4_recursive(n);
        let p4e = strict_partitions_n_into_4_experimental(n);
        let d = p4e-p4;
        //let p5 = strict_partitions_n_into_5_recursive(n);

        let msg = format!("n:{n:3} p4:{p4:6} p4e:{p4e:6} d:{d:3}").white();
        let msg = match n % 2 {
            0 => msg.on_black(),
            _ => msg.on_bright_black()
        };

        let msg = match n % 3 {
            0 => msg.bright_green(),
            _ => msg.white()
        };

        print!("{msg} ");
        if n % 4 == 0 {
            println!();
        }
    }
}

