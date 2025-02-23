use std::time::Instant;

pub struct TimedContext {
    checkpoints: Vec<Instant>
}

impl TimedContext {
    pub fn new() -> Self {
        let start = Instant::now();
        Self { checkpoints: vec![start] }
    }

    pub fn checkpoint(&mut self) -> f32 {
        let new_checkpoint = Instant::now();
        let &last_checkpoint = self.checkpoints.last().unwrap();
        self.checkpoints.push(new_checkpoint);

        new_checkpoint.duration_since(last_checkpoint).as_secs_f32()
    }
}

pub fn timed(mut f: impl FnMut() -> ()) -> f32 {
    let start = Instant::now();
    f();
    let duration = start.elapsed();

    duration.as_secs_f32()
}

pub fn timed_with_context(f: impl Fn(&mut TimedContext)) -> f32 {
    let start = Instant::now();
    f(&mut TimedContext::new());
    start.elapsed().as_secs_f32()
}

#[cfg(test)]
mod tests {
    use std::{thread, time};

    use super::{timed, timed_with_context};

    #[test]
    fn test_timed() {
        let timed = timed(|| {
            thread::sleep(time::Duration::from_millis(500));
        });

        assert!(timed >= 0.5);
    }

    #[test]
    fn test_timed_with_context() {
        let total_time = timed_with_context(|context| {
            thread::sleep(time::Duration::from_millis(200));
            let time_elapsed = context.checkpoint();
            println!("First checkpoint at {time_elapsed}!");

            thread::sleep(time::Duration::from_millis(100));
            let time_elapsed = context.checkpoint();
            print!("Second checkpoint at {time_elapsed}!")
        });

        assert!(total_time >= 0.3);
    }
}