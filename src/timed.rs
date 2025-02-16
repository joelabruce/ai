use std::time::Instant;

pub struct TimedContext {
    start: Instant
}

impl TimedContext {
    pub fn checkpoint(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }
}

pub fn timed(f: impl Fn() -> ()) -> f32 {
    let start = Instant::now();
    f();
    let duration = start.elapsed();

    duration.as_secs_f32()
}

pub fn timed_with_context(f: impl Fn(TimedContext)) -> f32 {
    let start = Instant::now();
    f(TimedContext { start });
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
        let timed = timed_with_context(|context| {
            thread::sleep(time::Duration::from_millis(200));
            let time_elapsed = context.checkpoint();
            println!("First checkpoint at {time_elapsed}!");

            thread::sleep(time::Duration::from_millis(100));
        });

        assert!(timed >= 0.3);
    }
}