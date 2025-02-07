use std::time::Instant;

pub struct Checkpoint {
    pub tag: String,
    pub at: Instant
}

pub struct  Checkpoints {
    pub checkpoints: Vec<Checkpoint>
}

impl Checkpoints {
    pub fn tag(&mut self, tag: String) {
        self.checkpoints.push(Checkpoint {
            tag,
            at: Instant::now()
        })
    }
}

pub fn timed(f: impl Fn() -> ()) -> f32 {
    let start = Instant::now();
    f();
    let duration = start.elapsed();

    duration.as_secs_f32()
}

pub fn timed_checkpointed(f: impl Fn(Checkpoints)) {

}