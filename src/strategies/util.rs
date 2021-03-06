use super::super::interface::*;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{sleep, spawn};
use std::time::Duration;

// For values near winning and losing values, push them slightly closer to zero.
// A win in 3 moves (BEST-3) will be chosen over a win in 5 moves (BEST-5).
// A loss in 5 moves (WORST+5) will be chosen over a loss in 3 moves (WORST+3).
pub(super) fn clamp_value(value: Evaluation) -> Evaluation {
    if value > BEST_EVAL - 100 {
        value - 1
    } else if value < WORST_EVAL + 100 {
        value + 1
    } else {
        value
    }
}

// Undo any value clamping.
pub(super) fn unclamp_value(value: Evaluation) -> Evaluation {
    if value > BEST_EVAL - 100 {
        BEST_EVAL
    } else if value < WORST_EVAL + 100 {
        WORST_EVAL
    } else {
        value
    }
}

pub(super) fn timeout_signal(dur: Duration) -> Arc<AtomicBool> {
    // Theoretically we could include an async runtime to do this and use
    // fewer threads, but the stdlib implementation is only a few lines...
    let signal = Arc::new(AtomicBool::new(false));
    let signal2 = signal.clone();
    spawn(move || {
        sleep(dur);
        signal2.store(true, Ordering::Relaxed);
    });
    signal
}

// This exists to be wrapped in a mutex, because it didn't work when I tried a tuple.'
pub(super) struct ValueMove<M> {
    value: Evaluation,
    m: M,
}

impl<M> ValueMove<M> {
    pub(super) fn new(value: Evaluation, m: M) -> Self {
        Self { value, m }
    }

    pub(super) fn max(&mut self, value: Evaluation, m: M) {
        if value > self.value {
            self.value = value;
            self.m = m;
        }
    }

    pub(super) fn into_inner(self) -> (Evaluation, M) {
        (self.value, self.m)
    }
}
