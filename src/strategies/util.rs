use super::super::interface::*;

use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
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

// Return a unique id for humans for this move.
pub(super) fn move_id<G: Game>(s: &mut <G as Game>::S, m: Option<<G as Game>::M>) -> String
where
    <G as Game>::S: Zobrist,
{
    if let Some(mov) = m {
        if let Some(notation) = mov.notation(s) {
            notation
        } else {
            mov.apply(s);
            let id = format!("{:06x}", s.zobrist_hash() & 0xffffff);
            mov.undo(s);
            id
        }
    } else {
        "none".to_string()
    }
}

pub(super) fn pv_string<G: Game>(path: &[<G as Game>::M], s: &mut <G as Game>::S) -> String
where
    <G as Game>::S: Zobrist,
    <G as Game>::M: Copy,
{
    let mut out = String::new();
    for (i, m) in (0..).zip(path.iter()) {
        if i > 0 {
            out.push_str("; ");
        }
        out.push_str(move_id::<G>(s, Some(*m)).as_str());
        m.apply(s);
    }
    for m in path.iter().rev() {
        m.undo(s);
    }
    out
}

// This exists to be wrapped in a mutex, because it didn't work when I tried a tuple.
pub(super) struct ValueMove<M> {
    pub(super) value: Evaluation,
    pub(super) m: M,
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

// An insert-only lock-free Option<Box<T>>
pub(super) struct AtomicBox<T>(AtomicPtr<T>);

impl<T> Default for AtomicBox<T> {
    fn default() -> Self {
        Self(AtomicPtr::default())
    }
}

impl<T> AtomicBox<T> {
    // Tries to set the AtomicBox to this value if empty.
    // Returns a reference to whatever is in the box.
    pub(super) fn try_set(&self, value: Box<T>) -> &T {
        let ptr = Box::into_raw(value);
        // Try to replace nullptr with the value.
        let ret_ptr = if let Err(new_ptr) =
            self.0.compare_exchange(std::ptr::null_mut(), ptr, Ordering::SeqCst, Ordering::SeqCst)
        {
            // If someone beat us to it, return the original drop the new one.
            unsafe { drop(Box::from_raw(ptr)) };
            new_ptr
        } else {
            ptr
        };
        unsafe { ret_ptr.as_ref().unwrap() }
    }

    pub(super) fn get(&self) -> Option<&T> {
        let ptr = self.0.load(Ordering::Relaxed);
        unsafe { ptr.as_ref() }
    }
}

impl<T> Drop for AtomicBox<T> {
    fn drop(&mut self) {
        let ptr = *self.0.get_mut();
        if !ptr.is_null() {
            unsafe { drop(Box::from_raw(ptr)) };
        }
    }
}

#[test]
fn test_atomic_box() {
    let b = AtomicBox::<u32>::default();
    assert_eq!(None, b.get());
    b.try_set(Box::new(3));
    assert_eq!(Some(&3), b.get());
    b.try_set(Box::new(4));
    assert_eq!(Some(&3), b.get());
}
