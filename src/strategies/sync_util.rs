use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::Arc;
use std::thread::{sleep, spawn};
use std::time::Duration;

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

// Safe API around lockfree threadlocals for rayon threadpools.
// Only safe when used from a single threadpool, as this crate does.
pub(super) struct ThreadLocal<T> {
    // Our owned reference to all the locals.
    locals: Vec<T>,
    // Mutable reference from which each thread finds its local.
    ptr: *mut T,
}

// Values are only accessed from their individual threads and references do not leak.
unsafe impl<T: Send> Send for ThreadLocal<T> {}
unsafe impl<T> Sync for ThreadLocal<T> {}

impl<T: Send> ThreadLocal<T> {
    pub(super) fn new<F: Fn() -> T>(f: F, pool: &rayon::ThreadPool) -> Self {
        let n = pool.current_num_threads();
        let mut locals = (0..n).map(|_| f()).collect::<Vec<_>>();
        let ptr = locals.as_mut_ptr();
        Self { locals, ptr }
    }

    pub(super) fn local_do<F: FnOnce(&mut T)>(&self, f: F) {
        // It would be nice to keep a handle to the threadpool to ensure this
        // thread is from only our pool, but the lifetimes seem too
        // restrictive.
        let index = rayon::current_thread_index().unwrap();
        assert!(index < self.locals.len());
        f(unsafe { self.ptr.add(index).as_mut().unwrap() });
    }

    // With a &mut self, no other threads can be using it.
    pub(super) fn do_all<F: FnMut(&mut T)>(&mut self, f: F) {
        self.locals.iter_mut().for_each(f);
    }
}

#[test]
fn test_threadlocal() {
    use rayon::prelude::*;
    let pool = rayon::ThreadPoolBuilder::new().build().unwrap();
    let mut tls = ThreadLocal::<u32>::new(|| 0, &pool);
    let count = 100000;
    (0..count).into_par_iter().for_each(|_| tls.local_do(|x| *x += 1));
    let mut sum = 0;
    tls.do_all(|x| sum += *x);
    assert_eq!(sum, count);

    let result = std::panic::catch_unwind(|| {
        // Invalid call from outside pool.
        tls.local_do(|x| *x += 1);
    });
    assert!(result.is_err());
}

// 64-bytes is a common cache line size.
#[repr(align(64))]
pub(super) struct CachePadded<T> {
    value: T,
}

impl<T: Default> Default for CachePadded<T> {
    fn default() -> Self {
        Self { value: T::default() }
    }
}

impl<T> Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> DerefMut for CachePadded<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}
