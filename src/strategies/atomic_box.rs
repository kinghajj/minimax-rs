use std::sync::atomic::{AtomicPtr, Ordering};

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
