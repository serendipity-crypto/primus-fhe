//! Observe secret storage immediately before its allocator releases it.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    ptr,
    sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering},
};

use primus_fft::Complex64;
use primus_glwe::{
    FourierGlweSecretKey, GlweSecretKey, GlweSize, NttGlweSecretKey, SecretKeyDistr,
};

struct ObservingAllocator;
static WATCHED: AtomicPtr<u8> = AtomicPtr::new(ptr::null_mut());
static LENGTH: AtomicUsize = AtomicUsize::new(0);
static ERASED: AtomicBool = AtomicBool::new(false);

// SAFETY: allocation and deallocation are delegated unchanged to System. The
// observer only reads a watched allocation before System releases its storage.
unsafe impl GlobalAlloc for ObservingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies the layout required by GlobalAlloc.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, data: *mut u8, layout: Layout) {
        if WATCHED
            .compare_exchange(data, ptr::null_mut(), Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let len = LENGTH.load(Ordering::SeqCst);
            let mut erased = len <= layout.size();
            for index in 0..len.min(layout.size()) {
                // SAFETY: data is still allocated, and the test only watches
                // initialized integer/complex values within this allocation.
                erased &= unsafe { ptr::read_volatile(data.add(index)) } == 0;
            }
            ERASED.store(erased, Ordering::SeqCst);
        }
        // SAFETY: data/layout are the original allocation passed by the caller.
        unsafe { System.dealloc(data, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: ObservingAllocator = ObservingAllocator;

fn assert_erased_on_drop<K>(key: K, data: *const u8, len: usize) {
    ERASED.store(false, Ordering::SeqCst);
    LENGTH.store(len, Ordering::SeqCst);
    WATCHED.store(data.cast_mut(), Ordering::SeqCst);
    drop(key);
    assert!(
        WATCHED.load(Ordering::SeqCst).is_null(),
        "secret allocation was not released"
    );
    assert!(
        ERASED.load(Ordering::SeqCst),
        "secret allocation was released without erasure"
    );
}

#[test]
fn all_secret_key_representations_erase_storage_on_drop() {
    let size = GlweSize::new(1, 16);
    let mut coefficients = vec![1i32; 2 * size.mask_len()];
    let data = coefficients.as_ptr().cast();
    let len = size_of_val(coefficients.as_slice());
    coefficients.truncate(size.mask_len());
    assert_erased_on_drop(
        GlweSecretKey::<u32>::new(coefficients, size, SecretKeyDistr::UniformBinary),
        data,
        len,
    );

    let mut ntt = vec![1u32; 2 * size.mask_len()];
    let data = ntt.as_ptr().cast();
    let len = size_of_val(ntt.as_slice());
    ntt.truncate(size.mask_len());
    assert_erased_on_drop(
        NttGlweSecretKey::new(ntt, size, SecretKeyDistr::UniformBinary),
        data,
        len,
    );

    let mut fourier = vec![Complex64::new(1.0, 1.0); 2 * size.fourier_mask_len()];
    let data = fourier.as_ptr().cast();
    let len = size_of_val(fourier.as_slice());
    fourier.truncate(size.fourier_mask_len());
    assert_erased_on_drop(
        FourierGlweSecretKey::new(fourier, size, SecretKeyDistr::UniformBinary),
        data,
        len,
    );
}
