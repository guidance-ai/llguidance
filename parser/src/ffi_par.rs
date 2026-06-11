use std::ffi::c_void;
use std::panic::AssertUnwindSafe;

use crate::ffi::{LlgCallback, LlgConstraintStep};
use crate::panic_utils;

fn par_compute_mask_inner(constraints: Vec<LlgConstraintStep>) {
    use rayon::prelude::*;
    constraints.into_par_iter().for_each(|step| {
        // Wrap each step in catch_unwind to prevent panics from aborting the
        // process via rayon's spawn (which has no scope to propagate to).
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let cc = unsafe { &mut *step.constraint };

            // Validate step parameters — set per-constraint error instead of panicking.
            if step.mask_byte_len % 4 != 0 {
                cc.set_error("llg_par_compute_mask: mask_byte_len is not a multiple of 4");
                return;
            }
            if step.mask_dest.is_null() {
                cc.set_error("llg_par_compute_mask: mask_dest is null");
                return;
            }
            let mask_elts = step.mask_byte_len / 4;

            if let Some(constraint) = &mut cc.constraint {
                let mut num_copied = 0;
                let mut add_eos = false;
                let eos = constraint.tok_trie().eos_token() as usize;
                match constraint.compute_mask() {
                    Ok(r) => {
                        if let Some(m) = r.sample_mask.as_ref() {
                            num_copied = std::cmp::min(m.len(), mask_elts);
                            // SAFETY: mask_dest is non-null (checked above), and
                            // mask_byte_len guarantees sufficient space.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    m.as_ptr(),
                                    step.mask_dest,
                                    num_copied,
                                );
                            }
                        }
                        add_eos = r.is_stop();
                    }
                    Err(e) => cc.set_error(&e.to_string()),
                }

                let left = mask_elts - num_copied;
                if left > 0 {
                    // SAFETY: mask_dest + num_copied is within the buffer.
                    unsafe {
                        std::ptr::write_bytes(step.mask_dest.add(num_copied), 0, left);
                    }
                }
                if add_eos && eos / 32 < mask_elts {
                    // SAFETY: eos / 32 < mask_elts, so this is within bounds.
                    unsafe {
                        *step.mask_dest.add(eos / 32) |= 1 << (eos % 32);
                    }
                }
            }
        }));

        if let Err(e) = result {
            // A panic escaped from compute_mask despite inner catch_unwind —
            // record it on the constraint handle so the caller can observe it.
            let cc = unsafe { &mut *step.constraint };
            cc.set_error(&panic_utils::mk_panic_error(&e));
        }
    });
}

pub(crate) fn par_compute_mask(
    constraints: Vec<LlgConstraintStep>,
    user_data: *const c_void,
    done_cb: LlgCallback,
) {
    struct CbData {
        user_data: *const c_void,
    }
    // SAFETY: `CbData` wraps a raw pointer that the C caller guarantees
    // remains valid until `done_cb` is invoked. The caller also guarantees
    // `done_cb` is thread-safe (it's an `extern "C" fn`).
    unsafe impl Send for CbData {}

    if let Some(cb) = done_cb {
        let ptr = CbData { user_data };
        rayon::spawn(move || {
            par_compute_mask_inner(constraints);
            cb(ptr.user_data);
            #[allow(clippy::drop_non_drop)]
            drop(ptr);
        });
    } else {
        par_compute_mask_inner(constraints);
    }
}
