//! Unit tests for the translation query.
//!
//! The kernel answers for the test process itself: an arm64 build always runs
//! natively, and an `x86_64` one runs translated exactly on Apple Silicon.

use super::translated;

#[test]
fn the_kernel_answers_for_this_process() {
    let answer = translated();
    assert!(answer.is_some(), "sysctl.proc_translated gave no answer");
    if cfg!(target_arch = "aarch64") {
        assert_eq!(answer, Some(false), "an arm64 process is never translated");
    }
}
