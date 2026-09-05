//! The status a process ends with survives the detach that kills it.
//!
//! `d3d9.dll` ends a process that has created a device from its
//! `DLL_PROCESS_DETACH` (it cannot survive the allocator's thread-local
//! teardown on Wine's 1 MB main-thread stack), and the code of that
//! `TerminateProcess` is the one the unix side of Wine exits with, so a
//! status the process named itself has to be carried into it. Nothing inside
//! the process can read its own unix exit status, so the assertion belongs
//! to the runner: the test creates a device, declares on stdout the code it
//! is about to end with, and ends with it through the `ExitProcess` import
//! of this binary, and `unix/e2e` fails the test unless the process ended
//! exactly there.
//!
//! It is a binary of its own because it ends the process it runs in, and it
//! uses the shared harness on purpose: a static import is what puts
//! `d3d9.dll` in the process before the exit, which is the shape a game has.

use mtld3d_tests::Harness;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn ExitProcess(exit_code: u32) -> !;
}

/// The status this process ends with; no other exit in the suite uses it.
const EXIT_STATUS: u32 = 42;

/// The libtest name of the test below, which the runner reads back off stdout.
const TEST_NAME: &str = "the_exit_status_survives_the_device_detach";

#[test]
fn the_exit_status_survives_the_device_detach() {
    let harness = Harness::new();
    assert!(
        harness.device_refcount() > 0,
        "the device the detach path needs is not live"
    );
    // A leading newline closes libtest's open `test <name> ... ` line so the
    // marker starts one of its own.
    println!("\n[e2e] test {TEST_NAME} ends this process with exit code {EXIT_STATUS}");
    // SAFETY: kernel32's documented process exit, called with a status; it
    // does not return.
    unsafe { ExitProcess(EXIT_STATUS) }
}
