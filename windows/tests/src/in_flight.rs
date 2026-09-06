//! The test a thread is running, named on stdout for the e2e runner.
//!
//! libtest writes a test's name before it runs only when there is one test
//! thread; on more it writes nothing until the test finishes, so a process
//! that dies with several tests in flight leaves no account of which they
//! were and the runner has to run every test left to find out. A test that
//! reaches `d3d9.dll` names itself here instead, and the runner reads the
//! names it has no outcome line for as the set that was in flight.
//!
//! The name is the thread's, because libtest names every test thread after
//! its test. The main thread is skipped: that is where libtest runs the
//! tests when there is one thread, and there its own start line already
//! names them. Once per thread, since a test may build more than one
//! interface and libtest gives each test a thread of its own.

use std::{cell::Cell, thread};

/// The stdout marker; what follows it is the test's libtest path.
const RUNNING: &str = "[e2e] running ";

thread_local! {
    /// Whether this thread has already named its test.
    static NAMED: Cell<bool> = const { Cell::new(false) };
}

/// Name the test this thread runs on stdout, once per thread.
pub fn announce() {
    if NAMED.with(|named| named.replace(true)) {
        return;
    }
    let thread = thread::current();
    let Some(name) = thread.name().filter(|name| *name != "main") else {
        return;
    };
    println!("{RUNNING}{name}");
}
