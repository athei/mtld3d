//! The test a thread is running, named on stderr for the e2e runner.
//!
//! libtest writes a test's name before it runs only when there is one test
//! thread; on more it writes nothing until the test finishes, so a process
//! that dies with several tests in flight leaves no account of which they
//! were and the runner has to run every test left to find out. A test that
//! reaches `d3d9.dll` names itself here instead, and the runner reads the
//! names it has no outcome line for as the set that was in flight.
//!
//! The name goes to stderr because libtest's report goes to stdout, and a
//! result line there is three writes: the name, the outcome word, and the
//! newline. A print from a test thread that lands between the last two
//! joins the result and the announcement into one line, which is a result
//! the runner has to work to read back.
//!
//! The name is the thread's, because libtest runs every test on a thread
//! named after it, at any thread count. The main thread is skipped: libtest
//! runs a test there only when it cannot spawn a thread, and then the name
//! would be nobody's. Once per thread, since a test may build more than one
//! interface and libtest gives each test a thread of its own.

use std::{cell::Cell, thread};

/// The stderr marker; what follows it is the test's libtest path.
const RUNNING: &str = "[e2e] running ";

thread_local! {
    /// Whether this thread has already named its test.
    static NAMED: Cell<bool> = const { Cell::new(false) };
}

/// Name the test this thread runs on stderr, once per thread.
pub fn announce() {
    if NAMED.with(|named| named.replace(true)) {
        return;
    }
    let thread = thread::current();
    let Some(name) = thread.name().filter(|name| *name != "main") else {
        return;
    };
    eprintln!("{RUNNING}{name}");
}
