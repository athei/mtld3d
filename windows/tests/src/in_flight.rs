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
//!
//! A thread the test spawns itself carries no name of its own, so a device
//! it creates would name nobody and a panic on it would read
//! `thread '<unnamed>'`, which names no test either: the runner would have
//! to run every test that was in flight again to find the culprit. Every
//! worker of the suite is therefore spawned through [`spawn_scoped`], which
//! gives it the spawning thread's name. That name is the test's libtest
//! path on a thread libtest created, and on a worker it is the name that
//! worker was given, so a worker of a worker is named after the test too.
//! Each worker then announces once, the same name as its test, and the
//! runner reads a name it has seen before as the one test.

use std::{
    cell::Cell,
    thread::{self, Builder, Scope, ScopedJoinHandle},
};

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

/// Spawn a scoped thread named after the current one.
///
/// The name is what a panic report and an announcement carry, so a worker
/// spawned here fails and names itself as the test it works for rather
/// than as `<unnamed>`. A thread with no name (the main thread of a binary
/// that runs its tests there) spawns an unnamed worker, since there is no
/// test to name.
///
/// # Panics
///
/// Panics when the thread cannot be spawned, which no test can recover
/// from.
pub fn spawn_scoped<'scope, 'env, F, T>(
    scope: &'scope Scope<'scope, 'env>,
    f: F,
) -> ScopedJoinHandle<'scope, T>
where
    F: FnOnce() -> T + Send + 'scope,
    T: Send + 'scope,
{
    let current = thread::current();
    current
        .name()
        .map_or_else(Builder::new, |name| Builder::new().name(name.to_owned()))
        .spawn_scoped(scope, f)
        .expect("a test's worker thread can be spawned")
}
