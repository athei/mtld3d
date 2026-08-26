//! The PE-side logger's sink: a queue drained by one logging thread.
//!
//! `env_logger` hands every formatted line to [`Sink::write`], which only
//! pushes the bytes onto an unbounded channel: no unix call and no blocking,
//! so the API and encoder threads pay an allocation and a queue push per line
//! and nothing else. One logging thread, started by [`start`] from the first
//! `Direct3DCreate9` (never from `DllMain`, which runs under the loader lock),
//! drains the queue and forwards each line through the `WriteLog` thunk to the
//! process's unix stderr. Lines logged before the thread exists wait in the
//! queue; the queue keeps their order.

use std::sync::{
    LazyLock, Mutex,
    mpsc::{Receiver, Sender, channel},
};

use mtld3d_shared::WriteLogParams;

use crate::unix_call::unix_call;

struct Queue {
    tx: Sender<Vec<u8>>,
    /// Handed to the logging thread by [`start`]; `None` once it has been.
    rx: Mutex<Option<Receiver<Vec<u8>>>>,
}

static QUEUE: LazyLock<Queue> = LazyLock::new(|| {
    let (tx, rx) = channel();
    Queue {
        tx,
        rx: Mutex::new(Some(rx)),
    }
});

/// The `env_logger` target: queues the line for the logging thread.
pub struct Sink;

impl std::io::Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // A closed receiver means the logging thread is gone; the line is
        // dropped rather than the caller failing.
        let _ = QUEUE.tx.send(buf.to_vec());
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Start the logging thread; a no-op after the first call.
///
/// Called from `Direct3DCreate9`, the first entry point a game reaches outside
/// `DllMain`. The handle is dropped: the thread is detached and lives for the
/// process.
pub fn start() {
    let rx = QUEUE
        .rx
        .lock()
        .expect("log queue receiver lock poisoned")
        .take();
    let Some(rx) = rx else {
        return;
    };
    let spawned = std::thread::Builder::new()
        .name("mtld3d-log".into())
        .spawn(move || {
            for line in rx {
                forward(&line);
            }
        });
    // On a spawn failure the closure, and the receiver with it, is dropped:
    // every later push fails and the line is discarded, which is the most
    // the logger can do without a thread of its own.
    drop(spawned);
}

/// Hand `line` to the unix side right now, on the calling thread.
///
/// For the crash path only: a fault or panic handler cannot rely on the
/// logging thread ever running again, so it thunks synchronously instead
/// of queueing. Everything else goes through [`Sink`].
pub fn write_raw(line: &[u8]) {
    forward(line);
}

/// One formatted line across the boundary; the unix side writes it to stderr.
fn forward(line: &[u8]) {
    let Ok(len) = u32::try_from(line.len()) else {
        return;
    };
    let mut params = WriteLogParams {
        ptr: line.as_ptr() as usize as u64,
        len,
        pad0: 0,
    };
    unix_call(&mut params);
}
