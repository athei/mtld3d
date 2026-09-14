use std::{
    sync::{Arc, Condvar, mpsc},
    time::Duration,
};

use super::*;

struct RejectSpawner {
    attempts: usize,
    accepted: usize,
}

impl Spawner for RejectSpawner {
    fn spawn<'scope, 'env: 'scope>(
        &mut self,
        scope: &'scope Scope<'scope, 'env>,
        work: impl FnOnce() + Send + 'scope,
    ) -> io::Result<()> {
        self.attempts += 1;
        if self.attempts > self.accepted {
            return Err(io::Error::other("injected thread creation failure"));
        }
        NativeSpawner.spawn(scope, work)
    }
}

#[test]
fn every_job_runs_once_and_results_keep_input_order() {
    let calls: Vec<_> = (0..128).map(|_| AtomicUsize::new(0)).collect();
    let stop = AtomicBool::new(false);
    let actual = map_with_spawner(
        &calls,
        &stop,
        MAX_WORKERS,
        &|calls| calls.fetch_add(1, Ordering::Relaxed),
        &mut NativeSpawner,
    );
    assert_eq!(actual, (0..128).map(|index| (index, 0)).collect::<Vec<_>>());
    assert!(calls.iter().all(|calls| calls.load(Ordering::Relaxed) == 1));
}

#[test]
fn rejected_worker_keeps_jobs_with_coordinator_and_existing_workers() {
    for accepted in [0, 2] {
        let mut spawner = RejectSpawner {
            attempts: 0,
            accepted,
        };
        let actual = map_with_spawner(
            &[1, 2, 3, 4, 5, 6],
            &AtomicBool::new(false),
            MAX_WORKERS,
            &|value| value * 2,
            &mut spawner,
        );
        assert_eq!(
            actual,
            vec![(0, 2), (1, 4), (2, 6), (3, 8), (4, 10), (5, 12)]
        );
        assert_eq!(spawner.attempts, accepted + 1);
    }
}

#[test]
fn worker_cap_includes_the_coordinator() {
    let mut spawner = RejectSpawner {
        attempts: 0,
        accepted: usize::MAX,
    };
    let actual = map_with_spawner(
        &[1; 32],
        &AtomicBool::new(false),
        usize::MAX,
        &|value| *value,
        &mut spawner,
    );
    assert_eq!(actual.len(), 32);
    assert_eq!(spawner.attempts, MAX_WORKERS - 1);
}

#[test]
fn empty_or_cancelled_batch_starts_no_worker_or_job() {
    let mut spawner = RejectSpawner {
        attempts: 0,
        accepted: 0,
    };
    for (items, cancelled) in [(&[][..], false), (&[1][..], true)] {
        let result = map_with_spawner(
            items,
            &AtomicBool::new(cancelled),
            MAX_WORKERS,
            &|_: &i32| panic!("no work admitted"),
            &mut spawner,
        );
        assert!(result.is_empty());
    }
    assert_eq!(spawner.attempts, 0);
}

struct ReleaseGate {
    released: Mutex<bool>,
    changed: Condvar,
}

impl ReleaseGate {
    fn wait(&self) {
        let (released, timeout) = self
            .changed
            .wait_timeout_while(
                self.released.lock().expect("gate lock"),
                Duration::from_secs(5),
                |released| !*released,
            )
            .expect("gate wait");
        let completed = *released;
        drop(released);
        assert!(
            completed && !timeout.timed_out(),
            "supervisor released admitted work"
        );
    }
}

struct ReleaseOnDrop<'a>(&'a ReleaseGate);

impl Drop for ReleaseOnDrop<'_> {
    fn drop(&mut self) {
        *self.0.released.lock().expect("gate lock") = true;
        self.0.changed.notify_all();
    }
}

#[test]
fn cancellation_waits_for_in_flight_jobs_and_leaves_other_device_running() {
    let stop = AtomicBool::new(false);
    let gate = ReleaseGate {
        released: Mutex::new(false),
        changed: Condvar::new(),
    };
    let (admitted, arrivals) = mpsc::channel();
    thread::scope(|scope| {
        let release = ReleaseOnDrop(&gate);
        let first = scope.spawn(|| {
            map_with_spawner(
                &[0; 32],
                &stop,
                2,
                &|_| {
                    admitted.send(()).expect("supervisor receives admissions");
                    gate.wait();
                    1
                },
                &mut NativeSpawner,
            )
        });
        arrivals
            .recv_timeout(Duration::from_secs(5))
            .expect("first job admitted");
        arrivals
            .recv_timeout(Duration::from_secs(5))
            .expect("second job admitted");
        stop.store(true, Ordering::Release);
        let other = map_with_spawner(
            &[2; 32],
            &AtomicBool::new(false),
            2,
            &|value| *value,
            &mut NativeSpawner,
        );
        assert_eq!(other.len(), 32);
        assert!(!first.is_finished(), "in-flight jobs retain the barrier");
        drop(release);
        assert_eq!(first.join().expect("batch completed").len(), 2);
    });
}

#[test]
fn returned_values_have_one_owner_after_all_workers_finish() {
    let value = Arc::new(());
    let results = map(&[0; 32], &AtomicBool::new(false), |_| Arc::clone(&value));
    assert_eq!(Arc::strong_count(&value), 33);
    drop(results);
    assert_eq!(Arc::strong_count(&value), 1);
}
