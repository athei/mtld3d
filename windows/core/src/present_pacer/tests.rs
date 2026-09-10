use std::time::{Duration, Instant};

use super::PresentPacer;

const PERIOD: Duration = Duration::from_millis(8);

#[test]
fn the_first_present_starts_the_cadence_without_waiting() {
    let mut pacer = PresentPacer::new();
    let now = Instant::now();
    assert_eq!(pacer.deadline(now, PERIOD), None);
    assert_eq!(
        pacer.deadline(now + Duration::from_millis(3), PERIOD),
        Some(now + PERIOD)
    );
}

#[test]
fn a_kept_deadline_advances_by_the_period_without_drift() {
    let mut pacer = PresentPacer::new();
    let now = Instant::now();
    assert_eq!(pacer.deadline(now, PERIOD), None);
    assert_eq!(
        pacer.deadline(now + Duration::from_millis(5), PERIOD),
        Some(now + PERIOD)
    );
    assert_eq!(
        pacer.deadline(now + Duration::from_millis(9), PERIOD),
        Some(now + 2 * PERIOD)
    );
}

#[test]
fn an_overrun_restarts_the_cadence_from_now() {
    let mut pacer = PresentPacer::new();
    let now = Instant::now();
    assert_eq!(pacer.deadline(now, PERIOD), None);
    let late = now + Duration::from_millis(30);
    assert_eq!(pacer.deadline(late, PERIOD), None);
    assert_eq!(
        pacer.deadline(late + Duration::from_millis(2), PERIOD),
        Some(late + PERIOD)
    );
}

#[test]
fn a_decision_is_reported_once_until_it_changes() {
    let mut pacer = PresentPacer::new();
    assert!(pacer.report_changed(Some(PERIOD)));
    assert!(!pacer.report_changed(Some(PERIOD)));
    assert!(pacer.report_changed(None));
    assert!(!pacer.report_changed(None));
    assert!(pacer.report_changed(Some(PERIOD)));
}
