//! Unit tests for the abort-recovery queue behind `Staged` VB/IB and texture uploads.
//!
//! The propositions pinned here are the ones the encoder's correctness rests on: an entry
//! survives until its seq is acknowledged rather than merely retired, an abort re-arms the
//! entries at or below the failed seq plus the rest of their key's tail, a re-armed entry
//! lands behind everything already queued so the queue stays monotonic in seq, the replay
//! chain is bounded, and an entry re-armed under a later seq is released normally when that
//! seq retires cleanly.

use super::{MAX_REISSUE_ATTEMPTS, UploadFate, UploadRecoveryQueue};

/// Payload stand-in: the queue never looks inside one, so a label is enough.
fn queue() -> UploadRecoveryQueue<&'static str> {
    UploadRecoveryQueue::new()
}

#[test]
fn nothing_settles_before_its_seq_retires() {
    let mut q = queue();
    q.push(1, 5, "a");
    // coherent_seq = 4: the frame carrying "a" has not retired.
    assert!(q.settle(4, 0).is_empty());
    assert_eq!(q.len(), 1);
}

#[test]
fn a_retired_and_unfailed_entry_is_released() {
    let mut q = queue();
    q.push(1, 5, "a");
    let settled = q.settle(5, 0);
    assert_eq!(settled.len(), 1);
    assert_eq!(settled[0].0, UploadFate::Released);
    assert_eq!(*settled[0].1.payload(), "a");
    assert!(q.is_empty());
}

#[test]
fn retirement_alone_does_not_acknowledge_an_aborted_submit() {
    let mut q = queue();
    q.push(1, 5, "a");
    // The GPU retired seq 5 and failed at seq 5: retired, not acknowledged.
    let settled = q.settle(5, 5);
    assert_eq!(settled.len(), 1);
    assert_eq!(settled[0].0, UploadFate::Reissue);
}

#[test]
fn an_abort_rearms_the_entries_at_or_below_the_failed_seq() {
    let mut q = queue();
    q.push(1, 5, "a");
    q.push(2, 6, "b");
    q.push(3, 7, "c");
    // failed_seq = 6: "a" and "b" rode aborted submits, "c" did not. All
    // three retired, so all three settle.
    let settled = q.settle(7, 6);
    let fates: Vec<_> = settled
        .iter()
        .map(|(fate, entry)| (*entry.payload(), *fate))
        .collect();
    assert_eq!(
        fates,
        vec![
            ("a", UploadFate::Reissue),
            ("b", UploadFate::Reissue),
            ("c", UploadFate::Released),
        ]
    );
}

#[test]
fn an_aborted_key_drags_its_whole_tail_into_the_replay() {
    let mut q = queue();
    // Two uploads to one destination, then one to another. Only the first
    // rode the aborted submit, but replaying it alone would put its bytes
    // on top of the newer ones at seq 7.
    q.push(1, 5, "old");
    q.push(1, 7, "new");
    q.push(2, 7, "other");
    let settled = q.settle(7, 5);
    let fates: Vec<_> = settled
        .iter()
        .map(|(fate, entry)| (*entry.payload(), *fate))
        .collect();
    assert_eq!(
        fates,
        vec![
            ("old", UploadFate::Reissue),
            ("new", UploadFate::Reissue),
            ("other", UploadFate::Released),
        ]
    );
}

#[test]
fn a_poisoned_key_replays_even_before_its_own_seq_retires() {
    let mut q = queue();
    q.push(1, 5, "old");
    q.push(1, 9, "in_flight");
    // coherent_seq = 5: only "old" retired, but "in_flight" writes the same
    // destination and must land after the replay, so it comes out too.
    let settled = q.settle(5, 5);
    let payloads: Vec<_> = settled.iter().map(|(_, e)| *e.payload()).collect();
    assert_eq!(payloads, vec!["old", "in_flight"]);
    assert!(settled.iter().all(|(fate, _)| *fate == UploadFate::Reissue));
    assert!(q.is_empty());
}

#[test]
fn an_unrelated_key_stays_queued_across_a_recovery() {
    let mut q = queue();
    q.push(1, 5, "failed");
    q.push(2, 9, "untouched");
    let settled = q.settle(5, 5);
    assert_eq!(settled.len(), 1);
    assert_eq!(*settled[0].1.payload(), "failed");
    assert_eq!(q.len(), 1);
}

#[test]
fn a_rearmed_entry_lands_behind_everything_already_queued() {
    let mut q = queue();
    q.push(1, 5, "replayed");
    q.push(2, 9, "later");
    let mut settled = q.settle(5, 5);
    let entry = settled.pop().expect("one entry settled").1;
    q.requeue(entry, 12);
    // Queue order is the drain order, and it stays non-decreasing in seq:
    // the entry queued at seq 9 is still in front of the replay at 12.
    let order: Vec<_> = q.settle(12, 0).iter().map(|(_, e)| *e.payload()).collect();
    assert_eq!(order, vec!["later", "replayed"]);
}

#[test]
fn requeue_holds_exactly_one_entry_so_bytes_are_counted_once() {
    let mut q = queue();
    q.push(1, 5, "a");
    let mut settled = q.settle(5, 5);
    assert!(q.is_empty());
    let entry = settled.pop().expect("one entry settled").1;
    q.requeue(entry, 8);
    assert_eq!(q.len(), 1);
}

#[test]
fn an_entry_rearmed_under_a_later_seq_is_released_when_that_seq_is_clean() {
    let mut q = queue();
    q.push(1, 5, "a");
    let mut settled = q.settle(5, 5);
    let entry = settled.pop().expect("one entry settled").1;
    q.requeue(entry, 8);
    // The replay rode seq 8, which retired without failing. failed_seq is
    // still 5, below the entry's new seq, so the abort no longer applies.
    let settled = q.settle(8, 5);
    assert_eq!(settled.len(), 1);
    assert_eq!(settled[0].0, UploadFate::Released);
    assert_eq!(settled[0].1.attempts(), 1);
}

#[test]
fn the_replay_chain_is_bounded() {
    let mut q = queue();
    q.push(1, 1, "a");
    let mut seq = 1;
    for expected_attempts in 0..MAX_REISSUE_ATTEMPTS {
        let mut settled = q.settle(seq, seq);
        assert_eq!(settled.len(), 1);
        assert_eq!(settled[0].0, UploadFate::Reissue);
        let entry = settled.pop().expect("one entry settled").1;
        assert_eq!(entry.attempts(), expected_attempts);
        seq += 1;
        q.requeue(entry, seq);
    }
    // Attempt MAX + 1: the entry is abandoned rather than held forever.
    let settled = q.settle(seq, seq);
    assert_eq!(settled.len(), 1);
    assert_eq!(settled[0].0, UploadFate::Abandoned);
    assert_eq!(settled[0].1.attempts(), MAX_REISSUE_ATTEMPTS);
    assert!(q.is_empty());
}

#[test]
fn a_zero_failed_seq_never_rearms() {
    let mut q = queue();
    // Seq 0 is the pre-first-frame sentinel; an entry can never be stamped
    // with it, but the atomic reads 0 until the first abort.
    q.push(1, 1, "a");
    let settled = q.settle(1, 0);
    assert_eq!(settled[0].0, UploadFate::Released);
}

#[test]
fn drain_all_empties_the_queue() {
    let mut q = queue();
    q.push(1, 5, "a");
    q.push(2, 6, "b");
    let drained = q.drain_all();
    let payloads: Vec<_> = drained.iter().map(|e| *e.payload()).collect();
    assert_eq!(payloads, vec!["a", "b"]);
    assert!(q.is_empty());
    assert!(q.drain_all().is_empty());
}

#[test]
fn release_acknowledged_stops_at_the_first_entry_owing_a_replay() {
    let mut q = queue();
    q.push(1, 5, "aborted");
    q.push(2, 6, "behind");
    // Both seqs retired, but the front owes a replay, so nothing is freed:
    // releasing "behind" first would let its bytes settle underneath a
    // replay that has not happened yet.
    assert!(q.release_acknowledged(6, 5).is_empty());
    assert_eq!(q.len(), 2);
}

#[test]
fn release_acknowledged_frees_the_prefix_above_the_failed_seq() {
    let mut q = queue();
    // The abort at seq 5 was settled in an earlier pass; what is left was
    // queued after it.
    q.push(1, 6, "a");
    q.push(2, 7, "b");
    let released = q.release_acknowledged(7, 5);
    let payloads: Vec<_> = released.iter().map(|e| *e.payload()).collect();
    assert_eq!(payloads, vec!["a", "b"]);
    assert!(q.is_empty());
}

#[test]
fn release_acknowledged_frees_everything_retired_when_nothing_failed() {
    let mut q = queue();
    q.push(1, 5, "a");
    q.push(2, 6, "b");
    q.push(3, 9, "c");
    let released = q.release_acknowledged(6, 0);
    let payloads: Vec<_> = released.iter().map(|e| *e.payload()).collect();
    assert_eq!(payloads, vec!["a", "b"]);
    assert_eq!(q.len(), 1);
}
