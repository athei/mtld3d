use super::*;

/// The page offset of a local in a frame of its own, as the pinned call's callees see it.
#[inline(never)]
fn callee_offset() -> usize {
    let local = 0u8;
    core::ptr::from_ref(core::hint::black_box(&local)) as usize % PAGE
}

/// The callee's page offset under the pinned call, entered from `depth` extra frames down.
#[inline(never)]
fn pinned_offset_from(depth: usize) -> usize {
    if depth > 0 {
        let pad = [0u8; 48];
        core::hint::black_box(&pad);
        return core::hint::black_box(pinned_offset_from(depth - 1));
    }
    run_pinned(callee_offset)
}

#[test]
fn pinned_call_starts_at_one_page_offset_whatever_the_caller_depth() {
    let offsets: Vec<usize> = (0..200).map(pinned_offset_from).collect();
    // The gap moves in STEP units, so the callee lands in one STEP-wide window,
    // widened by the one register pair a small gap frame may save fewer of.
    let lowest = offsets.iter().min().copied().unwrap();
    let highest = offsets.iter().max().copied().unwrap();
    assert!(
        highest - lowest <= STEP + 16,
        "pinned offsets spread {lowest:#x}..{highest:#x}: {offsets:x?}"
    );
}

/// Whether [`at_pin`] holds at `depth` extra frames down, inside the pinned call or not.
#[inline(never)]
fn at_pin_from(depth: usize, pinned: bool) -> bool {
    if depth > 0 {
        let pad = [0u8; 48];
        core::hint::black_box(&pad);
        return core::hint::black_box(at_pin_from(depth - 1, pinned));
    }
    if pinned { run_pinned(at_pin) } else { at_pin() }
}

#[test]
fn at_pin_holds_inside_the_pinned_call_and_not_everywhere_else() {
    assert!((0..200).all(|depth| at_pin_from(depth, true)));
    assert!((0..200).any(|depth| !at_pin_from(depth, false)));
}

#[test]
fn pinned_call_returns_its_value() {
    let mut calls = 0;
    let value = run_pinned(|| {
        calls += 1;
        41 + calls
    });
    assert_eq!((value, calls), (42, 1));
}
