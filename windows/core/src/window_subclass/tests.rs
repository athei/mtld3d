use super::*;

const WINDOW: usize = 0x1_0000;
const OTHER_WINDOW: usize = 0x2_0000;
const GAME_PROC: usize = 0x40_1000;
const HOOK: usize = 0x7F_0000;
const FIRST: usize = 0xA000;
const SECOND: usize = 0xB000;

/// A window procedure slot standing in for `SetWindowLongPtr(GWLP_WNDPROC)`.
struct WindowProc {
    current: usize,
    writes: u32,
}

impl WindowProc {
    const fn new() -> Self {
        Self {
            current: GAME_PROC,
            writes: 0,
        }
    }

    fn hook(&mut self) -> usize {
        self.writes += 1;
        core::mem::replace(&mut self.current, HOOK)
    }

    fn restore(&mut self, original: usize) {
        self.writes += 1;
        self.current = original;
    }
}

#[test]
fn a_second_device_joins_the_hook_instead_of_hooking_the_hook() {
    let mut proc = WindowProc::new();
    let mut subclasses = WindowSubclasses::default();
    assert!(subclasses.register(WINDOW, FIRST, || proc.hook()));
    assert!(!subclasses.register(WINDOW, SECOND, || proc.hook()));
    assert_eq!(
        proc.writes, 1,
        "only the first device replaces the procedure"
    );
    assert_eq!(
        subclasses.route(WINDOW),
        Some((FIRST, GAME_PROC)),
        "messages reach the first device and forward to the game's procedure, not the hook"
    );
}

#[test]
fn releases_in_either_order_give_the_window_its_procedure_back() {
    for first_leaves_first in [true, false] {
        let mut proc = WindowProc::new();
        let mut subclasses = WindowSubclasses::default();
        subclasses.register(WINDOW, FIRST, || proc.hook());
        subclasses.register(WINDOW, SECOND, || proc.hook());

        let (leaves, stays) = if first_leaves_first {
            (FIRST, SECOND)
        } else {
            (SECOND, FIRST)
        };
        assert!(!subclasses.unregister(WINDOW, leaves, |original| proc.restore(original)));
        assert_eq!(proc.current, HOOK, "the hook stays while a device is left");
        assert_eq!(
            subclasses.route(WINDOW),
            Some((stays, GAME_PROC)),
            "the device left behind receives the messages"
        );

        assert!(subclasses.unregister(WINDOW, stays, |original| proc.restore(original)));
        assert_eq!(
            proc.current, GAME_PROC,
            "the last device puts the game's procedure back"
        );
        assert_eq!(subclasses.route(WINDOW), None);
        assert_eq!(proc.writes, 2, "one install and one restore");
    }
}

#[test]
fn an_unregistered_device_or_window_changes_nothing() {
    let mut proc = WindowProc::new();
    let mut subclasses = WindowSubclasses::default();
    subclasses.register(WINDOW, FIRST, || proc.hook());

    assert!(!subclasses.unregister(OTHER_WINDOW, FIRST, |original| proc.restore(original)));
    assert!(!subclasses.unregister(WINDOW, SECOND, |original| proc.restore(original)));
    assert_eq!(subclasses.route(WINDOW), Some((FIRST, GAME_PROC)));
    assert_eq!(proc.current, HOOK);

    // Registering the same device twice keeps one registration.
    subclasses.register(WINDOW, FIRST, || proc.hook());
    assert!(subclasses.unregister(WINDOW, FIRST, |original| proc.restore(original)));
    assert_eq!(proc.current, GAME_PROC);
}

#[test]
fn a_device_moving_between_windows_leaves_the_one_it_came_from() {
    let mut first_window = WindowProc::new();
    let mut second_window = WindowProc::new();
    let mut subclasses = WindowSubclasses::default();
    subclasses.register(WINDOW, FIRST, || first_window.hook());
    subclasses.register(OTHER_WINDOW, SECOND, || second_window.hook());

    // A Reset that retargets the first device onto the second device's window.
    subclasses.unregister(WINDOW, FIRST, |original| first_window.restore(original));
    subclasses.register(OTHER_WINDOW, FIRST, || second_window.hook());
    assert_eq!(first_window.current, GAME_PROC);
    assert_eq!(second_window.writes, 1, "the second window is hooked once");
    assert_eq!(subclasses.route(OTHER_WINDOW), Some((SECOND, GAME_PROC)));
}
