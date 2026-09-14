# Issue 673 review

## Failure mechanism and chosen recovery

The original worker spawn converted the OS error to None. Its failed closure dropped only a cloned PrewarmSender while EncoderThread retained the original sender. The encoder could not leave prewarm_rx.recv to receive Frame, Reset or Shutdown. This was source-confirmed, not an observed game hang or an actual OS-exhaustion experiment.

The extracted core helper creates the result channel and places its sole sender inside the worker closure. A rejected spawn consumes/drops that closure, closes the channel, logs the actual OS error and leaves no join handle. Core receive returns None for a disconnected channel or an explicitly unusable cache. Some(warm), including Some(empty), is successful completion. The PE encoder ingests None as empty with writes_disabled=true.

The safe recovery continues startup cold and disables persistence for that device. Returning an empty writable cache would be unsafe because no prewarm load validated or repaired the existing file. Running prewarm synchronously after a failed spawn would require retaining/recovering its closure and add startup work to the API thread. Existing device/encoder failure-abort behavior is otherwise unchanged.

## Source checks

FrameEncoder::new only initializes encoder-owned caches and starts the idle submit worker, so prewarm can start first with the already-created Metal device. The encoder still consumes the result before any EncoderMessage. Prewarm work and frame translation never overlap. Device release still calls cancel_and_join before encoder cleanup. Cancellation preserves the existing AtomicBool ordering and JoinHandle::is_finished polling rather than using Wine's potentially invalid OS thread handle.

Both cache_write_record and cache_write_pipeline return on CACHE_DISABLED before opening CacheWriter. That is the source proof that disconnected startup cannot modify an unvalidated file. The host test executes the real core receive policy and asserts None, without pretending a sentinel-file imitation exercises the PE encoder.

The cap-1 result channel still owns completed results until encoder intake. The success payload holds the same raw Metal handle collections as before. Original send and extracted helper send both drop the returned SendError if the receiver is gone. No new disconnect path is introduced during normal device lifetime: release retains the receiver in the encoder until prewarm finishes. Both worker-spawn and encoder-spawn fatal error ordering have a valid Metal device already created; an encoder-spawn error still aborts rather than unwinding an active prewarm.

## Regression evidence

Four host tests execute the real core startup/cancellation helper. Rejected spawn injects an io::Error while consuming and dropping the work closure; the body never runs, the receiver immediately reports Disconnected, receive produces None, and a consumer blocked behind the barrier accepts shutdown and finishes within the bounded test deadline. Other tests cover a queued frame waiting for the actual warm payload, cancellation with a completed but unread payload, and unusable-cache None versus validated-empty Some(empty).

The red test is a mutation test, not an execution on unmodified baseline source. retained-sender-mutation.patch adds the original sender-retention mechanism to the extracted helper. The rejected-spawn test fails immediately because try_recv returns Empty instead of Disconnected (exit 101). The fixed helper passes all four tests. The mutation is not part of the commit.

## Rules check

CONTRIBUTING.md: fmt/check, full isolated test, and full conformance are required. No schema, key, render state or config semantics change, so no companion cache bump, classifier, caps, config sample or coverage row is needed. Existing restart/cache e2e tests exercise healthy prewarm.

CONVENTIONS.md: pure startup ownership, cancellation and receive policy live in mtld3d-core, with tests in shader_prewarm/tests.rs. The private boxed closure spawner exists solely to exercise real thread rejection without OS exhaustion. Metal/compiler work remains PE wiring. No new dependency, derive, lint suppression, config key, wire field, unsafe operation or explicit static. The existing receiver log_once warning moves into core and the new rare spawn error logs each failure. The old worker lifecycle is removed from PE rather than duplicated. Added must_use follows clippy.

ARCHITECTURE.md: startup completion still precedes frames, API/encoder/submit overlap and cap-1 backpressure remain, and cancellation still precedes device cleanup. Documented the failure's persistence policy beside existing prewarm ordering.

## Final verification

make fmt and make check passed (check-3.log). make ISOLATED=1 test passed (test-final.log): 1192 Windows-workspace host tests, 404 Unix-workspace host tests, 575 end-to-end tests per PE architecture, four processes each. Both snmalloc_drift probes reported their passing result then exited code 5; the runner accepted their completed results. They do not load d3d9. Existing new-device and process-restart pipeline prewarm tests passed on both architectures.

Full make ISOLATED=1 conformance passed on i686 and x86_64, all four subtests each, with no regressions or Metal-validation errors. The SDK is wine-11.0-13-g28fe1321fd7 while the baseline records wine-11.0-11-gd8c13289d4d; only existing ceiling/flaky reductions were tolerated. No baseline edit. Raw outputs remain in conformance-raw. build-identity.txt verifies installed implementation DLLs match the worktree build except for the expected winebuild builtin DOS-stub signature.
