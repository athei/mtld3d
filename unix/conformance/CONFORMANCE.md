# d3d9 conformance against Wine's test suite

The de-facto conformance suite for any D3D9 reimplementation is Wine's
`dlls/d3d9/tests/` (there is no public, portable Microsoft D3D9 conformance
kit — the WHQL/HLK tests are driver-certification machinery). Those tests build
into one `d3d9_test.exe` per architecture, with four subtests selected by
source-file stem: `device`, `visual`, `stateblock`, `d3d9ex`.

Because our `d3d9.dll` is installed as a Wine *builtin* (`make install`), running
`d3d9_test.exe` exercises our implementation directly.

## Running

```
make conformance                # diff both arches vs baseline.txt
make conformance-i686           # one arch, one runner process (what CI runs)
make conformance-x86_64
make conformance-intel          # both arches under the intel.* config keys
make conformance-intel-i686     # one arch under the intel.* keys
make conformance-scale          # both arches at render.scale = 0.75
make conformance-scale-i686     # one arch at render.scale = 0.75 (what CI runs, on one image)
make conformance-baseline       # (re)record this machine's six legs of baseline.txt in sequence
ARM64=1 make conformance        # also both arches under the arm64 Wine (The arm64-runtime legs, below)
EC=1 make conformance-arm64x    # the x86_64 binary against the ARM64X build, same Wine
```

A leg is one architecture under one variant on one GPU family. The `intel`
variant (`--variant intel`) runs the same binary with every `intel.*` key of
`mtld3d.conf` turned on, so the suite sees the answers an Intel/AMD Mac gives:
packed 16-bit formats expanded, 32-bit float filtering denied, Managed
buffers, the 256-byte linear texture alignment. Its results record under
`[<arch>+intel/<subtest>]` entries of the same `baseline.txt`. The GPU family
is read from the machine, not chosen: an Apple Silicon Mac records the plain
entries, an Intel/AMD Mac records `[<leg>@mac2/<subtest>]` entries, because
the GPU underneath still decides what the suite sees past the forced answers
(a tile-based Apple GPU elides depth stores and merges hidden overdraw before
the visibility counter, encodes special floats its own way, and its validation
layer applies different texture rules). The `scale` variant (`--variant
scale`) runs the same binary at `render.scale = 0.75`, so every frame is
rasterized on a smaller grid and read back through the resolve that brings it
up to the reported size; its results record under `[<arch>+scale/<subtest>]`
and CI runs it on one image and one arch, like the scaled end-to-end leg.
The sites share the classifications below whatever the leg, since a site's
nature does not depend on the leg that hit it. On the Apple family the two keys that only change a code path and no
answer, `intel.managedMemory` and `intel.linearAlign256`, must move no count
at all, and a site that fails only under the variant is expected to trace to
one of the two caps keys. The `@mac2` entries are recorded by CI, whose
Intel image is the one Mac2 machine the project runs on: dispatch the
workflow with `record_intel_baseline` (and `intel_only`, so only the Intel
legs run) and copy the `@mac2` sections out of the `baseline-mac2-<arch>`
artifacts (`make conformance-baseline` on an Apple Silicon machine leaves
them untouched, the merge being leg-scoped). Pull requests do not run the
Intel image, so a change that fixes a site updates the Apple entries and
leaves the site's `@mac2` pin above the new count; the next nightly Intel run
then fails on a stale baseline, not a regression. The fix is that same
dispatch, `record_intel_baseline` with `intel_only`, best made on the pull
request's branch before it merges so the `@mac2` sections land with it, or
on `main` once the nightly shows the stale pin.

### The arm64-runtime legs

`ARM64=1 make conformance` adds `conformance-i686-arm64` and
`conformance-x86_64-arm64`, and `EC=1` adds `conformance-arm64x`. Each runs
the SDK's own `d3d9_test.exe` of its arch, the binary `conformance-<arch>`
runs (a PE test runs under any Wine), under the arm64 Wine that `WINE_ARM64`
names: the two x86 legs against the i686 and x86_64 builds, the ARM64X leg
the x86_64 binary against the ARM64X build. Every leg runs in a private clone
of that Wine with its own prefix, created afresh after the leg's install and
configured by `configure-test-prefix` as the other legs' prefixes are.

None of them has baseline entries of its own. Each records under its arch's
label, so the runner judges it against that arch's entries, which were taken
on the x86_64 Wine the SDK is (the header's `Wine:` line names it, and the
runner warns that the arm64 Wine's version differs), and on the same Apple GPU
family. What a leg reports is therefore how its runtime differs from that
baseline: CrossOver's arm64 Wine and its x86 translation, plus, for the
ARM64X leg, the ARM64X build. How such runs should be keyed, as legs of their
own or under the existing entries, is not decided, so none has a baseline
target, and a site one of them moves is not reclassified here on its evidence
alone.

Set `MTLD3D_CONFORMANCE_RAW_DIR=<dir>` to also persist each subtest's full raw
output to `<dir>/<leg>-<subtest>.log`. The normal run reduces output to per-site
counts and drops the assertion text; the raw logs keep every
`<file>.c:<line>: Test failed: Got <actual>, expected <expected>` message (plus
the Metal-validation lines), which is what the per-cluster audit below was built
from — the *actual-vs-expected* values distinguish a real defect from an
acceptable `caps` difference. Off unless the variable is set.

The runner reads the test process's stderr as it arrives, and the driver's
GPU-hang line (`kIOAccelCommandBufferCallbackErrorHang`, or the
`...SubmissionsIgnored` line the driver prints for every command buffer it
drops afterwards) stops the subtest at once and ends the leg: exit code 3, no
verdict, no baseline write. Every count after that line is a read off a GPU
that runs nothing, and waiting out the subtest's budget would only make the
same non-verdict cost minutes. On the Intel CI image the paravirtual GPU stays
hung for the rest of the machine's life, so the later subtests would hang too.
The job marks the leg with a `GPU hang` annotation, and the answer is a re-run
of the failed jobs, which lands them on fresh runners; CI does that once by
itself (`CONTRIBUTING.md`, "Pull requests", says when).
A hang on a real GPU is worth a look on its own (a shader that hangs the GPU is
a bug), but the leg has to run again for its counts either way.

A raw log is stdout followed by stderr, and ends with how the process ended:
`[conformance] subtest exited: code N` or `signal N` (a number, never a name,
so a fault the process survived cannot read as a crash to the scanner), or the
`TIMED OUT` line when the runner killed it. A process the runner kills for its
budget is sampled first (`sample <pid> 2`, every thread's stack), and so is the
wineserver serving its prefix, the process it may be waiting on: the samples
are kept beside the raw log as `<leg>-<subtest>.sample.txt` and
`<leg>-<subtest>.wineserver-sample.txt`, both named on that line; without a raw
directory they go to the runner's stderr. They are the one account of where a
hang was: the raw log of a process parked in a syscall ends in that line and
nothing else, and a thread parked in `wine_server_call` shows nothing of the
server that owes it a reply. The server is matched on two things, so no other
prefix's is ever read: the binary it was exec'd from, which the Makefile names
with `--wineserver` out of the same Wine SDK as the loader, and the directory
it works from, which wineserver names after the device and inode of the prefix
it serves. When no such server is running, the file says so instead of failing
the leg. A run without the framework's `tests executed` summary is a crash
whatever else it holds, and this line is what tells an unhandled Win32
exception (Wine ends the process with the exception code, of which unix keeps
the low byte: `code 5` is an access violation) from a signal (11 `SIGSEGV`,
10 `SIGBUS`, 6 `SIGABRT`, 9 `SIGKILL`).
A run that reached its summary reads `code 0` once a device existed (the layer
ends the process from its detach, see CONTRIBUTING.md), else the framework's
failure count capped at 255.
Each process is also told `log.dir=Z:<dir>/<leg>-<subtest>`, so its log file
(`d3d9_test-<pid>.log`, and any GPU trace) lands in a directory beside its raw
output, one per process. The runner's `--log <filter>` (`LOG=` for the Makefile
targets) is the `RUST_LOG` those processes run under; the default `off` writes
nothing, and the crash handlers' lines then go to stderr, i.e. into the raw log.

A repeat run (`make conformance-isolate REPEAT=<n>`) keeps every run,
`<leg>-<subtest>-<n>.log`, with its own log directory. Dispatching the workflow
with `conformance_repeat=<n>` runs it on every conformance job, under
`LOG=debug`, and uploads the lot as `conformance-raw-<image>-<arch>`: the way to
make a subtest that dies one run in a few die in one sitting, on the machine
where it does.

In CI the raw artifact of every conformance job also carries `system/`: the
system log of the job's span (the job budget plus five minutes), with the
kernel and GPU lines the e2e legs collect and every line of the `wine`
processes that is an error or fault or names an exception, and
`collection-status.txt` saying what was collected and how. A sample shows
where a hung process waits; when it waits on a main-thread request after
AppKit swallowed an Objective-C exception (the sample then has a thread
named `SOME_OTHER_THREAD_SWALLOWED_AT_LEAST_ONE_EXCEPTION`), the system log
is where AppKit reports the exception, if it logs it there and does not
redact it.

There is no conformance-specific input to set. The test binaries ship inside the
Wine SDK bundle (`$WINE_SDK/lib/wine/tests/{i386,x86_64}-windows/d3d9_test.exe`,
published by the [wine-build](https://github.com/athei/wine-build) bundle step),
which is the same install `make install` puts our builtin `d3d9.dll` into, so a
CI job needs nothing but that tarball and no Wine build tree at all. The
binaries are **not** vendored here: they are large and drift with the Wine
version, so `baseline.txt` records the Wine version it was taken against
instead.

The runner is the Rust tool `mtld3d-conformance` (`unix/conformance/`). It takes
the loader and one test binary as explicit paths (`--wine`, `--exe`, plus
`--arch` as the label to record under) and resolves nothing itself: every Wine
location lives in the Makefile. One invocation therefore covers one leg,
which is what lets the 32-bit and 64-bit gates, and the native and Intel
variants, be separate CI jobs, and `--update-baseline` rewrites only its own
leg's entries. It runs each
subtest as its own process, so a crash in one cannot poison another's counts,
with Metal API validation left on in `nslog` mode (it logs rather than aborting,
so it cannot mask the failure counts) and with our logs and Wine's debug
channels silenced.

The layer's validation *errors* are reported, deduplicated, as
`metal-validation:` lines; its *warnings* are ignored. A warning is a
performance hint, not misuse (a resource bound to an encoder no draw went on to
read, a state setter overwritten before the next draw), and a leg emits
thousands of them, so leaving them on buried the error lines in output that
looked identical. A `metal-validation:` line therefore means the layer
committed API misuse, so any of them fails the leg. The expected number is
zero, kept as a constant in the runner next to the reporting code rather than
in `baseline.txt`, which records machine-owned per-site counts and nothing
else. There is no tolerance to keep in step: a leg that logs a message has
started misusing Metal, and the fix is the misuse, not the number.

The layer writes a report as a headline naming the check that fired
(`Sampler Descriptor Validation`) followed by unadorned detail lines, and the
detail is the half that names what was rejected. The runner keeps them
together: a recognised line opens a message and the lines under it are its
detail, up to the next `NSLog` line, Wine channel line or blank line, so the
whole report reaches the log indented under its `metal-validation:` line
instead of only in the raw output. One report counts once.

This is **not** part of `make test`: many checks fail by design (see below), so
it is a tracked-score tool, not a pass/fail gate on zero. The runner exits
non-zero on a *regression* vs the baseline — a per-site failure count that went
up, a new failing site, or a subtest that started crashing — and equally on a
*stale baseline* — a count that dropped, a site that disappeared, or a crash
that cleared. An improvement fails the gate on purpose: tolerating it would let
baseline.txt overstate reality, and the surplus becomes a budget a later
regression can hide in. The fix for a stale baseline is `make
conformance-baseline` plus the matching triage edit here, not a code hunt. The
`flaky` and `ceiling` classes are the two tolerances (see below).

A recognized upstream skip is an absent measurement, not a passing assertion.
The runner retains the skip line separately from failures and reports the
bypassed assertion as `SKIPPED`, without requiring a re-record. This applies
only to the reviewed source location and message below, in a completed
subtest with no failure at the affected assertion. Other missing failures
still gate as stale, and crashes and counts above the pin still gate.
Baseline recording retains an explicitly skipped assertion's prior count and
reports that retention; it never invents a count when no prior pin exists.

Metal validation is the third verdict, and it is independent of the counts: a
leg that logged any `metal-validation:` line exits non-zero even when every
site holds its pin, because API misuse is invisible to a pass/fail count. A
`--update-baseline` or `--repeat` run never gates, so neither one applies it.

## Kept divergences

Divergences from D3D9 kept on purpose, because closing them costs frame time,
memory, or a game that relies on the looser behaviour. `docs/STATUS.md` lists them
in one line each; this section is where the reasoning lives. Where Wine's
suite observes a divergence, the cluster below carries the site-level detail
and is named here; the rest have no conformance site and this is their only
record. A knob, where one makes sense, is named with its default.

- **`IDirect3DTexture9::LockRect` serves a level of a DEFAULT-pool 2D texture
  created without `D3DUSAGE_DYNAMIC`**, which D3D9 rejects. The surface entry
  point, cube and volume locks still reject it. A game that streams into such
  a texture would otherwise lose every upload. The cost is system memory: the
  level's staging, released once its upload retires, is re-created, and a
  partial lock after that release leaves the pixels outside its rect out of
  step with the GPU copy (warned once per texture). A render-target texture
  holds no staging until a CPU path first needs a level, and every lock of a
  level reads it back from the GPU first, unless the level holds a CPU write
  no upload has carried yet. No knob.
- **`GetData(D3DGETDATA_FLUSH)` can answer a pending occlusion query at once**
  instead of waiting for the GPU. Off by default. This saves API-thread time
  only for a title verified to use the poll as a submission throttle, without
  reading the count or gating reuse of CPU-writable dynamic storage. Metal
  does not track a CPU write through `D3DLOCK_NOOVERWRITE` into pages a queued
  draw still reads, so an immediate answer cannot safely fence that reuse. The
  `wow` profile currently turns it on because skipping its loading-screen waits
  was measured to save seconds per load; that benefit does not establish the
  absence of this hazard. Knob: `query.flushImmediate`, default `false`.
- EVENT query polls queue their open frame even without `D3DGETDATA_FLUSH`.
  D3D9 permits a poll without the flag to remain pending indefinitely. We
  make progress for callers that fence storage reuse before Present without
  setting the flag. The first poll splits the frame and queues its work on
  the existing bounded encoder channel; later polls only read retirement.
  There is no synchronous encoder or submit completion wait, but admission
  can backpressure like Present. Completion still requires GPU retirement,
  unless the next entry's knob is on. No knob.
- **An EVENT query poll can answer completed at once** instead of waiting for
  the GPU to retire the frame the query was issued in. Off by default. A title
  that polls the query only to keep the CPU from running ahead of the GPU
  gains nothing from the real answer, because the encoder and submit threads
  already bound how far ahead it can get; waiting for retirement makes its
  CPU and GPU work run back to back instead. The immediate answer cannot fence
  reuse of CPU-writable dynamic storage behind a `D3DLOCK_NOOVERWRITE` lock,
  so it is only for titles verified not to rely on that. The `wow` profile
  turns it on for `gxFixLag`, which took 3.3.5a from 120 FPS to 77 with the
  same GPU time per frame; through 0.9.0 every EVENT poll answered at once
  on both clients. Knob: `query.eventImmediate`, default `false`.
- **Depth and stencil are discarded at every `Present` on a surface nothing
  samples.** D3D9 keeps a depth-stencil surface's contents across `Present`
  and leaves them undefined only when the game asks for it:
  `D3DPRESENTFLAG_DISCARD_DEPTHSTENCIL` on the implicit surface, or
  `Discard = TRUE` passed to `CreateDepthStencilSurface`. We consult neither.
  The last pass of a frame to use a depth surface stores nothing
  (store-action Rule B) and the first pass of the next frame loads nothing
  (Rule A), unless the surface is sampleable or has been bound as a texture.
  Preserving depth costs a store and a load of every depth surface on every
  frame of every game that clears it anyway, which is all tested ones.
  Content that tests against depth or stencil from an earlier frame without
  clearing reads undefined values. Sites: `z_range_test` and `texdepth_test`
  below. No knob: the gates are compile-time constants in
  `windows/core/src/passes.rs`.
- **A partial `Lock` of a dynamic vertex or index buffer without
  `D3DLOCK_DISCARD` returns a pointer into memory a queued draw may still
  read.** D3D9 keeps the game's writes from landing under a draw the GPU has
  not reached; here `D3DLOCK_NOOVERWRITE` semantics apply by default. Matching
  D3D9 means stalling or renaming the backing on every such lock, and a
  dynamic buffer is what a UI or particle batcher locks dozens of times per
  frame. The rename path has been measured peaking near 1.4 GB of retained
  backings, which is why `memory.vbibRetentionCapMB` exists. Site:
  `test_map_synchronisation` below. No knob.
- **A partial `LockRect` of a texture level without `D3DLOCK_NOOVERWRITE` or
  `D3DLOCK_READONLY` returns a pointer into staging an upload may still
  read.** The same trade for a font atlas or lightmap page written a few
  rectangles at a time. A whole-level lock is renamed and its contents
  preserved, because Half-Life 2's lightmap pages rely on that. The partial
  lock and a write into a level the game holds mapped by a lock or a device
  context are the only writes left in place under an upload that still reads
  the level. `UpdateSurface`, `UpdateTexture`, `ColorFill` and `GetDC` move
  the level to fresh pages first, except when every pending upload of it
  belongs to the frame being recorded and no draw or other GPU operation on
  the texture was recorded since, which nothing can observe. Writes into a
  render-target or depth texture and read-backs from the GPU always move it.
  No knob.
- **A DEFAULT-pool `D3DUSAGE_WRITEONLY` static vertex or index buffer keeps
  no CPU copy once every byte has reached the GPU.** D3D9 preserves contents
  across a plain `Lock` whatever the usage says, so a title that reads back
  through the pointer sees zeros, and one that writes past its announced
  window loses those bytes (warned once). Inside a large-address-aware 32-bit
  title those copies measured near a gigabyte of the 4 GiB the title needs
  itself. An indexed triangle fan on a released index buffer copies it back
  off the GPU once, at one mid-frame GPU wait. Knob: `buffer.ignoreLockBounds`
  keeps the copy, default `false`.
- **The window procedure mtld3d subclasses is the device window's, and it
  follows a `Reset` that names another one.** D3D9 subclasses the focus
  window and leaves that subclass where it is across a retarget. Our hook is
  what realizes the D3D9 cursor over the client area and what keeps a windowed
  back buffer matching the client rect, so it belongs on the window the device
  presents into rather than on the one that holds focus. Sites: `test_wndproc`
  and `test_wndproc_windowed` for where the hook sits,
  `test_device_window_reset` for its following a retarget, all below. No knob.
- **A rejected `Reset` leaves render target 0 on the back buffer and the
  depth stencil on the implicit surface.** Like a successful one it releases
  every binding and returns every state to its default before it fails, and
  it leaves the attachments a successful `Reset` leaves. Wine and DXVK unbind
  both before their outstanding-resource check and do not bind them again on
  failure, so `GetRenderTarget(0)` and `GetDepthStencilSurface` answer
  `D3DERR_NOTFOUND` there. What Windows answers is not observable: no d3d9
  test reads the device after a rejected `Reset`, and Wine's d3d8
  `test_reset` keeps its `GetIndices` after one under `if (0)` because it
  crashes on Windows. The layer has no state without render target 0: the
  pass machine, the snapshot and every draw path assume one, so an unbound
  slot 0 would be a new device state for a device that can do nothing but
  `Reset` again. No site observes it. No knob.
- **A texture whose Metal allocation is refused at an extent within the
  reported limits still creates.** `CreateTexture`, `CreateCubeTexture` and
  `CreateVolumeTexture` record the Metal texture's creation into the frame
  and return; the encoder thread makes it before any operation that uses it.
  When Metal returns nil there, the refusal is logged with the descriptor and
  the device's memory figures, the texture samples as opaque black, and the
  uploads into it and draws and copies through it are dropped, each logged.
  Wine and DXVK allocate at the create and answer `E_OUTOFMEMORY` or
  `D3DERR_OUTOFVIDEOMEMORY` when it fails. Answering at the create here would
  make every texture create wait on the encoder thread, which owns the
  texture records, or move the allocation onto the API thread, on the
  streaming path that loads thousands of textures, for a failure Metal
  reports only when the GPU is out of memory. A width or height past
  `MaxTextureWidth` or `MaxTextureHeight` is refused at the create with
  `D3DERR_NOTAVAILABLE`, as Wine refuses a texture past its 2D limit. A
  volume past `MaxVolumeExtent` on any axis is refused the same way, which is
  ours alone: Wine checks no 3D limit and DXVK checks no size, but Metal
  aborts the process on a 3D descriptor past it. No site observes the
  deferred refusal. No knob.
- **`ATI1` creates where no query offers it.** `CreateTexture` makes a 2D
  `ATI1` texture in every pool and `CreateOffscreenPlainSurface` a plain
  surface in every pool it takes, backed by BC4, while `CheckDeviceFormat`
  answers `D3DERR_NOTAVAILABLE` for the format on every resource type. The
  lock of such a level reports the BC4 block pitch, eight bytes per
  four-texel block row, where D3D9 drivers report the format as if it held a
  byte per texel and the application works the block layout out itself.
  Advertising the format would move titles that probe for it, Half-Life 2
  and Team Fortress 2 among them, onto a path no tested title runs; refusing
  the create would end a title that makes one without probing. Reporting the
  D3D9 pitch would need staging sized to that pitch and lock offsets in it
  rather than in the block layout the upload reads, on every lock path.
  Wine's `test_surface_blocks` accepts a create of an extension format the
  queries refuse (`may_succeed`), so no site observes the split. No knob:
  either side of it is a behaviour change for a title, not a trade.
- **`D16_LOCKABLE` and `D32F_LOCKABLE` create where no query offers them.**
  The auto depth-stencil of `CreateDevice` and `Reset`, and a
  `CreateTexture` with `D3DUSAGE_DEPTHSTENCIL`, take either format and serve
  its depth on `Depth32Float`, while `CheckDeviceFormat`,
  `CheckDepthStencilMatch` and a standalone `CreateDepthStencilSurface`
  refuse both. What the formats promise beyond depth is a `LockRect` of the
  depth surface, and the layer keeps no CPU copy of depth to hand back, so
  that lock answers `D3DERR_INVALIDCALL`, warned once. They are
  single-sampled; a multisampled swap chain refuses them, as
  `CheckDeviceMultiSampleType` does, which DXVK refuses too. Wine serves
  both formats and DXVK serves each on some vendors, so a title can ask for
  one as its auto depth-stencil without probing; refusing the create would
  end that title where it otherwise runs with a working depth test and only
  the lock missing. Advertising them would promise the lock. Wine's tests
  probe both formats before using them (`test_clear_different_size_surfaces`
  falls back to D24S8, and the fetch4 and shadow tests skip them), so no
  site observes the split. No knob: the lock a knob would restore does not
  exist.
- **`CreateDevice` and `Reset` substitute a BGRA8 back buffer for any
  format.** The layer presents one drawable format, so a back buffer asked
  for in R5G6B5, A1R5G5B5, X1R5G5B5, A2R10G10B10, a float or any other
  format is created BGRA8, warned once, and reports X8R8G8B8 from
  `GetDesc`; A8R8G8B8 and X8R8G8B8 keep their own. D3D9 refuses a format
  outside its back-buffer set (DXVK answers `D3DERR_INVALIDCALL`), and a
  16-bit back buffer would need a conversion pass on every present.
  `CheckDeviceType` answers only formats inside that set, the 16-bit ones
  where the device renders into them as `test_display_formats` requires, so
  a title that probes lands on a standard format; one that skips the probe
  keeps a device rather than losing it. No knob.
- **`D3DRS_MULTISAMPLEANTIALIAS = FALSE` is ignored.** Metal ties the sample
  count to the pass's attachments with no per-draw override.
  `D3DPRASTERCAPS_MULTISAMPLE_TOGGLE` is not advertised, which is how D3D9
  says the toggle is unavailable, and the first write is logged. No knob.
- **A `D3DSAMP_BORDERCOLOR` other than the three Metal presets reads as
  opaque black.** Under `D3DTADDRESS_BORDER` D3D9 returns the colour the game
  set, and the reference implementations pass it to their samplers. A Metal
  sampler takes only transparent black, opaque black or opaque white
  (`MTLSamplerBorderColor`), so those three map exactly and every other
  colour takes opaque black, logged once per colour. Returning an arbitrary
  colour would take a coordinate test and a select around every sample of a
  stage that addresses by border, carried in each shader as a variant bit and
  a per-stage colour uniform, while the common use, a shadow map whose
  outside reads as lit or as shadowed, needs white or black. Wine's suite
  sets a non-preset border colour only on a cube texture, where the border is
  never read, so no site observes it. No knob: the behaviour a knob would
  restore needs a shader path the layer does not have, so there is nothing
  to trade.
- **The adapter mode list leaves out the display sizes win32u cannot scale
  the monitor to.** After a mode-set, win32u recomputes the monitor's scale
  as `dpi * physical / size` on each axis, reduces it by the greatest common
  divisor of its terms, and packs each term into 16 bits. CrossOver 27's
  win32u asserts that the reduced numerator fits (`make_ratio`,
  `sysparams.c:324`) and aborts the process when it does not; at 96 dpi on a
  3456x2234 display that is 2992x1934, 2992x1870, 2336x1510, 2056x1329,
  2056x1285, 1496x967, 1496x935, 1168x755 and 1168x730, all sizes Wine lists
  itself. A CrossOver 27 build cannot be told apart from here, so the sizes
  are left out on every Wine: `EnumAdapterModes` and the main module's
  `EnumDisplaySettingsW` never offer one, and a fullscreen request for one
  follows the window, as a request for no display mode does, rather than
  setting the mode. The physical size is the largest extent on each axis of
  Win32's mode list, which under `EmulateModeset` is the physical mode, and
  the DPI is `GetSystemDpiForProcess`. The desktop mode always stays, and the
  sizes left out are logged once. No site observes it directly:
  `test_reset_fullscreen` sets the first served size other than the
  desktop's, which was 2992x1934 and aborted the `device` subtest on
  CrossOver 27. No knob: the sizes a knob would restore end the process on
  the Wine that lists them.
- **A windowed device's `SetGammaRamp` changes nothing on screen**, and only
  the implicit swap chain carries a ramp at all. The ramp is stored and
  `GetGammaRamp` reports it back either way, and it starts applying as soon as
  a `Reset` takes the device fullscreen. D3D9 ramps the whole desktop for a
  windowed device, through a path that outlives the process that asked for it;
  that is the display's configuration rather than this window's frame, and the
  present pass is the only place we are willing to carry a ramp. A second
  device, or the compositor, would otherwise inherit a brightness a game set
  for itself. The hardware cursor keeps its own brightness for the same
  reason: macOS composites it and no API ramps it. Wine's suite has no
  `GammaRamp` coverage, so no site observes any of this. No knob: the
  behaviour a knob would restore is one we cannot implement, not a trade.
- **A draw whose shader library or render pipeline is still building can be
  left out of its frame.** D3D9 draws every call in the frame that makes it.
  A first-use Metal compile takes tens of milliseconds, and building it
  inline stalls that frame by as much; building it on a worker thread and
  leaving the draw out until it lands trades one or two frames of a missing
  draw for no stall. A draw is left out only when every colour target its
  pass attaches is the discard-effect back buffer or was cleared whole in
  this frame and the one before, and the depth and stencil planes it tests
  or writes were cleared in both frames too: those are rebuilt every frame,
  so the draw appears once its build lands. One clear does not qualify,
  because a target cleared and drawn once, at load, is cleared exactly when
  its shaders are cold. A draw into any other target, and every draw while
  an occlusion query counts, is kept in its frame, whose submission waits
  for its build, and so is a draw into a target that was copied or sampled
  into kept content in the last 600 frames. Only the first frame of such a
  read can miss a skipped draw (one more per `StretchRect` or mid-frame
  flush on the way to the kept target), and a read-back to system memory,
  such as a screenshot, sees the frame as drawn, skipped draws included. The
  runner pins the knob off, so no site observes it. Knob:
  `shader.asyncCompile`, default `true`.
- **`GetRenderTargetData` from an X render target into its A counterpart
  copies the padding as alpha.** An X8R8G8B8 render target read into an
  A8R8G8B8 system-memory surface (and X8B8G8R8 into A8B8G8R8, X1R5G5B5 into
  A1R5G5B5) is accepted, and the read-back copies the bytes, so the
  destination's alpha holds whatever the source's padding holds where D3D9
  reads an X surface's alpha as one. The colour channels are exact. A
  read-back is one blit of the bytes into the destination's memory, and
  forcing alpha would add a render pass into a scratch texture or a CPU pass
  over every pixel of every read-back, the path an application takes for
  each screenshot and the end-to-end harness for every pixel it checks.
  `StretchRect` from an X surface into an A one does write alpha one. No
  conformance site is known to observe it. No knob: an application reading
  such a surface asked for the colour, and a knob would only add a second
  read-back path to keep in step.
- **The fixed-function specular add clamps before fog.** With
  `D3DRS_SPECULARENABLE` on, the interpolated specular colour joins the
  cascade result after the last texture stage, RGB only, and the sum is
  clamped to [0, 1] before fog blends it with the fog colour. wined3d
  (`ffp_varying_specular * specular_enable + ret`) and DXVK
  (`state.current.xyz += in_Color1.xyz;`) both fog the unclamped sum, so a
  fogged pixel whose diffuse and specular add past one comes out brighter
  there; without fog the render target's clamp makes the two agree on a
  normalized target. The clamp follows the colour-sum stage of
  fixed-function hardware, which OpenGL's fixed pipeline specifies as
  clamped to [0, 1] before fog, and World of Warcraft draws fogged scenes
  through this path with the add switched on, so the reference behaviour is
  not adopted on reference agreement alone.
  What Windows draws is not measured, and no site observes it: Wine's suite
  turns the specular add on only in tests that leave fog off. Reopen on a game
  that shows fogged highlights brighter or darker than on Windows. No knob:
  the alternative is a different shader, not a trade.
- **An extended device copies user memory once.** A `pSharedHandle` on a
  single-level system-memory texture or offscreen plain surface names the
  application's pixels, and D3D9Ex makes that memory the surface's own:
  `LockRect` hands back the application's pointer at its packed pitch, and a
  later write to the memory is a write to the surface. Here the pixels are
  copied into the level once at creation and the level keeps the layer's
  staging, so a lock maps the copy at the 4-byte-aligned pitch every other
  system-memory level has. Wrapping the application's memory would give one
  kind of level two backings, one of them owned by the application and freed
  whenever it likes, on every path that reads a system-memory level
  (`UpdateTexture`, `UpdateSurface`, `GetRenderTargetData`, `GetDC`). The
  one title known to pass user memory, Source on its extended path, fills a
  texture this way, uploads it with `UpdateTexture` and lets it go, which a
  copy serves. Writes through the surface, a `GetDC` included, do not reach
  the application's memory either. The `d3d9ex.c/test_user_memory` and
  `d3d9ex.c/test_user_memory_getdc` clusters carry the sites. No knob.
- **No occlusion or mode-change status on an extended device.** D3D9Ex answers
  `PresentEx` and `CheckDeviceState` with `S_PRESENT_OCCLUDED` while another
  device holds the display exclusively or the window is covered, and with
  `S_PRESENT_MODE_CHANGED` after a mode change. No exclusive mode is ever
  taken here (the device-loss decision), and a present into a covered window
  already skips the drawable without costing the application anything, so
  there is no state to report. The failure latch the layer keeps
  (`docs/STATUS.md`, device loss) is still reported, in the answers D3D9Ex
  documents: `TestCooperativeLevel` answers `D3D_OK` on an extended device
  whatever the latch holds, as the extended runtime does; `Present`,
  `PresentEx`, the swap chain's `Present`, `CheckDeviceState`, `Reset` and
  `ResetEx` return the latched code, with `E_OUTOFMEMORY` reported as
  `D3DERR_OUTOFVIDEOMEMORY`, since `PresentEx` and `CheckDeviceState` document
  the latter and not the former, and neither reset clears it. The latch stands
  for a frame the layer failed to encode or submit, which no `Reset` repairs,
  and it stays one model for both kinds of device: an extended device keeps
  the code a plain one reports, `D3DERR_DEVICELOST`, which `CheckDeviceState`
  documents as a lost device, and `D3DERR_DEVICEREMOVED`, which would send a
  D3D9Ex application to a new device, is not used. No site observes the latch;
  Wine's suite never fails a frame. The `d3d9ex.c/test_lost_device` cluster
  and `d3d9ex.c/test_wndproc` 2957 carry the occlusion sites. No knob.
- **The maximum frame latency is stored, not enforced.**
  `SetMaximumFrameLatency` takes 0 to 30 and `GetMaximumFrameLatency` reports
  what it stored, 3 until then. The frames the application can run ahead are
  bounded by the encoder and submit queues, whose depth is fixed; holding
  the application back further for a lower latency would cost the overlap
  those queues buy. A value below 3 is logged once. No knob.
- **`D3DPRESENT_*` flags are logged, not honoured.** `D3DPRESENT_DONOTWAIT`
  asks a present to fail with `D3DERR_WASSTILLDRAWING` rather than wait for
  the queue, and `D3DPRESENT_FORCEIMMEDIATE` to present without waiting for
  the display; both present here as a flagless present would. The overlay
  flags have no overlay to act on. Each flag is logged once. No knob.
- **`GetDisplayModeEx` reports the identity rotation, and the extended mode
  argument is not used.** The display is never rotated by the layer, so the
  identity is what a game can act on. `CreateDeviceEx` and `ResetEx` take a
  `D3DDISPLAYMODEEX` naming the fullscreen mode; `ResetEx` refuses one that
  disagrees with the request as D3D9Ex does, and both then set the mode the
  back buffer names, as `CreateDevice` and `Reset` do, so the mode's refresh
  rate and scanline ordering go unused. No knob.
- **A maximized window's back buffer follows its client rect.** A windowed
  device, plain or extended, on a `WS_MAXIMIZE` window gets a back buffer of
  the window's client rect rather than the size it asked for: the window
  manager sizes a maximized window, as it does a fullscreen one, and a back
  buffer of another size would be scaled into it at every present. Every
  window the `d3d9ex.c` helper `create_window` makes is maximized, as are the
  windows of a few `device.c` tests such as `test_scissor_size` (the
  `device.c` helper is not), so in those tests a request for a smaller back
  buffer reads the window's size, and a probe at a fixed coordinate reads
  another part of the frame. `device.c/test_scissor_size` carries the plain
  sites, and 26 `d3d9ex.c` sites come from the same rule: `test_user_memory`
  872, the ten of `test_reset`, the ten of `test_reset_ex`,
  `test_backbuffer_resize` 3925/3926 and the three of `test_sysmem_draw`. No
  knob.
- **Every back-buffer index of a swap chain is one surface.** With
  `BackBufferCount` above one, D3D9 keeps a chain of back buffers that
  `Present` rotates, and `GetBackBuffer` hands back a distinct surface for
  each index. Here every index of a swap chain, the implicit one included,
  hands back the same surface object over the one texture the device
  presents from, so a game that writes one index sees it in all of them, and
  a `StretchRect` that stretches between two indices is refused with
  `D3DERR_INVALIDCALL` as a stretch inside one surface, where D3D9 copies it.
  A chain of real buffers would cost a texture per index and a rotation at
  every `Present` that no tested game needs: they draw into the current back
  buffer and present it. `visual.c/test_flip` 22053 and 22064 observe it:
  that test creates three back buffers and reads the shared backing rather
  than a rotating chain (see its cluster below). No knob.
- **An extended device's leave undoes a move another thread makes during the
  mode restore.** Leaving fullscreen, D3D9Ex restores the display mode, then
  the window's style and visibility, and leaves its position and size alone;
  wined3d and DXVK do the same with `SWP_NOMOVE | SWP_NOSIZE`. Under Wine on
  macOS the restore can move the window: AppKit keeps a window it still shows
  below the menu bar when the screen changes, and Wine applies that move to
  the Win32 rect whenever the leaving thread processes driver events, which
  it does while the mode change's `WM_DISPLAYCHANGE` broadcast waits for
  other threads' windows. So the rect is read before the mode restore and
  passed to the final `SetWindowPos`, which on Windows is the same rect,
  unless the window's own procedure moved the window during the restore (a
  game answering `WM_DISPLAYCHANGE`, or the driver's move, by placing its
  window): the device's window subclass sees that move arrive inside a call
  to the application's procedure, and the window stays where the application
  put it. What differs from D3D9 is a move the application makes from
  another thread during the restore, which arrives as the driver's does and
  is undone with it. A driver move the thread applies while the
  application's procedure is itself waiting is kept, as before. No
  conformance site observes either; the end-to-end `d3d9ex_reset` tests pin
  the undone move on a hidden and a shown window and the kept move of the
  window's own procedure. No knob.

## Range-fog coverage

`D3DPRASTERCAPS_FOGRANGE` is advertised. Fixed-function computed vertex fog
uses the transformed vertex's distance from the eye when
`D3DRS_RANGEFOGENABLE` is set, including sequential and indexed vertex
blending. Table fog, pretransformed supplied factors and programmable vertex
fog do not use that switch. Ordinary vertex fog keeps its Z-based formula.

`FOGTABLE` and `WFOG` are also advertised, so Wine's table- and range-fog
visual cases run. Table fog evaluates LINEAR, EXP and EXP2 per pixel, using
Z for an orthographic projection and reciprocal rasterizer W otherwise.
It overrides vertex and range fog and preserves output alpha. SM3 shaders
implement their own fog; their draw keys suppress the automatic fog stage.
The end-to-end tests cover that boundary, shader rebinds and stateblocks
without depending on Wine's capability gates.

## Fill-mode coverage

`D3DFILL_WIREFRAME` uses Metal's native triangle-line fill, preserving triangle
culling and the existing triangle-list conversion for fans. `D3DFILL_POINT`
remains unimplemented: Metal exposes solid and line polygon fill only, and the
layer warns once before drawing solid. D3D9 has no fill-mode capability bit.

The upstream `stateblock.c` covers the solid default, stored wireframe value,
and pixel-stateblock membership (`render_state_indices`,
`render_state_default_data_init`, `render_state_test_data_init` and
`render_state_test_init`). It has no visual fill-mode assertion. The end-to-end
suite covers rendered edges and interiors, state transitions and internal
clear triangles; these capabilities do not imply a baseline count reduction.

## Alpha-to-coverage controls

The explicit ATOC and A2M controls work independently of the reported adapter
vendor. ATOC through `ADAPTIVETESS_Y` requires `ALPHATESTENABLE`; A2M1/A2M0
through `POINTSIZE` set and clear an independent latch. Either request enables
coverage on multisampled RT0. Effective coverage replaces the ordinary alpha
test; a single-sampled target restores it. Disabling one request leaves the
other intact. A2M tokens are not advertised as resource formats.

Numeric POINTSIZE writes and RESZ preserve the latch. A2M and RESZ controls
preserve the last numeric point size, while GetRenderState retains the raw
DWORD. ALL and VERTEX state blocks restore both hidden values; PIXEL blocks
restore neither. Recorded blocks retain which components their writes touched
across Capture: refreshing a raw DWORD cannot add or remove latch or numeric
size membership. Reset restores the default numeric size and clears A2M.

This is a compatibility policy for sequences on which the references differ.
WineD3D keeps an independent latch, but retains the ordinary alpha test and
reads control DWORDs as numeric point size. DXVK preserves numeric size for
controls and suppresses alpha testing under coverage, but derives its request
from raw POINTSIZE on selected invalidations. mtld3d combines independent
latch lifetime with preserved numeric size and alpha-test suppression, avoiding
behavior that depends on which later state update refreshed a cached request.
Complete native AMD behavior for these mixed sequences has not been measured.
The local sequence tests define this policy; Wine's AMD visual branch covers
only enable, disable and VERTEX-state-block restoration.

## Fetch4 state-block membership

The Fetch4 latch that `D3DSAMP_MIPMAPLODBIAS` GET4 and GET1 commands set and
clear is a hidden component of the sampler state, beside the raw DWORD that
`GetSamplerState` returns. ALL and PIXEL state blocks capture and restore the
latch of every pixel sampler with the raw value; VERTEX blocks restore
neither. Recorded blocks follow the A2M rule: a recorded GET4 or GET1 carries
the latch, which a Capture refreshes from the device, while a recorded numeric
bias carries none, and a Capture cannot add it. Reset clears the latch.

This is a compatibility policy. Wine has no Fetch4. DXVK captures only the raw
DWORD and decodes it again on Apply, so a block captured while a numeric bias
follows GET4 does not restore the latch there. mtld3d restores the latch the
block saw, for the same reason as the A2M latch: the result does not depend on
which bias write came last. Native AMD behavior for these sequences has not
been measured; `samplers.rs` defines the policy.

## Dynamic depth texture coverage

Wine's resource-access test creates a DEFAULT-pool, `D3DUSAGE_DYNAMIC` depth
texture and uses the plain texture probe to derive the expected HRESULT.
Dynamic D16, D24X8 and D24S8 2D textures now support packed locks and explicit
mip chains.
Dynamic D32 and legacy, lockable or vendor formats remain unavailable. Depth
attachments with DYNAMIC, CPU pools, plain depth surfaces, depth cubes and
volumes remain unavailable. AUTOGENMIPMAP queries return NOAUTOGEN and
creation falls back to one level, without enabling automatic generation.

RESZ into a dynamic destination copies depth and the common stencil plane,
using sample zero for multisampled sources. A depth-only source preserves
the destination stencil. This follows DXVK's common-aspect ResolveZ contract
and Wine's plane copy behavior; native AMD documentation only establishes
depth sample zero. Scaled attachments are resampled on the GPU into the
logical-sized destination. Readback and packing wait until a preservation
or READONLY lock needs the GPU-authoritative bytes; a whole DISCARD skips
that readback and versions any destination still named by pending work.
The readback counts as the level's initial upload, so a READONLY lock of a
level only the GPU wrote publishes nothing and the texture keeps the
transferred depth rather than its packed code.

A multisampled RESZ into any other destination, and a depth-to-depth
`StretchRect` out of a multisampled surface, take the same transfer: sample
zero of depth and of the common stencil plane, written by compute and blit
work. Neither uses a render-pass depth resolve, whose destination the Intel CI
image's paravirtual device leaves reading what an earlier pass stored there
(resz_test 17724 and 17862 read the INTZ's clear through one). Every transfer
lands a clear still waiting for a pass first, so a `Clear` issued just before
it is what it reads.

Both architectures and all local variants pass device.c:13838. Hosted Mac2
recordings also pass on both architectures; its obsolete pins are removed.

## Managed 2D dirty publication coverage

Managed 2D textures retain their sampled image after a NO_DIRTY_UPDATE lock
(19210/19217) until an explicit publication. AddDirtyRect publishes its region
across the application mip levels (19232), using the existing partial-upload
path. Initial contents and eviction still publish the CPU image. READONLY
adds no publication after initialization.

The earlier eager-publication rationale no longer applies to this path.
Separate policies remain: mip staging is independently allocated, dirty
rectangles union to one bounding box per mip, and overlapping partial locks
may modify bytes an earlier queued upload still reads. This change neither
adds exact disjoint-region tracking nor changes that partial-lock policy.

Sites 19210, 19217 and 19232 pass on both architectures in the local normal,
Intel and scale variants and in both hosted Mac2 recordings. Their obsolete
baseline pins are removed.

## Premultiplied texture-alpha blending

`BLENDTEXTUREALPHAPM` is advertised and emits
`saturate(arg1 + arg2 * (1 - texture_alpha))` for color and alpha operations.
Wine's `visual.c/texop_test` exercises its previously skipped row. The direct
E2E also checks implicit texture-alpha use with nontexture arguments,
argument modifiers and saturation before a following texture stage.

An implicit-only missing texture uses zero alpha, following the native
observation recorded in DXVK commit `0b49a39896f25896b83ed01c0609393dfc3bb85c`.
This is a reference choice, not a new native measurement: Wine's D3D9 GL
dummy texture uses alpha one. The ordinary `BLENDTEXTUREALPHA` follows the
same rule, so with no texture bound and no argument naming it the stage
yields its second argument, for colour and alpha alike; wined3d yields the
first. Existing explicit-unbound-argument handling remains unchanged.

## What the baseline records — and where classes live

Each datum has exactly one authoritative home, split by who writes it:

- **`baseline.txt` (machine-owned)** records the current *results*: for each
  `(arch, subtest)`, the crash bit plus every failing assertion as a
  `<file>.c:<line>` site with a hit count. No classifications — the file is
  freely rewritten by `make conformance-baseline` without ever touching prose.

  ```
  [i686/device] crash=1
    device.c:125 count=28
    ...
  ```

- **This document (human-owned)** records *why* each site fails: the
  per-cluster section below declares every site's classification as a
  `<line>=<class>` token on a `Sites:` line, next to the rationale prose.
  The runner loads classes from here (for the flaky/ceiling tolerances and
  untriaged reporting); a unit test in the runner crate fails `make test` unless the
  two files cover exactly the same sites — so a new baseline site stays loud
  until someone writes its rationale, and a fixed site's prose must be
  removed rather than lingering as history.

Per-site granularity is what makes the score actionable: the 74 `device`
failures, for instance, are really three source lines hit repeatedly in a loop,
not 74 distinct defects. Recording the location (not just a total) means a fixed
bug and a new regression can no longer net out to the same number and hide each
other.

`make conformance-baseline` re-records `baseline.txt` and prints exactly which
sites are new (add them to a cluster below, with a rationale) and which were
dropped (delete their tokens and trim the prose). A run whose Wine version
differs from the baseline's recorded version warns that `file:line` sites may
have drifted (a Wine update renumbers source lines) and a re-baseline is
expected — the `Sites:` tokens here renumber with it.

## Classification tags

Each failing site carries one tag. The tag is a deterministic property of the
divergence's NATURE — never of fixability, difficulty, or in-game value (a
hard-to-fix or low-value defect is still `real`):

- **`real`** — a genuine defect we intend to fix: our output/behavior is wrong
  and no deliberate design rationale covers the divergence. A mixed line (any
  intend-to-fix component alongside by-design assertions) is `real`, with the
  remainder explained in prose.
- **`caps`** — the failure exists only because the test assumes a capability we
  deliberately don't advertise, AND our actual behavior is the conformant
  response for a device without that capability (correct pixels, or the
  spec-correct rejection). A cap-*respecting* test simply passes and never
  lands here; `caps` covers cap-*blind* assertions (Wine's tests assume caps
  that real desktop drivers always have) and escapes offered only under
  `broken()`, which the runner does not honor. If our response to the missing
  capability is itself non-conformant, the site is `real`.
- **`expected`** — we deliberately do not implement this and intend to keep it
  that way, for a positive, documented reason: a scope decision (device loss,
  desktop mode switching, the D3D9-on-D3D12 bridge — see below), a kept perf
  tradeoff (the TBDR depth-store elision, buffer-rename over stalls), or an
  accepted platform limitation (Metal's 0xffff primitive restart, GPU-defined
  NaN encodings). "We don't want to fix it" or "the fix is invasive" is not a
  rationale — without a positive reason to keep the divergence, the site is
  `real`.
- **`flaky`** — environmental/non-deterministic (display config, Retina scale,
  macdrv window-manager timing). Count changes in either direction never gate.
  Tag reactively — only once a flutter actually trips the gate — and pin the
  HIGHER observed count so a flutter back up is not a false regression.
- **`ceiling`**: the pinned count is a MAXIMUM, not an exact value, in two
  cases. First, the same baseline serves environments where the site
  legitimately reads lower (a CI runner's virtual display accepts the mode
  changes this machine's macdrv rejects, so the desktop-mode sites read zero
  there; the fetch4 counts wobble with the attached display). Second, the
  site's count flaps below the pin on one environment, and only a read above
  the pin would mean something (test_wndproc 4319, and 4302/4328/4329).
  Reading below the pin is tolerated and does not demand a re-record; reading
  above it gates like any regression. The tag adds only that tolerance; the
  divergence's nature stays in the cluster prose. Like `flaky` it is assigned
  reactively, from a measured cross-environment delta or after the flap
  tripped the gate, never speculatively. For a flapping site, choose
  `ceiling` over `flaky` when a count above the highest observed one would
  be a real failure (for instance every iteration of a loop failing), and
  `flaky` only when the count carries no meaning in either direction.
- **`crash`** — a site attributed to a crash/abort path.
- **`untriaged`** — an explicit placeholder for a site a human has not yet
  triaged. Normally untriaged means *absent from this document* (the sync test
  stays red until prose exists); writing `=untriaged` is the escape hatch for
  landing a re-baseline before the triage is done — the runner still flags it
  on every run.

The counts are the signal, not a target of zero. Wine's `todo_wine`/`broken()`
annotations are tuned for a real-GPU driver, not for us, so a raw failure is
not necessarily a real defect — the classification is what turns the number into
something actionable. Note that when a subtest crashes, the counts cover only the
failures reached *before* the crash truncated the run.

## Per-cluster classification

This section is the authoritative home of every failing site's classification
and rationale, grouped by enclosing Wine test function. The classes exist only
here (`baseline.txt` holds counts); the runner loads them at gate time, and a
unit test in the runner crate fails if any baseline site has no `Sites:` entry
below, any entry names a site that no longer fails, or a site is declared
twice. A second test recounts those tokens against the "Current
classifications" sentence below and fails with the sentence to copy in when
they disagree. When a re-baseline adds or removes sites, update the matching
cluster block (and its rationale) in the same commit.

Line numbers refer to the Wine version recorded in the baseline header. A
`Sites:` line lists every baseline site of the cluster as `<line>=<class>`;
prose explains why. One source line can fire many assertions and can mix
sub-causes — a `real` line may carry a by-design remainder (noted in prose),
per the mixed-site rule: if any intend-to-fix component remains on a line,
the line is `real`.

Audit provenance: every cluster below was re-derived on 2026-07-20 from the
Wine test source, the raw actual-vs-expected failure messages
(`MTLD3D_CONFORMANCE_RAW_DIR`), and the implementation — independently
re-checked before retagging. Current classifications, counted from the
`Sites:` tokens below on 2026-10-07: 0 `real`, 182 `expected`, 2 `caps`,
25 `ceiling`, 3 `flaky`, 0 `untriaged`, 212 unique sites in all.
The audit recorded all 24 Apple-family subtest-legs `crash=0`.
(2026-10-07: with D3D9Ex implemented the `d3d9ex` subtest creates its
extended devices and runs every test, adding the same 64 sites, 63
`expected` and 1 `caps`, on each of the six Apple-family legs. The site set
held over 20 isolated runs per architecture with no crash; that repeat ran on
the native legs only, not the `+intel` caps legs. The Intel CI dispatch on
the branch recorded both `@mac2` `d3d9ex` legs with the same 64 sites at the
same counts, so none of them is `@mac2`-only.)
(2026-10-05: test_wndproc 4302 moved from `expected` to `ceiling`, and
4328/4329 joined it, pinned at one on the `i686` and `i686+intel` device
legs; the cluster says why.)
(2026-09-30: the adapter mode table stopped leaving out sizes more than 15 %
from the desktop's aspect, which made 640x480 a settable mode on the Intel CI
image: its display runs 3840x2160 and user32 lists 640x480 for it, so a
fullscreen request for that size used to follow the window and now sets the
mode. The six `test_get_display_mode` sites 14378 to 14391 and the four
`test_multisample_get_front_buffer_data` sites 17167 to 17181, which asked
for a 640x480 fullscreen device and read 3200x1800 back or had a 640x480
read-back refused, left the `@mac2` legs and their clusters left this
document. That display lists 25 sizes; it is not single-mode as the
2026-09-05 note below says.)
(2026-09-05: the two answers a device
without the packed 16-bit formats derives from its render-target answer,
`CheckDeviceType` for a 16-bit back buffer and the `AUTOGENMIPMAP` probe, were
made to follow it, so device.c:3626 and device.c:7927 dropped off the Intel
legs and their clusters left the audit; `ValidateDevice` learnt the sampler
filter rules the same day and device.c:8181 went with them.)
(2026-09-05: the `scale` legs added 32
visual.c sites in eleven clusters, every one `expected` for the one reason
"The scaled leg" below gives; they carry no device.c site of their own since
#408. The `resz_test` site they also carried, 17946, was `real` and is fixed:
a lockable render target created at the reported back-buffer size declined the
scale while the depth-stencil bound with it took it, issue #407.) (2026-09-05: the `@mac2` legs, recorded
on the Intel CI image, added 22 sites, every one `expected` and every one a
property of that machine, its single-mode display or the GPU family, in the
`test_mode_change`, `test_get_display_mode`, swizzle-format, `test_fetch4`,
`fp_special_test`, `test_multisample_get_front_buffer_data` and
`multisampled_depth_buffer_test` clusters; the eight `@mac2` subtest-legs are
`crash=0` too.) (2026-09-04: the Intel legs, which run every subtest
under the `intel.*` config keys, added device.c:3626, 7927 and 8181 as `real`
(issues #362, #363; all three are fixed and gone) and visual.c:28024 as
`caps`; no site moved under the two keys that change only a code path,
`intel.managedMemory` and `intel.linearAlign256`. On the Apple family the
`ceiling` and `flaky` pins of the native device legs are carried on the Intel
legs at the same counts, since the environment they depend on is the same; the
`@mac2` legs, recorded on the Intel CI image, carry their own counts, see the
family paragraph under "Running".) (2026-08-27: device.c:15088 moved from `expected`
to `ceiling`, it fires only where the Wine build ships a loadable d3d12.dll;
the SRGBTEXTURE decode landing the same day changed no site counts — the
newly-running `srgbtexture_test` passes. 2026-08-28: honouring
`D3DCREATE_NOWINDOWCHANGES` dropped test_window_style 5215, and test_wndproc
4551 was re-derived from the raw capture and corrected from `real` to
`expected`. Multisampling then moved the counts in both directions. Four
`device.c/test_reset` sites now pass, because a fullscreen `Reset` with a
zeroed `D3DPRESENT_PARAMETERS` is rejected for its `D3DFMT_UNKNOWN`
back-buffer format. Seven `visual.c` tests stopped skipping (every one of
them gates on `CheckDeviceMultiSampleType`) and five of the seven pass
outright; the clusters below cover what the other two and the tests they
unblocked leave failing. The four `real` sites they added are since fixed,
test_multisample_get_front_buffer_data 17179 and 17181 by the system-memory
read-back destinations and resz_test 17724 and 17862 by the RESZ depth
resolve, and multisampled_depth_buffer_test 17476 went with them once the
depth-to-depth `StretchRect` resolved a multisampled source, so its cluster
leaves this document too. 2026-09-06: a fullscreen `Reset` that names another
device window hands the session over instead of giving the old window back,
so test_device_window_reset 5968 passes and that cluster leaves the document
as well; retargeting the Metal layer and the cursor subclass onto that window
in the same `Reset` then added 5975 and 5978, the same window-procedure
decision test_wndproc already records, and the cluster comes back for them.
2026-09-24: both multisampled depth resolves (RESZ and the depth-to-depth
`StretchRect`) copy through a depth transfer instead of a render pass with a
depth resolve attachment, so the paravirtual device's resolve fault no longer
reaches multisampled_depth_buffer_test and its `@mac2` cluster (17330, 17476)
leaves the document; test_multisample_mismatch 20959 and 20962 pass on the
`@mac2` legs as well.) Only two tags change what the gate tolerates:
`flaky` (count changes in either direction) and `ceiling` (reads below the
pin). Every other tag is documentation, so a correction between `real`,
`expected` and `caps` is never a gate change.

#### Desktop mode switching, and how fullscreen honors the requested size

A fullscreen device sets the display mode the app asked for through user32,
as native does, then takes a borderless window over the monitor. The
mode-set is meant to stay virtual: with Wine's `EmulateModeset` on (the
harness pins it, and so does the launcher) win32u leaves the physical display
alone, answers the desktop mode, `GetSystemMetrics`, `GetMonitorInfo`, the
client rect and every mouse coordinate in the mode, and scales the window
onto the physical monitor; without it the mac driver would hand the change to
`CGDisplaySetDisplayMode` and switch the whole desktop. The device leaves the
z-order alone: raising the window to the topmost level deadlocks winemac (see
test_window_style 5220).

The mode list `EnumAdapterModes` serves is a bounded subset of
`EnumDisplaySettingsW`'s (the sizes that fill the display, largest first,
which are those win32u's uniform scale onto the physical display leaves a bar
of less than one physical pixel for, then the standard sizes of another shape
that user32 lists, largest first: 2560x1440, 1920x1080, 1600x900, 1280x720,
1024x768, 800x600 and 640x480, which win32u letterboxes), so an enumerated
mode is one win32u accepts by construction, and a fullscreen request for any
mode in the full list is set whether or not the bounded list carries it. Both
lists leave out the sizes win32u cannot scale the monitor to (see Kept
divergences). The test binary, being the process's main module, enumerates the
same sizes through its own `EnumDisplaySettingsW` import, in user32's order
rather than the served one (d3d9 redirects it at load; user32's list is
untouched and `ENUM_CURRENT_SETTINGS` passes through), so a mode the test
picks from either list is one user32 accepts. When the app requests one, the
device sets it and the back buffer is that mode; present scales it to the
drawable, which stays at the display's size (MetalFX when enlarging, the same
resample `render.scale` rides). Both halves of the contract then agree with
the size the app rendered for: the default viewport and scissor, the reported
present parameters, the device's and swap chain's `GetDisplayMode`, and the
Win32 metrics and mouse. (Until 2026-08 the back buffer honored the mode under
a monitor-sized window, which kept the D3D9 half right and left mouse input in
monitor space; before that it followed the window and apps that sized their
viewport from their own request rendered into a corner.) A request that
matches no mode user32 accepts still follows the window: native would reject
it, so nothing can depend on it being honored, and the apps that make such
requests (WoW's windowed-to-fullscreen toggle carries its window size) size
their rendering and mouse handling from the window, so the window-sized back
buffer is the assignment that keeps them consistent. We still do not reject
such a request, which is the one `expected` site left in this area.

The focus half follows native too: `WM_ACTIVATEAPP FALSE` puts the registry
mode back and `WM_ACTIVATEAPP TRUE` sets the mode and re-covers the monitor
again; the window is never minimised and the device is never lost. The
harness pins emulated mode switching and Retina mode so window-management
assertions use a stable physical-pixel coordinate space.

Both pins are registry values, and a wineserver session enumerates the display
once, when its desktop starts, and serves that geometry to every process in it
afterwards. A pin therefore only takes effect in a session that started after
it was written, and the session that creates a prefix predates them by
construction, so `configure-test-prefix` ends that session once the keys are
in. Without it the first leg in a fresh prefix runs against monitor geometry
in the point space: `test_window_position` 15023 and `test_reset_fullscreen`
4903 fail outright, and the desktop-mode `ceiling` sites read 0 because the
mode change the test asks for is accepted, exactly as on the CI runner's
virtual display.

Two refinements landed 2026-08 after the CI runner exposed them (its virtual
display accepts the mode changes this machine's macdrv rejects, so the tests
walk further):

- **One source of display truth.** `EnumAdapterModes` /
  `GetAdapterDisplayMode` come from `EnumDisplaySettingsW`, the same view
  win32u validates `ChangeDisplaySettingsW` against and derives
  `GetMonitorInfoW` from, instead of `NSScreen`: the current mode for
  `GetAdapterDisplayMode` (read live, so it follows a mode-set), a bounded
  subset of the enumerated list for `EnumAdapterModes`, the sizes that fill
  the display and then the standard ones (so a mode a game picks is one user32
  accepts, the sizes that fill the display come first, and a menu built for a
  driver's short list does not overflow). On this machine the two views agree
  under the pinned Retina mode; on the runner's virtual display they disagreed
  by exactly 2x (Win32 2048x1536, `NSScreen` 1024x768), which split
  `GetDisplayMode` from the monitor rect (test_get_display_mode 14472/14474)
  and fed the tests modes that user32 then refused. Seeding the list from
  user32 is also what made the tests' own `ChangeDisplaySettingsW` calls
  succeed here (test_wndproc 4161/4231, test_reset 2234-2238,
  test_mode_change), since they pick their mode from `EnumAdapterModes`.
- **The mode contract.** A fullscreen device sets the requested mode
  (2026-08), so the Win32 half of the contract holds: the desktop mode
  follows a create or Reset, `GetSystemMetrics` and the window rect report
  it, and the registry mode comes back when the device loses focus
  (`WM_ACTIVATEAPP FALSE`), leaves fullscreen (windowed `Reset`, final
  release) or the process exits. Where we diverge from native on purpose:
  the mode is set again on `WM_ACTIVATEAPP TRUE` rather than at the app's
  next `Reset`, because the device is never reported lost and so nothing
  would prompt that `Reset` (test_wndproc 4302).

#### The scaled leg

The `scale` legs run every subtest at `render.scale = 0.75`. For the tests'
640x480 device that rasterizes the back buffer, and every render target or
depth buffer the test creates at that size, on a 480x360 grid, and every
pixel the test reads comes back through the readback resolve, a filtered
copy of the render grid up to the reported size (the display path's
`MetalFX` scaler is not used for a readback: it writes an opaque alpha and
takes only a few formats). Every coordinate the suite hands in is converted
between the two spaces on the way in, so a probe several pixels clear of a
colour boundary reads exactly what was rasterized at any scale, which is why
the whole end-to-end suite holds at 0.75. What cannot survive the pair is a
probe within a pixel of a boundary, a one-pixel feature, or a target too
small for any pixel of it to be interior: those read the blend the resolve
leaves, and the tell is a channel at one eighth or seven eighths of the
neighbour (`0x20`, `0xdf`) or within a step of it (`0x04`, `0xfb`). A site
of that shape is `expected`: the space separation is the design, and a probe
on a boundary has no exact answer under a resample. A CPU write into a scaled
target (an `UpdateSurface` into a render-target surface or the back buffer, or
any CPU write into a level of a render-target texture) converts its rect the
same way and resamples only the rows inside it, so a render texel the rect's
reported edge cuts takes the written colour or keeps the old one whole: the
written edge moves by under half a render texel, and a probe on it is the same
boundary case. A site whose values do
not fit that mechanism is `real`, exactly as on any other leg; the legs carry
none today.

### The `real` backlog

Empty. No failing site is classified `real` on any leg.

Every other failing site is a recorded decision (`expected`), a capability we
do not advertise (`caps`), a pin above the usual read, because the site
reads lower on other hardware or flaps below it here (`ceiling`), or a known
flap (`flaky`), each with its rationale in the
per-cluster section below.

The `device` subtest used to die silently inside test_volume_get_container
(a `GetContainer` that answered E_NOINTERFACE with a null container, which
the test then released), and the baseline recorded before the runner learnt
to treat a missing end-of-run summary as a crash carried only the sites
before that point. Every cluster from test_occlusion_query on was
re-triaged when the run first reached its end again.

Vertex streams 1..15 and `SetStreamSourceFreq` instancing are implemented, so
the clusters that used to sit on "single-stream rendering" (stream_test,
fixed_function_decl_test, the stream-1 half of test_sysmem_draw, and the
state-block stream capture in resource_check_data) no longer appear in the
baseline.

### device.c clusters

### device.c/test_wndproc
Sites: 4207=expected 4212=expected 4214=expected 4219=expected
Sites: 4223=expected 4248=expected 4257=expected 4293=expected
Sites: 4298=expected 4302=ceiling 4319=ceiling 4328=ceiling 4329=ceiling
Sites: 4340=expected 4420=expected
Sites: 4424=expected 4432=expected 4487=expected 4525=expected 4545=expected
Sites: 4572=expected 4161=ceiling 4231=ceiling 4551=expected 4475=flaky
Sites: 4480=flaky

4161/4231 are the test's own `ChangeDisplaySettingsW(CDS_FULLSCREEN)` call,
before any D3D9 object is involved; they read zero now that the mode the
test picks from `EnumAdapterModes` is one user32 accepts, and stay `ceiling`
pins from when it was not. The rest of the fullscreen focus lifecycle we
deliberately do not drive: no focus/foreground mutation (4212/4214), no focus-
window subclass (4223/4572), no WM_* activation/mode message generation
(4207/4248/4293/4319/4340/4432/4525/4545), no focus-window minimize
(4420), device-never-lost TestCooperativeLevel (4257/4298/4424/4487).
4302 (both iterations) expects the desktop still at the registry mode after
the app is re-activated, native leaving the mode-set to the app's next
`Reset`; we set the device's mode again on `WM_ACTIVATEAPP TRUE`, because a
device that is never lost gives the app no reason to `Reset`.
Caveat on 4219: it fails because OUR cursor wndproc subclass replaced the
device window's proc — a deliberate, load-bearing hook we keep (cursor
realization), not a missing feature.

4319 checks the hidden-device-window focus-loss message sequence immediately
after `SetForegroundWindow(GetDesktopWindow())`: `WM_DISPLAYCHANGE` on the
device window, then `WM_ACTIVATEAPP(FALSE)` on the focus window. It runs once
with ordinary creation flags and once with `D3DCREATE_NOWINDOWCHANGES`, so
the assertion can fail at most twice. An unchanged i686 build read one
failure and then two; a clean control also read two. Both count-two raw
captures lack the focus window's `WM_ACTIVATEAPP(FALSE)` in both iterations.
The count-one run has no raw capture, so which iteration differed and why
remain unknown. `ceiling` retains the pin of two while tolerating lower
message-observation counts; unlike `flaky`, it still rejects counts above
the pin. This changes no focus or activation behavior and does not establish
that the message contract is fixed.

4302, 4328 and 4329 move together now and then on i686: in one iteration
4302 passes, and 4328/4329, the desktop mode read back after the hidden
device window loses focus, fail, so the device reads `4302 2 -> 1`,
`4328 0 -> 1` and `4329 0 -> 1`. That happened on 2026-09-13 and again on
2026-10-04, the second time with no other Wine session on the machine, in
one of about a hundred local i686 device runs between 2026-10-01 and
2026-10-05. None of the 985 i686 device legs CI ran between 2026-09-21 and
2026-10-05 showed it, and neither local run kept a raw capture. The reading
the counts allow, which is conjecture: a deactivation that macdrv delivers
late restores the registry mode before 4302 reads it, the app is then
inactive when the test's `Reset` sets the device's mode again, and the
second `SetForegroundWindow(GetDesktopWindow())` changes no activation, so
no `WM_ACTIVATEAPP(FALSE)` arrives to restore it. 4302 stays the decision
recorded above; `ceiling` keeps its pin of two and tolerates one. 4328 and
4329 are pinned at one on the two i686 device legs recorded on this machine
(`i686`, and `i686+intel`, which carries the native leg's pins);
`i686+scale` is recorded on CI, where they read zero, and a first read there
or on any other leg is a new site. A read of two, both iterations failing to
restore the registry mode on focus loss, still fails the gate. A re-record
on a run where they read zero drops the 4328/4329 pins from both legs, and
they must be put back by hand.

4257/4298/4424/4487 are the kept device-loss divergence, not an unwritten
stub: no exclusive mode is ever taken, so nothing is ever lost, and
`TestCooperativeLevel` answers `D3D_OK` across a focus change. The
transition is detectable (the device window's subclass already handles
`WM_ACTIVATEAPP` for the registry-mode restore), so this is a decision
rather than a gap: reporting a loss that did not happen sends every
fullscreen game through releasing and rebuilding its whole `D3DPOOL_DEFAULT`
working set on each activation change, which costs frame time and risks the
game's own recreate path, for a device that lost nothing. The
`D3DERR_DEVICENOTRESET` half is real and implemented: a failed `Reset`
latches it until one succeeds. Listed under Deliberately not implemented
in `docs/STATUS.md`.

4551 is `expected`, and follows from the same no-modeset decision as the
message sites above. Both mode-change `Reset`s expect the device window to be
moved: `mode_change_messages` and `mode_change_messages_hidden` each start
with a `WM_WINDOWPOSCHANGING` for it, and 4525 and 4545 read `Expected
message 0x46`, the walk stopped on that first entry. A fullscreen
mode-change `Reset` resizes the back buffer, not the window, which already
covers the monitor and keeps covering it, so the device window gets no
position message at all. 4551 reads the `WINDOWPOS` the test's wndproc
captures only when the walk reaches the fifth entry of
`mode_change_messages_hidden`, the second `WM_WINDOWPOSCHANGING`; with the
walk stopped on the first entry the capture stays zeroed and the assertion
compares against a null HWND. `d3d9ex.c` 3203 and 3209 are the same two
reads in the extended copy of the test. Reaching the capture needs the
`Reset` to move the window, which here only a real mode-set would do, so the
line moves only with that decision.

4475/4480 are flaky macdrv window-message timing sites;
mtld3d does not call `SetWindowPos` or `MoveWindow` on those paths.

### device.c/test_reset
Sites: 2234=ceiling 2237=ceiling 2238=ceiling 2250=ceiling
Sites: 2251=ceiling

The fullscreen half of this cluster passes since a fullscreen device sets
the requested mode: the request-side assertions (the default viewport
matching the request at 2133/2134 and 2172/2173, `GetPresentParameters`
reporting it at 2187/2189) and the Win32 half (2126/2127, 2179/2180,
2250/2251 reading the mode back from `GetSystemMetrics(SM_CXSCREEN)`).
2234/2237/2238 are the test's own `ChangeDisplaySettingsW` call, before any
D3D9 object is involved; it succeeds now that the mode it picks from
`EnumAdapterModes` is one user32 accepts. All five stay `ceiling` pins from
when they failed here.

The fullscreen Resets to a mode user32 rejects (32x32, 801x600) return
INVALIDCALL, for a reason that has nothing to do with the resolution: each
zeroes its whole `D3DPRESENT_PARAMETERS`, so `BackBufferFormat` is
`D3DFMT_UNKNOWN`, which a fullscreen Reset has to reject. The resolution
itself is not validated against the mode list; such a request follows the
window instead.

The windowed API contract in this test passes: Reset rejects an outstanding
app reference to a DEFAULT-pool resource or an implicit surface, and a
failed Reset latches DEVICENOTRESET until one succeeds.

### device.c/test_scissor_size
Sites: 3685=expected 3700=expected

The default scissor rect must equal the back buffer the app asked for, both
after create (3685) and after a Reset (3700). Every window in this test is
created `WS_MAXIMIZE`, and a maximized window is sized by the window manager
rather than the app, so we take its client rect and ignore the requested size
— the same rule as fullscreen, for the same reason. The scissor itself is
correct: it matches the back buffer we actually created.

Note 3700 additionally expects the *full screen* size, while a maximized
window's client rect is the work area (screen minus menu bar and Dock), so
this line would differ even if the create path honoured the request.

### device.c/test_wndproc_windowed
Sites: 4681=expected 4697=expected 4701=expected 4708=expected 4751=expected
Sites: 4774=expected 4778=expected 4785=expected

4701/4778 expect the focus window subclassed in fullscreen (we don't).
The other six expect the device window's wndproc UNCHANGED and fail because
of our cursor subclass — the same deliberate hook as test_wndproc 4219,
kept on purpose (cursor realization is driven from it).

### device.c/test_reset_fullscreen
Sites: 4871=expected

WM_ACTIVATEAPP delivery on a windowed→fullscreen Reset; we do not
force-show/activate the window.

### device.c/test_fpu_setup
Sites: 5041=expected 5051=expected

i686 only. Native D3D9 rewrites the x87 control word to single precision
(0x7f) at device creation and keeps it for callbacks; we deliberately never
touch the FPU control word. On x86_64 the same checks are todo_wine (free).

### device.c/test_window_style
Sites: 5220=expected

5220 is `expected`: the fullscreen extended style must carry `WS_EX_TOPMOST`.
We deliberately leave the z-order alone, because raising a window to the
topmost level makes Wine's mac driver re-derive the Cocoa window's level and
parent while holding winemac's per-window lock and hop to the main thread to
do it; a focus event arriving meanwhile re-enters `NtUserSetWindowPos` on
another thread and the process deadlocks. Reproduced in the `visual` subtest.
A borderless window covering the monitor already presents as fullscreen, so
the z-order buys nothing. (5200, the window-rect adoption, now passes.)

5215 passes since `D3DCREATE_NOWINDOWCHANGES` is honoured: a device created
with that flag leaves the device window's style, rect and visibility to the
app, in fullscreen as in windowed mode, so a window the app kept hidden is
still hidden after the fullscreen round trip. The three todo_wine lines that
the flag also covers (5179/5197/5238) now succeed inside their todo blocks,
which the runner does not count.

### device.c/test_mode_change
Sites: 5509=ceiling 5533=ceiling 5537=ceiling 5584=ceiling
Sites: 5602=ceiling 5622=ceiling 5636=ceiling 5639=ceiling 5646=ceiling
Sites: 5671=ceiling 5674=ceiling
Sites: 5598=expected 5618=expected 5641=expected 5676=expected 5900=expected

Desktop display-mode-change lifecycle (`ChangeDisplaySettingsW` success,
`EnumDisplaySettings` reflecting changes/restores, fullscreen window resize).
The whole cluster passes since a fullscreen device sets and restores the
mode and the test's own CDS calls pick a mode user32 accepts; the `ceiling`
pins date from when physical mode switching was disabled and they failed
here while reading zero on a CI runner. 5552/5554 (the back buffer must keep
the size a fullscreen create asked for across an external mode change) pass
because the back buffer honors the request and never follows the window.
5598/5618/5641/5676/5900 ("Failed to restore display modes") fire on the
`@mac2` legs only, where the test's restore of the original display modes
fails. Why it fails there is not established: the Intel CI image's display
runs 3840x2160 at 30 Hz and user32 lists 25 sizes for it, so it is not a
single-mode display. Desktop mode switching is out of scope, so `expected`.

### device.c/test_device_window_reset
Sites: 5975=expected 5978=expected

Both read the window procedures after a `Reset` that retargets a fullscreen
device from the focus window onto a separate device window: the device
window's must be the application's own (5975) and the focus window's must
still be D3D9's (5978). Native subclasses the focus window and leaves it
there across the retarget. Ours is the cursor subclass, and it follows the
window the device presents into, because that is the window the D3D9 cursor
is realized over and the one whose client area the back buffer follows. Both
halves of that decision are already recorded on test_wndproc 4219/4223/4572
and test_wndproc_windowed 4701/4778: our hook sits on the device window and
never on the focus window. These two sites are the same decision seen through
the retarget, so they move only if the hook moves.

### device.c/test_occlusion_query
Sites: 6780=expected

The >2^32-sample query (65 fullscreen 8192x8192 quads under one query, depth
test off) undercounts because Apple's TBDR hidden-surface removal merges
same-encoder opaque overdraw before fragment processing: the visibility
counter reports the samples that *survive* HSR, not every sample that would
have passed the depth test on an immediate-mode GPU. Proven by
instrumentation, not inferred: the GPU-written slot value itself is short
(our BEGIN..END span is a single slot in a single frame, summed correctly),
the value is always an integer number of 8192-wide quad ROWS (0x1de98f00 =
8192 x 61260; an earlier environment read 0x077a63c0 = 8192 x 15315), and
tile-row-granular partial renders decide how many overdraw layers escape
culling, which is why the number moves between environments. The same test's
single-quad section counts bit-exactly (0x75cf00 = one 3456x2234 quad), so
the machinery is precise whenever HSR has nothing to merge. Counting all
overdraw layers would need an encoder per draw under active queries,
destroying pass batching; the kept optimization is single-encoder pass
batching, so this is `expected`. Real-game occlusion (a bounding box tested
against a populated depth buffer, read as zero/non-zero) is unaffected.

Wine first times 1,000 query BEGIN/END pairs. If that loop takes more than
70 ms, device.c:6706 prints `Tests skipped: Test loop took too long (...)`,
ending with `skipping large query tests.`, and jumps to cleanup before 6780.
An Intel CI run measured 100 ms and took that branch. The scanner recognizes
that exact site and message shape, including the elapsed time above 70 ms,
as evidence that 6780 did not run. The mapping must be reviewed with the
baseline's source locations when Wine changes. The classification remains
`expected` and its count stays pinned at one: `ceiling` would also tolerate
an assertion that ran and passed, which this observation does not establish.

### device.c/test_timestamp_query

No failing sites. The test first creates `TIMESTAMPFREQ` and skips when it
returns `D3DERR_NOTAVAILABLE`, before probing `TIMESTAMP` or
`TIMESTAMPDISJOINT`. That skip establishes no result for either later query.
All three types are unsupported; the end-to-end query test checks each
capability probe and creation against `D3DERR_NOTAVAILABLE`, with EVENT and
OCCLUSION as supported controls.

### device.c/test_lockrect_invalid
Sites: 8664=expected 8682=expected 8701=expected

We PASS the accept-invalid lock checks (the `broken()`-guarded Win7 reject
alternative is not what we take). These offset assertions then compare our
returned pointer against blind `top*pitch + left*bpp` arithmetic on the
invalid rect. `parse_rect` clamps invalid rects (negatives to 0,
inverted/zero-area to the full mip), so our offsets differ; matching XP
exactly would require handing out pointers OUTSIDE the staging allocation,
which the lock-safety model forbids (`lock_region_ptr` bounds assert).
Deliberate safety tradeoff, kept. (Cube's garbage offsets are pointer diffs
across unrelated per-lock allocations: meaningless, not out-of-bounds.)

### device.c/test_pinned_buffers
Sites: 10074=expected 10079=expected

The test expects a DISCARD re-lock to return the same pinned pointer with
prior contents intact, a driver-specific optimization probe with no cap
branch. Our rename-on-DISCARD model returns fresh backing by design, and
DISCARD contents are spec-undefined, so our behavior is legal. Intent to
keep (the rename model is core).

### device.c/test_lost_device
Sites: 12144=expected 12146=expected 12153=expected 12155=expected
Sites: 12199=expected

Focus-loss/device-lost lifecycle: TestCooperativeLevel/Present/Reset must
report DEVICELOST/DEVICENOTRESET across a fullscreen focus cycle. Our
device is never lost to focus by design (no exclusive fullscreen, no GPU
loss on Metal). The non-OK `TestCooperativeLevel` answers we give are the
DEVICENOTRESET latch a failed `Reset` leaves behind, which is the windowed
API contract test_reset exercises, and the DEVICELOST a frame the layer
failed to encode or submit latches, which no focus change produces.

### device.c/test_check_device_format
Sites: 12689=expected 12694=expected

CheckDepthStencilMatch(..., D3DFMT_D32): native returns NOTAVAILABLE; we
return D3D_OK because D32 genuinely maps to Depth32Float and works. We
advertise MORE than native here, deliberately; not an omitted-cap (`caps`)
case, and our answer is truthful for our backend. 12694 is the R5G6B5
render-target row of that check, so it fires on the native legs only: on the
Intel legs R5G6B5 is no render target and the answer is the NOTAVAILABLE the
test expects.


### device.c/test_miptree_layout
Sites: 12784=expected 12823=expected

The test asserts each mip's lock pointer sits at a contiguous offset from
level 0 (single-allocation mip chain). Our staging is one PageBox per mip,
which is load-bearing for the rename-at-overlap versioning model (each
mip's Arc swaps independently); a contiguous chain is structurally
incompatible with that design, which we keep. Site 12823 is the same pointer
layout assertion across six cube faces and their mip levels. Cube staging is
also one PageBox per subresource so a face or mip can rename independently.
Per-subresource pixel data is correct.

### device.c/test_cursor_clipping
Sites: 14930=ceiling

The test creates a fullscreen device at the first served mode of at least
640x480 that differs from the desktop mode on both axes, and the cursor clip
must then equal the virtual screen, i.e. the mode. Under Wine's emulated
mode-set win32u clips the foreground fullscreen window to the physical
monitor and reports that rect mapped back into the mode, so a mode of
another aspect than the display's has its letterbox bars inside the clip:
640x480 on a 3456x2234 panel reads "(-51,0)-(691,480)". That is win32u's
mapping, not ours; the device sets the mode exactly as native does.
`ceiling` because the served list puts the sizes that fill the display
first, so the pick is one of them and the site reads zero: 2624x1696 on a
3456x2234 panel, 1600x1200 on the CI runner's 4:3 virtual display. It fires
only where no second size fills the display and the pick is a standard size
of another shape.

### device.c/init_d3d9on12_modules
Sites: 15088=ceiling

`win_skip("Direct3DCreate9On12 is not supported…")`: under Wine, win_skip
counts as a test failure. We don't provide the D3D9-on-D3D12 bridge; N/A on
Metal. Ceiling, not expected, because the site only fires where the Wine
build ships a loadable d3d12.dll: the win_skip sits after the three
LoadLibrary calls, and a failed load takes a plain `skip()` that counts
nothing. The pinned CI release is built with Vulkan and reads 1; the current
local dist is built without (no winevulkan.dll, no i386 d3d12.dll at all)
and reads 0. Not test-source drift: `dlls/d3d9/tests/` is identical between
the two builds.

### device.c/test_d3d9on12
Sites: 15160=expected

`win_skip("Failed to load d3d9on12 modules…")`: the companion to 15088,
same D3D9-on-D3D12 rationale. `expected`, not `ceiling`, because it fires
on both kinds of build: this skip is the module-load failure itself, which
under a Vulkan-less build happens one dll earlier but still lands on this
line's win_skip in `test_d3d9on12`.

### visual.c clusters

### visual.c/z_range_test
Sites: 3887=expected 3889=expected 3891=expected 3894=expected
Sites: 3963=expected 3965=expected

All six depend on a depth clear (0.75) written BEFORE a Present surviving
into later frames with ZWRITE off. Store-action Rule B flips the auto DS
store to DontCare at Present — the deliberate TBDR depth-store elision (the
preserve fix was implemented and reverted to keep the optimization). The
broken() r500 alternatives are ignored by the runner; the primary
assertions need cross-Present depth.

### visual.c/texdepth_test
Sites: 5360=expected 5398=expected 5436=expected 5454=expected

The ps_1_4 depth-gradient math is correct (the same-frame cycle passes and
is absent here). The failing cycles read the gradient across Presents —
the same Rule B depth-store elision as z_range_test.

### visual.c/pixelshader_blending_test
Sites: 12008=expected

Renders into a one- or two-channel texture (G16R16, R16F, G16R16F, R32F,
G32R32F) with blending on, then samples it and expects absent channels to
read as 1.0. Render targets now keep separate attachment and sampling views,
including both linear and sRGB roles for X8 formats. Native Apple GPUs apply
the sampling swizzle while stored channels retain their blended values.

The remaining failure is the Mac2 Paravirtual device's ignored texture-view
swizzle, shared with ordinary sampled textures below. It reads absent colour
lanes as zero despite reporting the requested view swizzle. The E2E controls
compare those lanes with an ordinary same-format texture on that device and
still verify stored lanes, mip and cube addressing, and view retirement.
Physical-GPU tests additionally require the specified all-ones missing lanes.

### visual.c/test_fetch4
Sites: 15668=ceiling 15727=expected

Fetch4 uses native gather for point-filtered 2D L8/L16/R16F/R32F/A8 and
INTZ/DF16/DF24 samplers. GET4 and GET1 commands through D3DSAMP_MIPMAPLODBIAS
set and clear a latch; numeric bias writes preserve it. Gather honors projected
coordinates and reads the base mip, including texldb/texldd/texldl. DF16/DF24
raw reads now return (depth, 0, 0, 1), while INTZ broadcasts depth.

The basic gather and DF raw-depth checks pass on local and `@mac2` legs.

15668 keeps its existing ceilings. The test writes only four bytes of each
format texture, so projected L16/R16F samples can read unwritten texels. ATI1
is outside the supported Fetch4 format set and keeps ordinary sampling, with
alpha one rather than a fourth gathered texel. A8R8G8B8 also keeps ordinary
sampling; its unwritten texels can have alpha one rather than the expected zero.
These cases account for the remaining format-site failures. Keep the higher pins
because counts vary with allocation contents. 15727 (`@mac2` legs only) is the
3D section's `L8` volume texture, which reads its green and blue lanes through
the channel swizzle the paravirtual device ignores (the swizzle cluster above):
the sample carries red alone there.

### visual.c/fp_special_test
Sites: 16433=expected

VS special-float ops on NaN/±inf: the test accepts four distinct vendor
results (r500/r600/nv40/nv50) plus broken(warp) — special-value handling is
GPU-defined, not spec-mandated. The one failure left on the Apple family is
the `def1` case, a `def` constant holding a NaN moved into the output: our
Metal GPU produces a fifth valid IEEE result matching no vendor's encoding
(0x008000ff). Matching a specific vendor is neither feasible nor desirable.
No capability involved (old `caps` tag incoherent). `nrm` of a zero vector
returns its source, the zero vector the test's hardware gives, so it passes.
The `@mac2` legs count two: `def1`, and `pow`, whose result the paravirtual
device encodes its own way (0x00808000; `def1` reads the same value there).

### visual.c/float_texture_test, g16r16_texture_test, test_mipmap_autogen, test_signed_formats, volume_v16u16_test
Sites: 5090=expected 5169=expected 6034=expected 18787=expected
Sites: 18790=expected 18793=expected 18796=expected 20702=expected
Sites: 20751=expected

`@mac2` legs only, one mechanism. Every format whose D3D9 sample fills a
lane the Metal format lacks (R32F's green and blue, G16R16's alpha, V8U8's
and V16U16's blue and alpha, X8R8G8B8's alpha, L8's green and blue) is handed out as a
texture view with a channel swizzle. The paravirtual device on the Intel CI
image creates that view, reports the swizzle on it, and samples through the
base texture's lanes anyway, measured in the workflow's probe job; every
real GPU family applies the swizzle. So R32F reads its missing lanes as
zero (5090), G16R16 its alpha as zero (5169), V8U8 and V16U16 their blue
as a stored lane (20702 and 20751, which count both formats' rows, and the
four V16U16 volume probes 18787, 18790, 18793 and 18796), and the autogen
X8R8G8B8 chain its padding byte as alpha (6034). A device limitation with no D3D9-side answer, so `expected`;
none of these fire on the Apple family, and a real Intel/AMD Mac is expected
to read zero here.

### visual.c/test_multisample_mismatch
Sites: 20880=expected 20883=expected 20959=expected 20962=expected

The whole test draws with a multisampled render target beside a
single-sampled depth buffer and the other way round. Metal rejects a render
pass whose attachments disagree on sample count, so mtld3d drops the
mismatched depth attachment; the draws land but the depth test does not
gate them. The pipelines and clear quads built for such a pass declare no
depth or stencil format either, since Metal rejects a pipeline that names a
format the pass has no attachment for. D3D9 never defined the case: every assertion here carries a second accepted colour under
`broken()`, and the comments in the test record that AMD and Nvidia disagree
about whether the draw happens at all.

### visual.c/test_flip
Sites: 22053=expected 22064=expected

The device is created with D3DSWAPEFFECT_DISCARD, under which post-Present
backbuffer contents are UNDEFINED by spec; the test observes native's
incidental flip-chain content rotation. Not emulating that is
spec-compliant. Surface identity and lockable read-back now pass. A title
relying on flip-chain read-back under FLIP/COPY swap effects would be a
different (real) matter.

The original `expected` classification of all five sites was wrong: their raw
result was `0xdeadbeef`, not a colour from an undefined backbuffer. The lockable
read-back helper accepts only `A8R8G8B8`, and the backbuffer's `GetDesc`
incorrectly reported `X8R8G8B8`. That descriptor defect was `real`, including on
the mixed sites whose remaining assertions observe undefined contents.

Reporting the requested `A8R8G8B8` format lets the helper read pixels. Sites
22055, 22066 and 22072 pass on both architectures across the Apple GPU legs
and the Intel/AMD `@mac2` runners. Their baseline pins have been removed. Sites
22053 and 22064 return actual colours from the shared backing rather than a
rotating chain, so only their by-design remainder is classified `expected`.
That shared backing is the kept divergence "Every back-buffer index of a swap
chain is one surface" above.

### visual.c/test_max_index16
Sites: 24133=expected 24135=expected

Metal treats index 0xffff as the un-disableable uint16 primitive-restart
sentinel, dropping the triangle that uses it; the test additionally writes
vertex 0xffff OUTSIDE its lock (UB, may never reach the GPU). broken(warp)
shows even the MS reference rasterizer fails this; the runner ignores
broken(). Accepted platform limitation (no cap branch — old `caps` tag was
wrong).

### visual.c/test_map_synchronisation
Sites: 25148=flaky

The failing config is exactly the plain (no DISCARD/NOOVERWRITE) PARTIAL
lock of a contended Direct buffer, which native stalls for. Our buffer-
rename design deliberately removed that stall (`plan_lock` → WriteInPlace);
re-adding it is the only fix and is a rejected perf regression. Whether the
test observes the divergence is a per-run CPU-vs-GPU race (the probe's Lock
write lands before or after the GPU consumes the in-flight draw), so the
count flutters between 0 and 1 across runs of the same binary; it read 0 on
the CI runner and tripped the stale-baseline gate (PR #12), hence the flaky
tolerance. That race is in the observation only, not in the decision:
`plan_lock` is a pure function of a `coherent_seq` its caller read once
with Acquire before calling, and only the unix side raises that counter
(`fetch_max`, on GPU retirement), so a stale read can turn a legal
in-place write into a needless rename but never the reverse. The kept
divergence itself is unchanged.

### visual.c/test_mipmap_upload
Sites: 27550=expected

The app writes the whole mip chain through a single level-0 lock pointer;
with per-mip PageBox staging the upper mips never receive the data. Same
architecture-we-keep rationale as test_miptree_layout — but this is the
weakest `expected` in the file: it produces wrong rendered pixels for a
real-app pattern (Wine cites shipped titles). If the per-mip staging
commitment is ever softened, retag `real` first.

### visual.c/test_default_attribute_components
Sites: 27902=expected

FLOAT→unorm rounding at exactly .5: Metal rounds 76.5 up (77), refrast
truncates (76). A ±1 GPU rounding-convention difference with no cap branch;
mimicking refrast exactly is not feasible or desirable.

### visual.c/test_format_conversion
Sites: 28024=caps

Intel legs only. The test expects `CheckDeviceFormatConversion(YUY2,
R5G6B5)` to answer D3D_OK, as every desktop driver does. A conversion
destination has to be renderable, since the StretchRect quad draws into it,
and on a device without the packed 16-bit formats R5G6B5 is no render target,
so the answer is NOTAVAILABLE. That is the conformant answer for a device
without the capability; the same rule is pinned by the e2e
`check_format_conversion` test, which asks the device first.

### visual.c/get_rt_readback
Sites: 199=expected

`point_match` walks outward from a point's centre until the colour changes
and compares that radius with the size the test set; under the scaled leg
every point's edge reads the resolve's blend one pixel wider than the
rasterized point, so the radius it finds is off by one at every size ("The
scaled leg").

### visual.c/texkill_test
Sites: 5545=expected 5554=expected 5570=expected 5572=expected 5606=expected
Sites: 5615=expected 5633=expected

Every probe sits one pixel inside a `texkill` boundary (x = 66, 575 or 578
on row 49) and reads the blend of the killed and the surviving side
(`0x9f6000`, `0xdf2000`) rather than the pure colour ("The scaled leg").

### visual.c/test_fragment_coords
Sites: 10769=expected 10771=expected 10773=expected 10775=expected

The four probes straddle the reported centre (319/320, 239/240), where the
four `vPos` quadrant colours meet, and each reads the blend of its two
neighbours (`0x00dfdf` for `0x00ffff`). The register itself answers in the
reported space since #403: before it every probe read the wrong quadrant
outright. The fraction assertion passes, `frc(vPos)` stays zero under the
scale ("The scaled leg").

### visual.c/test_pointsize
Sites: 11502=expected 11504=expected 11506=expected 11508=expected
Sites: 11556=expected 11567=expected 11571=expected 11574=expected
Sites: 11579=expected 11582=expected 11585=expected 11588=expected

Point sizes are stated in reported pixels and kept there (the vertex
epilogue converts them to render pixels, see `points.rs` in the end-to-end
suite), but the test probes each point's edge to the pixel and a one-pixel
point outright; every such probe reads the resolve's blend (`0xe3`, `0xfb`,
`0xdf` channels) ("The scaled leg").

### visual.c/test_viewport
Sites: 14064=expected 14081=expected

Pixel (1, 119) and (1, 360) lie on the first column and the first and last
rows of a viewport bound, and read a one-eighth blend (`0x202020`) of the
colour across it ("The scaled leg").

### visual.c/depth_buffer_test
Sites: 14557=expected

The test creates targets of 320x240, 480x360 and 640x480 and binds each
with the device's one depth buffer, expecting depth written through one to
be read through another at the same pixel. The 640x480 target is the
back-buffer size and rasterizes at the scale, the depth buffer with it; the
480x360 target is its own size and does not. Depth written through the
scaled target lands at three quarters of its reported coordinates, then is
read through the unscaled target at the unconverted ones, so the two probes
read the colour of the neighbouring quad. By design: the scale is a property
of the targets at the back-buffer size (`mtld3d.conf`, `render.scale`), and
a depth buffer shared between a scaled and an unscaled target of different
sizes has no single mapping that serves both.

### visual.c/clip_planes
Sites: 16129=expected 16131=expected

The plane cuts the quad at y = 240.5 in the reported space and the probes
sit on rows 240 and 241, the two rows either side of it; under the scale
that edge falls inside one render row and both probes read its blend
(`0x9ca13c`) ("The scaled leg").

### visual.c/test_filling_convention
Sites: 27409=expected

The test draws into 2x2 to 8x8 render targets and copies each onto the back
buffer with `StretchRect` before probing it pixel by pixel; a target that
small resampled onto the scaled back buffer has no pixel that is not a
boundary, so every probe reads a blend (`0x24db` for `0x00ff`). 221 hits,
one per pixel per case ("The scaled leg").

### visual.c/test_ffp_w
Sites: 28113=expected 28137=expected

Both probes sit on the edge of the w-tested quad and read the one-eighth
blend (`0xdf0020` for `0xff0000`) in both the declaration and the FVF form
("The scaled leg").

### stateblock.c clusters

### d3d9ex.c clusters

`Direct3DCreate9Ex` hands out an extended interface, so every test of the
subtest creates its extended device and runs to the end with no crash on
both architectures. Three causes carry most of the sites, each a kept
divergence (see "Kept divergences"). Every window `create_window` makes is
`WS_MAXIMIZE`, and a maximized window's back buffer follows its client rect
(the rule `device.c/test_scissor_size` explains), so a test that asked for a
smaller back buffer reads the window's size, and a pixel probe at a fixed
coordinate reads another part of the frame; 26 sites come from it. The
user-memory copy and the absence of occlusion status are the other two. The
window and focus sites are the same decisions the `device.c` clusters record.
Leaving fullscreen follows the device's kind as the reference does: an
extended device leaves the window at the fullscreen rect and gives it back
the style it had, visibility included, so `test_window_style` passes.

### d3d9ex.c/test_user_memory
Sites: 775=expected 806=expected 819=expected 842=expected 843=expected
Sites: 872=expected

775, 806, 819 and 843 expect `LockRect` to hand back the application's own
pointer, and 842 the packed pitch (33 for an L8 row 33 texels wide; ours is
36): the user memory is copied into the level, which keeps the layer's
staging. 872 reads the drawn ramp at (320, 240) of the maximized window's
back buffer, which is a tenth of the way across rather than the middle, and
reads 0x171717 where the middle reads 0x7f7f7f; the end-to-end suite reads
the middle of the same draw on a 640x480 back buffer.

### d3d9ex.c/test_reset
Sites: 1098=expected 1105=expected 1106=expected 1114=expected 1115=expected
Sites: 1145=expected 1152=expected 1153=expected 1162=expected 1163=expected

The scissor rect, the viewport and the swap chain's back buffer after a
windowed `Reset` to 400x300 and to 500x400 read the maximized window's client
rect. The state the extended `Reset` keeps (render states, the viewport's
depth range, the bindings) and the fullscreen sizes pass.

### d3d9ex.c/test_reset_ex
Sites: 1677=expected 1684=expected 1685=expected 1693=expected 1694=expected
Sites: 1726=expected 1733=expected 1734=expected 1743=expected 1744=expected

The `ResetEx` counterparts of the `test_reset` sites: the windowed requests
read the maximized window's client rect. The mode checks pass.

### d3d9ex.c/test_user_memory_getdc
Sites: 1990=expected 1996=expected 1997=expected

1990 expects the DIB a `GetDC` maps to be the application's user memory, and
1996/1997 expect what GDI draws through it to land there: the user memory is
copied once, so neither side reaches the other.

### d3d9ex.c/test_lost_device
Sites: 2039=expected 2046=expected 2049=expected 2051=expected 2153=expected

Each expects `S_PRESENT_OCCLUDED` from `CheckDeviceState` or a present while
another window is in front, where the device answers `D3D_OK`: no exclusive
mode is taken and an occluded present costs nothing, so no occlusion status
is reported. 2039, 2046 and 2153 accept `D3D_OK` as a `broken()` answer, and
`D3D_OK` is what the device gives, but `broken()` only holds on Windows, so a
run under Wine counts it; 2049 and 2051 accept no `D3D_OK` at all. The
`TestCooperativeLevel` and `Reset` assertions pass.

### d3d9ex.c/test_wndproc
Sites: 2908=expected 2913=expected 2915=expected 2920=expected 2924=expected
Sites: 2949=expected 2957=expected 3007=expected 3036=expected 3107=expected
Sites: 3183=expected 3203=expected 3209=expected 3230=expected

The extended copy of `device.c/test_wndproc`, and the same decisions: no
activation or mode message generation (2908, 2949, 3007, 3036, 3183, 3203),
no focus or foreground change (2913, 2915), the cursor subclass on the device
window rather than a subclass of the focus window (2920, 2924, 3230), no
focus-window minimize (3107), and a `WINDOWPOS` the message walk never
reaches (3209, as `device.c` 4551). 2957 expects `S_PRESENT_OCCLUDED` from
`CheckDeviceState` after the focus loss, the occlusion decision above.

### d3d9ex.c/test_wndproc_windowed
Sites: 3331=expected 3345=expected 3349=expected 3356=expected 3399=expected
Sites: 3421=expected 3425=expected 3432=expected

As `device.c/test_wndproc_windowed`: 3349 and 3425 expect the focus window
subclassed in fullscreen, and the other six the device window's procedure
unchanged, which the cursor subclass changes on purpose.

### d3d9ex.c/test_backbuffer_resize
Sites: 3925=expected 3926=expected

The back buffer held across `Reset` keeps its old size, as it should; the
size it had was the maximized window's rather than the 640x480 requested.
The detach itself passes: the held surface's container is the device and a
new back buffer object is handed out.

### d3d9ex.c/test_resource_access
Sites: 4585=caps

The volume loop creates an ATI2 volume in the scratch pool without asking
`CheckDeviceFormat` first, where the 2D and cube loops and `device.c`'s copy
of this loop skip ATI2 when the device does not offer it. The device offers
no ATI2 resource, and a FOURCC the driver does not know has no size the
runtime could allocate, scratch pool included, so `D3DERR_INVALIDCALL` is
the answer of a device without the format.

### d3d9ex.c/test_sysmem_draw
Sites: 4779=expected 4804=expected 4842=expected

Each draws a gradient over the whole back buffer and probes (320, 240),
expecting its middle; on the maximized window's back buffer that point sits
near the green corner (0x04e318). The draws from system-memory buffers
themselves are correct.

### d3d9ex.c/test_pinned_buffers
Sites: 4956=expected 4961=expected

As `device.c/test_pinned_buffers`: a `D3DLOCK_DISCARD` re-lock renames the
buffer rather than handing back the pinned pointer with its contents.

### Wide-format offscreen conversion

Same-size DEFAULT offscreen `StretchRect` converts A16B16G16R16 and
A32B32G32R32F directly into A8R8G8B8. UNORM16 channels round to the nearest
8-bit normalized value; float channels clamp to [0, 1] and round only at the
final destination, with NaN mapping to zero. Partial rectangles preserve the
pixels outside them through the existing staging ownership path. Contended
conversion staging is renamed so queued uploads and retries retain their
source bytes. A destination upload following an ordered GPU copy uses the
ordered blit stream instead of running ahead of that copy at the frame head.
Ordinary GPU copies and draws keep their existing routes.

The format gate rejects other cross-Metal-format offscreen pairs without a
CPU codec before scheduling uploads. This is an implementation limitation,
not a native format restriction; render-target destinations retain their GPU
conversion path. Wide-to-wide, other wide destinations and offscreen scaling
remain unsupported. Adding these two codecs does not broaden the separately
validated UpdateSurface/UpdateTexture format set. The existing device.c
StretchRect matrix covers A8R8G8B8, X8R8G8B8 and R5G6B5, so these corrections
are pinned by end-to-end pixel regressions rather than a baseline reduction.

### visual.c/yuv_layout_test, YV12 and NV12

The test creates a 20x16 DEFAULT offscreen plain per format, behind two
gates: `CheckDeviceFormat(SURFACE, usage 0)` (`visual.c:12991`) and
`CheckDeviceFormatConversion` into the back-buffer format (`visual.c:12997`).
Both now answer yes for YV12 and NV12, so the planar halves run instead of
skipping.

The layout the test writes is relative to the lock pitch, never to the width:
the chroma starts at `Pitch * height`, YV12 stores a V plane and then a U
plane whose rows stride `Pitch / 2`, NV12 one interleaved plane whose rows
stride `Pitch`. mtld3d reports the width rounded up to four bytes as the
pitch, allocates `Pitch * (height + ceil(height / 2))` bytes, uploads them
verbatim into one R8 texture as wide as the pitch, and addresses every plane
from the pitch in the fragment decode and in the CPU converter. wined3d
stores the same single-channel image but places the YV12 half-planes at half
the texture width, which agrees only when the pitch equals the width; DXVK
reports the same pitch and then addresses by the extent. Neither difference
is visible at the test's 20x16.

The probes expect reduced-range BT.601 within 1 (`visual.c:13074`,
`visual.c:13077`), the matrix the packed formats already decode with. DXVK
converts with BT.709 and answers the conversion query no for both formats, so
it takes the 12997 skip and its colours are never checked here.

Limits: DEFAULT pool only, YV12 heights even (the origin of the U plane of an
odd-height surface follows `floor(height / 2)` rows in the test's arithmetic
and `ceil(height / 2)` in a writer that stores the rounded-up row count, and
the test's even surface pins neither), no texture advertisement (wined3d's
choice; DXVK advertises one), and unfiltered chroma. The test's `ColorFill`
calls expect `S_OK` only and read nothing back; mtld3d returns it and leaves
the surface unfilled, as for YUY2 and UYVY.

### visual.c/tssargtemp_test

The fixed-function cascade supports the optional TSSARGTEMP capability.
RESULTARG selects CURRENT or a fragment-local TEMP register initialized to
zero. Both color and alpha read the pre-stage register values before the
whole result is assigned. The final output remains CURRENT. The Wine test
at 12124 combines independently retained red CURRENT and green TEMP values
into yellow; the focused end-to-end tests additionally cover initialization,
modifiers, destination changes, stateblocks, and Reset. No uniform or texture
allocation backs the register.

### visual.c/test_per_stage_constant and test_texture_blending

PERSTAGECONSTANT is advertised. D3DTA_CONSTANT selects the current stage's
D3DTSS_CONSTANT color, with alpha-replicate and complement modifiers. Values
use fragment constants rather than shader keys. Only a shader consuming a
stage constant extends the existing texture-factor buffer, to at most nine
float4 rows. Ordinary shaders keep their existing one-row construction and
unused-buffer bind suppression.

The dedicated test expects 0xa1b2c3, complement 0x5e4d3c, alpha-replicated
0x808080 and alpha-blended 0x80007f at 19538/19556/19574/19596. Together with
TSSARGTEMP, the capability also opens test_texture_blending after its two
capability guards. Its individual cases still check TextureOpCaps. Raw
D3DTSS_CONSTANT defaults remain zero, as device.c:7746 asserts. ALL and
recorded stateblocks retain the value; this feature preserves the existing
PIXEL preset exclusion rather than changing capture policy.

### visual.c/volume_v16u16_test and test_signed_formats

V16U16 textures use native RG16Snorm storage, with missing blue and alpha
set to one. The signed-format and volume tests run through the same native
byte uploads as other mapped formats. 2D and cube AUTOGEN requests answer
D3DOK_NOAUTOGEN and create one actual level with the public usage retained;
no mip-generation work is submitted. Render-target and sRGB queries remain
unavailable.

Q8W8V8U8 textures use native RGBA8Snorm storage with four signed channels.
The negative Q channel follows signed normalization, including a floor of
minus one for both minimum encodings. The legacy R200 exception in Wine's
signed-format alpha test is not an emulated target. All four lanes are
stored, so sampling needs no conversion and no view swizzle, and the
Q8W8V8U8 rows of test_signed_formats do not depend on the mechanism that
fails the V8U8 rows on the `@mac2` legs. Render-target and sRGB capabilities
stay absent; AUTOGEN texture/cube requests preserve usage but use one actual
level with no generated chain. DYNAMIC creation in the MANAGED and the SCRATCH
pool is rejected for every format and every texture type, the way D3D9 rejects
it, rather than for the signed and packed ten-bit formats alone or for volumes
alone. X8L8V8U8 and L6V5U5 remain separate capabilities.

Q16W16V16U16 textures use native RGBA16Snorm storage and the same scoped
NOAUTOGEN and unavailable render/sRGB policies. All four lanes are
stored and no view swizzle applies, so nothing here depends on the mechanism
that fails the V8U8 and V16U16 rows on the `@mac2` legs. Upstream Wine has no
Q16 positive row in test_signed_formats; existing mode/conversion references
do not establish signed pixel correctness. Dedicated end-to-end tests pin
all four signed16 lanes, alpha, eight-byte transport and float32 sampling
precision. This feature does not claim a Wine positive-pixel skip reduction.

ColorFill of a DEFAULT offscreen plain surface in V8U8, V16U16, Q8W8V8U8 or
Q16W16V16U16 writes each D3DCOLOR channel as the nearest nonnegative signed
code, R, G, B, A into U, V, W, Q, which is the value a clear of the matching
signed-normalized attachment to the colour's [0, 1] floats stores. The format
table of visual.c/color_fill_test has no signed row and no other upstream test
fills a signed surface, so no site moves. Dedicated end-to-end tests pin the
whole and sub-rect bytes, the texels around a sub-rect, the rejected CPU-pool
and texture-level destinations, and the upload the GPU samples.

### device.c/test_getdc, A2R10G10B10

A2R10G10B10 textures and CPU surfaces are creatable, stored as native
BGR10A2Unorm. D3D9 packs blue in the low ten bits, then green, red and the
two alpha bits, which is that Metal format's layout, so locks expose the
native words and sampling needs no conversion and no view swizzle. Every
lane is stored, so nothing here depends on the mechanism that fails the V8U8
and V16U16 rows on the `@mac2` legs.

The A2R10G10B10 row of test_getdc used to skip because neither the SYSTEMMEM
surface nor the MANAGED texture could be created. It now runs and passes:
the row expects GetDC to fail and the out pointer to stay untouched, and the
format has no GDI mapping. The row uploads no pixels. The other upstream
references are the display-format table of test_display_formats, where the
format stays no display, back-buffer or render-target format, and the
identity row of test_format_conversion. No failing site moves on any leg.

The end-to-end suite checks ten-bit precision per lane, the four alpha
codes and the raw lock and copy words. Render-target, sRGB and legacy
bump-map queries are rejected. Texture and cube AUTOGEN answers NOAUTOGEN
with one physical level and the requested usage retained. ColorFill of a
DEFAULT offscreen plain surface encodes each channel to its nearest code;
the rounding of a Windows driver has not been measured.

### A2B10G10R10 sampled textures

A2B10G10R10 textures and CPU surfaces are creatable, stored as native
RGB10A2Unorm. D3D9 packs red in the low ten bits, then green, blue and the
two alpha bits, which is that Metal format's layout and the mirror of
A2R10G10B10, so locks expose the native words and sampling needs no
conversion and no view swizzle. Every lane is stored, so nothing here depends
on the mechanism that fails the V8U8 and V16U16 rows on the `@mac2` legs.

Upstream names the format twice. test_display_modes (device.c:2620) expects
EnumAdapterModes to answer INVALIDCALL, and it still does: the format stays
no display, back-buffer or render-target format. test_format_conversion
(visual.c:27957) lists it in `all_formats`, where the identity pair already
answered S_OK and every other pair accepts either answer. Neither row creates
a resource, so there is no skip to open and no failing site moves on any leg.

The format shares the A2R10G10B10 policy: render-target, sRGB and legacy
bump-map queries are rejected, texture and cube AUTOGEN answers NOAUTOGEN
with one physical level, and ColorFill of a DEFAULT offscreen plain surface
encodes each channel to its nearest code. The
end-to-end suite runs the A2R10G10B10 scenarios in this lane order and adds
two witnesses of the order itself: the same word written to both formats
samples with red and blue exchanged, and copies between the two formats are
rejected with the destination unchanged.

### visual.c/volume_dxtn_test and device.c/test_volume_blocks

DXT1 to DXT5 volume textures are creatable in every pool, stored as native
BC1, BC2 and BC3 textures: a 3D texture when the base depth is above one, the
2D texture every depth-one volume already uses otherwise. Locks expose the
compressed blocks with a row pitch of one block row and a slice pitch of the
level's block rows, and uploads copy them verbatim, so nothing is decompressed
and no shader or draw-time work is added. The level's compressed-layout
marker (a zero bytes-per-pixel) is kept through volume creation, so an upload
of a level that has reached one slice counts block rows, not texel rows, and
the alignment repack never asks for bytes past the level's staging. That path
was reachable before this capability through a SCRATCH volume bound for
sampling. Cube creation keeps the marker as well, and the repack checks the
rows it is asked for against the staging length before it copies.

volume_dxtn_test (visual.c:18588) skipped its DXT1, DXT3 and DXT5 rows on the
VOLUMETEXTURE query and now runs them: three skips fewer and the row's eight
readbacks at visual.c:18631 per format on every visual leg, all expected to
pass, including the scaled legs (the back buffer is read at reported
coordinates) and the `@mac2` legs (BC formats store every lane and carry no
view swizzle). test_volume_blocks (device.c:10713) takes its supported branch
for the five DXT rows, where DEFAULT, SYSTEMMEM and MANAGED creates of every
block-aligned size now succeed as device.c:10760 expects; its pitch, offset
and misaligned-box checks already ran against the SCRATCH volume. A
successful create skips the null-pointer check of a failed one, so the device
assertion total can fall while nothing fails. No failing site is expected to
move on any leg, and the table has no DXT2 or DXT4 row in visual.c.

Volume queries accept DYNAMIC, FILTER, SRGBREAD, VERTEXTEXTURE and WRAPANDMIP
for the five formats and reject SRGBWRITE, RENDERTARGET, DEPTHSTENCIL,
POSTPIXELSHADER_BLENDING and AUTOGENMIPMAP. SRGBREAD covers DXT2 and DXT4 on
volumes, as on 2D and cube textures, because BC2 and BC3 have sRGB views, and
no upstream test asks it of either format. ATI1 and packed-YUV volumes stay
unadvertised and SCRATCH-only. DXT2 and DXT4 keep their format identities and
share the DXT3 and DXT5 block encodings: the stored blocks are sampled as they
are, with no multiply or divide by alpha, and the application chooses its
blend factors.
