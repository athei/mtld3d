# Architecture

mtld3d is a Wine-side translation layer that ships a D3D9 implementation backed by Metal. The runtime is split across three linkage units that meet at the Wine PE/Unix boundary.

```
test.exe → d3d9.dll → mtld3d.dll → mtld3d.so
(i386 PE)  (i386 PE)  (i386 PE)  (Mach-O, Wine's own arch)

test.exe → d3d9.dll → mtld3d.dll → mtld3d.so
(x64 PE)   (x64 PE)   (x64 PE)   (Mach-O, Wine's own arch)
```

The PE column is fixed by the game; the `.so` follows the arch of the Wine build that loads it (Wine resolves unix libraries out of `lib/wine/<cpu>-unix`), so it is built and shipped for both `x86_64-apple-darwin` and `aarch64-apple-darwin`. An x86_64 Wine loads the first, with the PE side translated by Rosetta 2; an arm64 Wine loads the second and translates the PE side itself (FEX).

- `d3d9.dll` — D3D9 API implementation. COM vtables, caps, state management. Calls Metal-level thunks via its internal `unix_call` caller stub (`windows/d3d9/src/unix_call.rs`).
- `mtld3d.dll` — PE shim. Links winecrt0, owns Wine unix-call globals, exports `mtld3d_unix_call()`. Forwards every cross-boundary call from `d3d9.dll` into `mtld3d.so`.
- `mtld3d.so` — native macOS side. Pure Metal abstraction layer: thunks expose Metal operations only, no D3D9 knowledge.
- `mtld3d-core` — pure-Rust rlib linked into `d3d9.dll`. Host-testable.
- `shared` — PE↔Unix wire-format definitions plus cross-linkage-unit helpers.
- `types` — D3D9 type definitions (vtables, caps structs) shared between d3d9 and tests.

## Workspaces and crates

Two Cargo workspaces, one per target platform: `windows/` builds the PE side for `i686-pc-windows-msvc` and `x86_64-pc-windows-msvc`, `unix/` the Mach-O side for `x86_64-apple-darwin` and `aarch64-apple-darwin` (the latter is also the native test target). Open each in its own editor window for rust-analyzer to work.

| Crate               | Workspace  | Output                                                 |
|---------------------|------------|--------------------------------------------------------|
| `d3d9`              | `windows/` | `d3d9.dll`                                             |
| `mtld3d`            | `windows/` | `mtld3d.dll`, the shim                                 |
| `mtld3d-core`       | `windows/` | rlib linked into `d3d9.dll`                            |
| `mtld3d-types`      | `windows/` | rlib, D3D9 type definitions shared with the tests      |
| `mtld3d-tests`      | `windows/` | the end-to-end suite                                   |
| `mtld3d-unix`       | `unix/`    | `mtld3d.so`                                            |
| `mtld3d-shared`     | `unix/`    | rlib shared by `d3d9.dll`, `mtld3d.dll` and `mtld3d.so` |
| `mtld3d-conformance`| `unix/`    | the conformance runner                                 |

`mtld3d-core` holds every platform-independent helper (DXSO to MSL emission, the render-pass state machine, the slab allocator, format / FVF / vertex-decl / dirty-rect math, fixed-function state) and compiles for the macOS host as well as PE, so `cargo test -p mtld3d-core --target aarch64-apple-darwin` runs its unit tests natively instead of through Wine.

`mtld3d-shared` is the crate every linkage unit depends on, primarily for the PE/Unix wire format (the `Command` enum, the `Thunks` enum, param structs, typed `mtl::` wire values). Pure data and pure-Rust helpers only, no FFI and no `#[link]`, so both workspaces can depend on it cleanly. The internal crates are path dependencies and are not published to crates.io.

## Threading model

The API thread (the game's calling thread) is the bottleneck and must be unblocked fast. Every D3D9 call snapshots the relevant state (cheap u32 copies, vertex memcpy) into a closure and pushes it onto the current frame's op list — no translation, no Metal lookup, no encoding.

A dedicated **encoder thread** (one per device, `sync_channel(1)` backpressure) does the real work. On `Present()`, the API thread sends the accumulated frame and immediately starts collecting the next. The encoder thread runs each closure with mutable access to a `FrameEncoder` that owns persistent caches (pipeline states, depth/stencil states) and translates D3D9 → Metal commands into a fixed-size array.

A dedicated **submit thread** (one per device) executes the `SubmitFrame` thunk — the cross-boundary command replay and the commit of the frame's upload and render command buffers — overlapping the encoder's build of the next frame. The frame crosses as an owned `FramePayload` holding every buffer the thunk aliases by raw pointer; two payloads ping-pong over a cap-1 work channel, so render-ahead is bounded at one frame. Rare synchronous submits (`Reset`, mid-frame flushes, GPU capture) first drain the submit thread to idle, then run the thunk inline on the encoder thread — the two paths never call `SubmitFrame` concurrently.

A native **presenter thread** (one per device, owned by `mtld3d.so`, `metal/presenter.rs`) presents what the submit thread committed. A present-bearing frame leaves a packet behind its commit: the layer, the texture to present, its sequence. The presenter takes the packets in order, acquires the drawable, encodes the present route into a command buffer of its own and commits it, so a frame's render work never waits for a drawable and present order is the packet order. The split creates one hazard, a later render overwriting the back buffer before a pending present has read it, and the submit thread resolves it before it commits: a present-bearing submit waits for every pending present to have committed, whether the newest reads the back buffer or a copy, which is the cadence the display already set, costs no copy and keeps the queue one deep except behind a barrier; a submit that must not wait, a no-present mid-frame flush or one a barrier hurried, copies the pending present's frame into a slot and retargets the packet at it; the slot array is as deep as the pipeline (`PRESENT_PIPELINE_DEPTH` in `mtld3d-shared`, which the encoder asserts against its channel and payload caps at compile time), since a barrier hurries every frame the pipeline holds and each copies once, and its textures are allocated only when a copy needs them. So a read-back waits for committed render work and the GPU, never for the display. A backlog a barrier leaves behind the presenter is shown frame by frame at the display's cadence, never skipped or hurried: the pipeline refills on the API thread's side, which keeps the presented cadence even. Barriers (`drain_submit_thread`) set the hurry through `SetPresentWaitPolicy` around their wait for the submits in flight; `Reset`, shutdown and the GPU capture additionally wait for the presenter to go idle and its last present to retire (`WaitForPresentIdle`) before the back buffer or the layer can go. `debug.presentGateFile` parks the presenter before each drawable while the named file exists, the seam the test suite holds a read-back against.

EVENT query polls queue an open frame through the same asynchronous path with
Present suppressed, then observe `coherent_seq` for GPU completion. Queue admission
retains the existing bounded backpressure; the poll does not wait for encoding or
submission to finish. A later poll of the same issue queues no additional frame.

Both device and swap-chain `Present` build their continuation frame after the
input polls return. A poll can dispatch `WM_SIZE`, whose auto-resize drains the
encoder and replaces the implicit textures. A frame built before that callback
would restore the retired handles after the resize reseeded `current_frame`.
If the callback leaves the device requiring `Reset`, `Present` returns
`D3DERR_DEVICENOTRESET` before creating or submitting another frame.

A **log thread** (one per process) is the only thread that thunks for logging. d3d9.dll's `env_logger` sink pushes each formatted line onto an unbounded channel and returns, so a log line costs the API and encoder threads an allocation and a queue push; the log thread drains the queue and forwards every line through the `WriteLog` thunk into the process's log file, which the unix side owns and its own logger and crash handler write too. The thread starts from the first `Direct3DCreate9`, never from `DllMain` (loader lock), after the resolved `mtld3d.conf` has named the file's location through the `OpenLog` thunk; lines logged before that wait in the queue on the PE side and in the file sink's backlog on the unix side, so the file starts with the identity lines. It lives as long as an `IDirect3D9` does: it holds a reference on `d3d9.dll` for its lifetime and exits through `FreeLibraryAndExitThread`, so a `FreeLibrary` cannot unmap the image under it, and the last interface's `Release` sends it the stop and waits for it to be gone, so a `FreeLibrary` that follows (the probe pattern of launchers: load, create, release, free) finds no thread of ours in the image. `DllMain` can do neither: its `DLL_PROCESS_DETACH` runs under the loader lock, which a thread exit needs too. Lines logged while no thread runs wait in the queue for the next interface.

**AppKit stays on the main thread.** Every `AppKit` object the unix side touches (the metal view, its window, the screen under it, `NSApp`, the notification center) is created, read and released on the main thread, whatever thread the thunk arrived on. The two dispatchers in `metal/macdrv.rs`, `run_on_main_thread_sync` and `run_on_main_thread_async`, and the registry helpers `retain_view` / `retain_layer` are the only doors: a thunk latches what it knows on the device's attachment record and dispatches the walk, and the dispatched closure runs in an autorelease pool of its own so what it autoreleases drains at the layer's frame rather than in winemac's request-loop pool. The marker inside is always the checked one, so an off-main walk aborts at a named site instead of corrupting `AppKit`'s per-thread state. `docs/CONVENTIONS.md` §"AppKit work runs on the main thread" is the rule; `make audit` bans the unchecked marker.

**Application multithreading.** A device created with `D3DCREATE_MULTITHREADED` may be called from any application thread, and so may every object it created. Every `IDirect3DDevice9` entry point, and every entry point of every child object, holds the device's `ApiLock` (`windows/core/src/api_lock.rs`) for its duration: a reentrant lock, an owner thread plus a depth, so `Reset` applying state through the setters an application calls, or a child `Release` reaching the device's own release, re-enter freely, and the outermost return releases. The guard is the first statement of the thunk, ahead of its `ApiTimer`, so a wait for the lock counts as API time, and `make audit` checks that placement. The lock is the outermost lock in the PE side: `live_textures` and the cursor module's `DEVICE_INSTANCES` are leaf mutexes taken under it. It is held across `Present`, so a second thread waits up to one frame behind the presenter, as it does on native. The encoder, submit, prewarm and log threads never take it: none of them calls back into the device, so a thread that holds it while waiting on them cannot form a cycle. The lock lives outside `DeviceInner` (leaked at creation) because a child `Release` can free the inner while the guard its thunk took is still live. A device created without the flag has no lock and pays a null test and a refcount load per entry point. The cursor window procedure does not take the lock: it runs on the window thread, and a `Reset` or fullscreen transition holding the lock sends that thread synchronous messages, so a window thread waiting for the lock would deadlock the thunk. Native D3D9 has the same hole and applications keep the window thread out of D3D calls during `Reset`. What the procedure touches unlocked is the cursor latches and the auto-resize on `WM_SIZE`, so a user resize on the window thread while another thread draws under the flag is the documented residual.

```
API thread                  Encoder thread           Submit thread              Presenter (unix)
──────────                  ──────────────           ─────────────              ────────────────
D3D9 call → snapshot state  (blocked on channel)     (blocked on channel)       (blocked on packets)
         → push closure
         ...
Present() → send frame ───→ run closures
          → start next      translate D3D9 → Metal
            frame           finalize payload ──────→ SubmitFrame unix call:
                                                     replay commands,
                                                     wait for prior present
                                                     (or snapshot it),
                                                     commit upload + render,
                                                     queue packet ──────────→ nextDrawable,
                                                                              present buffer
                                                                              (blit / MetalFX / HDR),
                                                                              presentDrawable + commit
```

## Thunk vs Command

Two paths cross the PE/Unix boundary:

- **Command** = `MTLRenderCommandEncoder` method. Closures accumulate commands into a fixed-size array; `submit()` sends the whole array in one `unix_call(SubmitCommandBuffer)` and the unix side replays them inside a render pass. Examples: `setRenderPipelineState`, `setViewport`, `drawPrimitives`, `setFragmentTexture`.
- **Thunk** = everything else (object creation/destruction, texture upload, blit). Individual `unix_call()`.

API-thread thunks are restricted to device lifecycle only (`CreateCommandQueue`, `AttachMetalLayer`, `CreateBackbuffer`, `DestroyCommandQueue`), with two exceptions: `SetCursorOverlay`, whose handler is a store plus a coalesced main-queue dispatch (about 1-2 µs) and which replaces the `SetCursor` wineserver round trip (~100 µs) a show, hide or cursor change costs on the hardware path; and `SetPresentWaitPolicy` at the head of a synchronous flush, a flag store that wakes the submit thread out of its present wait before the flush queues behind the encoder, on a path that is already a read-back. Everything else — texture/sampler/pipeline creation, texture upload, resource destruction — runs on the encoder thread. Metal textures are created lazily on first draw via `FrameEncoder.texture_cache`. Resource cleanup uses closures pushed to the current frame.

Thunks are Metal-level operations, not D3D9 calls. Name thunks after what they do in Metal (`GetDeviceInfo`, `CreateCommandQueue`), not after D3D9 methods. D3D9 logic stays in `d3d9.dll`. Objects with no Metal state (like `IDirect3D9`) are PE-only.

## Upload order and retirement

Texture and buffer uploads form an ordered prefix before the application's render passes.
A texture upload that requires a render pass carries the preceding upload blits in its
`leading_blits`; preservation copies and subsequent uploads therefore keep their API order
across both encoder kinds. Blits after the final upload render pass form a final blit-only
descriptor in the prefix. `SubmitFrameParams.upload_pass_count` counts that prefix, and
the unix side rejects a count beyond the supplied pass list.

The prefix executes in the upload command buffer, committed before the draw command buffer
on the same queue. Its completion handler advances `upload_coherent_seq` only after every
staging read in the prefix finishes. The draw buffer executes the remaining passes and
advances `coherent_seq`. Both buffers retain their encoded Metal resources, and
`FramePayload` owns all command, descriptor and inline-byte backing until submission
returns. Mip generation after an upload stays in the upload prefix; generation after an
application render-target write or `StretchRect` remains ordered among the application's
passes.

Two kinds of Metal buffer the submit thread creates are reused across submissions rather
than allocated per use: the upload ring (`metal/transient.rs`) that inline indices and inline
vertex streams past `SET_BYTES_MAX` are copied into, and the private planes a depth transfer
stages through (`PlanePool` in `metal/depth_transfer.rs`). Both live on the device record and
stamp each use with the submission's sequence and the command buffer it went into, upload or
render. A region is written again only once the counter of every command buffer that read it,
`upload_coherent_seq` or `coherent_seq`, has reached the stamped sequence; the ring only
appends to a chunk in flight and starts one again from its first byte after that point, and a
plane set may also serve a later transfer in the same command buffer, whose encoders Metal's
hazard tracking orders. A submission without a sequence gets pools of its own, dropped with
it, since nothing it used could ever be seen to retire.

A third buffer, the present buffer, is committed by the presenter after the render buffer
and advances a unix-side per-device counter, `present_retired`, kept on the presenter state
because nothing on the PE side reads it. It registers in the same in-flight map as the
other two, under that counter's address, so the one retirement wait serves presentation
too; the presenter's idle wait is the only caller, and it targets the last present buffer
the presenter committed rather than the last packet it consumed: a packet dropped for an
occluded window or a nil drawable has no buffer, so nothing retires its sequence and the
counter never moves for it. A present the GPU kills is logged by its
completion handler and never touches `failed_submit_seq`, so no upload is re-issued for
it. A snapshot buffer, one blit of the back buffer into a slot, precedes a frame's upload
and render buffers on the queue when the pending present had to be copied. Every buffer
registers at its commit and never before, so a submission that fails midway leaves nothing
registered that never commits.

## Texture attachment and sampling views

`TextureViews` carries linear and sRGB attachment handles separately from
linear and sRGB sampling handles. Identity-swizzled textures alias the existing
handles. A render target with a channel swizzle keeps renderable attachments
and pre-creates sampling views over the same storage. Bind getters load these
resolved handles directly; they do not create or discover views while drawing.
The creation batch owns one retain per distinct non-null handle, even when
several roles alias one native object.

Pass analysis resolves all views through one view-to-resource map. Its reverse
sRGB map contains only renderable attachment views. Release and rename detach
attachment selection immediately, while forward aliases survive final store
and clear-coalescing decisions until the existing GPU retirement boundary.
The retirement walk removes each alias before destroying its native object.

## The device record the wire names

`CreateCommandQueue` builds one record per D3D device on the unix side (`metal/record.rs`): the `MTLCommandQueue` and its retain, the presentation state the presenter thread and the submit thread share, and the device's presented-cadence probe. It hands the PE side a `DeviceRecordHandle` for it, which `DeviceInner` keeps and every later thunk that acts on that device carries: a submission, the wait-policy and idle-wait barriers, the creation-time clears, a read-back, and the `DestroyCommandQueue` that gives the handle back and frees the record. The unix side resolves a device by dereferencing what the caller holds, so nothing indexes devices by address and no `MTLCommandQueue` pointer rides the wire for the PE side to outlive.

The record's `Arc` is what orders its teardown: the presenter thread holds one while it runs, so the queue's retain, which the record releases when it drops, outlives the thread whatever the PE side does with its handle. A thunk that names a record already destroyed, or a device whose creation failed, returns without touching Metal, and what it answers depends on what the caller loses: a call that can simply be dropped warns once and reports success, while the retirement and present-idle waits, which a read-back, a `Reset` and shutdown are ordered by, log every occurrence at error level and report `STATUS_UNSUCCESSFUL`, so the PE side logs that the wait it asked for did not happen instead of continuing as if the GPU had caught up.

## One attachment record per device

D3D9 allows several devices per process, and the e2e suite creates two live ones. Everything the display decides for one device's window therefore lives on a per-device record on the unix side (`metal/macdrv/attachment.rs`), not in process statics: whether the layer carries the HDR configuration, the live EDR headroom, the present throttle, the window's occlusion, the backing scale published to the PE side, and the present-geometry streak that gates the MetalFX route. `AttachMetalLayer` registers the record, keyed by the raw address of the metal view it created, which is the handle the device's later thunks already carry: the presenter looks its record up by the `present_view` each packet carries, once per present, `DestroyCommandQueue` retires it by `view_handle`, `SetDisplaySyncEnabled` finds it by `layer_handle`, and `SetCursorOverlay` names it by the `view_handle` it carries. A thunk whose view has no record warns once and, for a present, uses the defaults a session on no display would (not occluded, headroom 1.0, no throttle, the stretch route).

The record is live exactly while it is in the registry map. The view and layer addresses and the two PE-side sink addresses it holds are valid only then: `DestroyCommandQueue` unregisters the record before it retires the view, and the PE side drops the box behind the sinks (`DisplaySinks`, owned by the device's `CursorState`) after that thunk returns. So every dereference of one of those addresses happens inside a registry helper that holds the map's lock and checks the record is still the one the map holds for its view, by `Arc` identity rather than by key, so a view address the allocator hands out again names a new record. The lock is held for a lookup plus one retain or one atomic store, never across `AppKit` work, and is taken nowhere else. Outside those helpers a record is plain data (atomics and immutable words) that the main-thread observers, the presenter thread and the API thread read freely.

A retired metal view is kept, not released (`retire_metal_view` in `metal/macdrv.rs`). Wine creates a client surface with a cocoa view of its own for every `get_win_data`, and a metal view in it for every `macdrv_view_create_metal_view`, so a device that goes through Wine gets a new `CAMetalLayer` every time; the kept view is instead handed to the next device that attaches to the same `HWND`, without going through Wine, and attach configures its layer exactly as it would a new one. The layer is kept because Metal's per-layer frame metrics, the GPU time the Metal HUD shows, follow the layer: a `CAMetalLayer` created later in the process reports no GPU time for its frames until it has presented more of them than any layer before it did, while a command queue replaced behind an unchanged layer leaves the metric intact. A few views are kept (`KEPT_METAL_VIEWS`, one per window, the oldest displaced and released when the park is full), and parking a view drops its layer's drawable pool, which is the window's size and which nothing presents into while the view waits: `CAMetalLayer` has no call that empties the pool, but writing `drawableSize` does, so the park writes the smallest size Metal takes and attach puts the layer's own geometry back with the rest of its configuration. What a parked layer still holds is the surface it displays, the frame the window shows until its next device presents. A kept view is reused inside its `HWND`'s Cocoa window (`macdrv_get_cocoa_window` says which window that is now, and none for a destroyed handle). One whose handle has no window any more is not released but moved into the next window that attaches with no kept view of its own (`adopt_metal_view`), which is what an application that destroys its device window between two devices and creates a new one needs for its layer to follow: Wine creates the client surface for the new window as for a new view, a synchronous call through Wine (`macdrv_view_get_metal_layer` on the kept view, which doubles as the check that it still carries the parked layer) orders the move after that surface's own queued frame, superview and unhide requests, since Wine's main-thread request queue runs in order and its synchronous calls ride it, and the view then takes the place in the surface's cocoa view `newMetalViewWithDevice` gives a new one: the client view's bounds, its subviews resizing with it, below every other subview, and the window's `windowDidDrawContent` through a local `WineWindow` binding, after which Wine treats the window as having content. A missing Cocoa window for an HWND alone does not establish an orphan: a live child HWND has no Cocoa window of its own, so the parked view must also be detached or hosted by a closing Wine window. The host check retains the view under the park lock on the main thread after verifying the snapshot still names the current slot, then releases the lock before walking AppKit; taking the slot checks its sequence again. The park lock is never held across a Wine call or a main-thread hop; `macdrv_get_cocoa_window` is called on the API thread only, since Wine holds its window-data lock across a synchronous main-thread request while it destroys a window. A kept view whose handle still has a window, on screen or not, belongs to that window and is never moved.

The MetalFX caches follow the same rule by another key: the scratch texture a readback resolve or an HDR present tone-maps into, and the `MTLFXSpatialScaler` that enlarges the frame, are both cached per command queue and geometry (`metal/upscale.rs`), the queue being the device identity every readback and every submit carries, and `DestroyCommandQueue` retires the queue's entries of both after its shutdown fence. Metal orders command buffers within one queue only, so a scratch shared by geometry alone let one device's resolve land between another's resolve and its blit, and each read the other's frame. A scaler is stateful on top of that, since its colour and output textures are properties the encode that follows reads, so a shared one let two devices presenting at one window size write each other's drawables; the cache lock is held across those property writes and the encode so that the separation does not rest on how many threads one device encodes from. Its bound (`MAX_CACHED_SCALERS`, sized for one window being resized) is per queue, and so is the deferred release of an eviction: a scaler evicted by one queue is released from a completed handler on a command buffer of that same queue, which is the only ordering Metal offers.

A scaler entry also owns its optional Private output texture. Presentation
uses the drawable directly only when its actual storage and usage satisfy the
scaler. Otherwise MetalFX writes this intermediate at the drawable's exact
extent and format, then a blit copies it into the drawable in the same command
buffer. The intermediate follows its scaler through the per-queue eight-entry
bound, eviction completion and shutdown retirement; it never enters the
separate scratch cache. Preflight evictions are retired on fallback submissions
too. Scaler texture bindings are cleared after encoding so idle cache entries
do not hold drawables out of the layer's pool. SDR uses Perceptual mode; HDR
still tone-maps at render size before scaling in HDR mode. A preparation or
copy failure returns to the original source's SDR or HDR present shader.


A Reset that changes `PresentationInterval` reaches the record through `SetDisplaySyncEnabled` on the encoder thread, whether it flips the vsync request (`IMMEDIATE` against the rest) or only moves the frame-rate ceiling (`ONE` against `TWO`, `THREE` or `FOUR`, which the PE side resolves to the reported refresh rate over N and folds with `present.maxFps` into the one ceiling the thunk carries). The thunk latches the new pacing on the record and queues that same reconciliation, so the throttle is re-derived on the main thread for the panel under the window within one present and nothing on the encoder thread reads a screen.

The process-lifetime observers walk the records rather than a latch: the occlusion observer marks every record whose window posted the notification, and a real screen-parameter change reconciles every live record against the screen its window is on now. What stays process-wide stays so on purpose: the screen-parameter filter and Wine's application delegate (a relationship with the one `NSApp`), the observer install latches, the cursor overlay (one system cursor, one overlay window), and the presented-cadence debug probe, into which two presenting devices interleave.

## The cursor overlay window

With `cursor.software` resolved on (the default under HDR), the PE side keeps a
blank HCURSOR realized over the client area. `SetCursorProperties`, `ShowCursor`
and retargets reconcile the software sprite and effective visibility, including
unchanged hashes. Pixels are sent until the Unix side acknowledges the upload;
a rejected hash-only update gets one full-pixel retry. Both cursor modes validate
A8R8G8B8 format, dimensions, scaling arithmetic, pointer and row pitch before
reading the bitmap. The pure validation lives in `mtld3d-core`; the PE wrapper
balances each successful COM lock with an unlock and preserves the previous
cursor on rejected input.

An identical accepted request preserves pending retries but does not dispatch
another apply once that state has completed. The pre-commit observer and the
display observers continue reconciling it.
The overlay still reconciles its layer configuration and its window frame while
hidden or inactive, but defers pointer geometry queries until it can show a sprite. The
show resolves current geometry before presenting pixels and position together.
Hardware-only processes publish cursor visibility without creating an overlay;
a native hide still wakes the main-thread pointer watch. After any software sprite
has been accepted, hardware takeover retains the apply needed to clear previous
software content, including failed work.

The Unix cursor mutex publishes the attachment's `Arc` identity, mode, sprite,
visibility and request revision together. The sprite store, the overlay's
textures and completed images, the PE side's uploaded set and its HCURSOR cache
are all bounded to `CURSOR_SPRITE_CACHE_ENTRIES` least-recently-used entries: a
hash-only request for an evicted sprite is rejected and the PE side sends the
pixels again, and an evicted HCURSOR is destroyed once it is neither the realized
nor the thread cursor. Admission resolves the attachment
registry while holding that mutex. Unregister releases the registry lock before
detaching cursor state, and detach compares `Arc` identities, so an old device
cannot clear a new owner even if its view address was reused. Hardware takeover
clears the software sprite. Uploaded sprites remain content-addressed and shared
between devices; a native reconciliation owns one attachment and sprite snapshot
throughout its work. Metal allocation and pixel upload happen outside the mutex.

There is one stationary, borderless, click-through overlay window, one level above
the followed game window. It is an AppKit child of that window, so it follows
native fullscreen Spaces as well as ordinary desktops. Each reconciliation binds
it to the live attachment's window; device retirement detaches and orders it out.
AppKit's weak parent reference does not retain a destroyed game window. Cursor
show/hide still changes only the image, not window ordering. Its sprite sublayer
mirrors the game layer's actual pixel
format, colorspace and EDR setting, including handoffs within the same HDR/SDR
class. A device change rebuilds the command queue and texture cache. Changes to
sprite geometry, layer configuration, owner or relevant headroom invalidate the
rendered content. The overlay window has the game window's frame, not its
screen's: Mission Control outlines a window together with its children, and a
screen-sized child made that outline cover the screen. As a child it moves with
the game window, so only a size change or a new game window re-frames it; the
sprite is clipped at the game window's edges. Pointer motion changes the sprite
layer position. Moving the window itself per event would make AppKit resolve the
cursor again and replace the game's blank cursor with an arrow; the one
re-resolution a re-frame costs is undone by the native blank repair below.

The main-run-loop observer at before-waiting and exit, ahead of Core Animation's
commit, is the one place the cursor is reconciled. It reads the pointer position
and returns unless that moved or an apply was requested: the thunk, a detach, a
GPU completion, a present, an activation change and a headroom refresh each
request one. No event monitor is involved: Wine can consume captured mouse events
before AppKit's `sendEvent`, and a warp the game makes through winemac delivers
no event at all, but both run on the main thread and move the pointer, which the
next observer pass reads. Presents only request a coalesced check.

Before the first native mouse event, Wine may have accepted a Win32 cursor
without delivering it to macdrv: the server has not yet associated the stationary
pointer with a Wine window. AppKit can also replace the native image during focus
or window changes while Wine still records it as hidden. Both device and swap-chain
Present paths query `GetCursorInfo` in their shared frame submission code, only for
the device's foreground HWND. A changed native hide is
published through `CursorOverlayFlags::NATIVE_HIDDEN`, including when a game draws
its own cursor and never supplies a D3D cursor surface. This adds no cursor image
or Metal window for such a game.

The main-thread pointer watch owns one native blank image. It selects that image
when the software overlay is visible or its software cursor is natively hidden,
only over the active, unobscured game client area and outside external captures.
It compares the current native cursor by identity so an unchanged blank needs no
setter call, while an AppKit replacement is repaired at the next reconciliation.
This does not move the pointer, synthesize input, or change cursor hide counts.
The image the
blank displaced is never put back: Wine selects its own cursor again on every
handle change, and what was displaced may be the arrow AppKit resolved for the
pointer rather than Wine's cursor. Device release
replaces an owned hidden blank HCURSOR with null before freeing it; visible
cursors still restore the window's class cursor.

Hardware cursor images and their hide/show transitions belong exclusively to
Wine. The sampled native-hide bit can lag behind a Win32 show; applying a blank
from that snapshot would overwrite Wine's restored cursor until pointer motion.

The hit test serves both cursor modes. While the cursor is shown or natively
hidden and the application active, the window a click at the pointer would land
on is read: the game window means the pointer is over the game, a window of this
process over it (a dialog) hides the sprite and needs nothing else, and a window
of another process inside the client rectangle is an external capture, such as
the screenshot tool, whose own window takes the hit the moment it appears. The
return of the hit test to the game window requests the existing null-then-set
kick through live attachment sinks, restoring Wine's native cursor after the tool
left the system one behind. The callback that asks for this kick uses the
attachment registry's lifetime checks.

Changed sprites are rendered offscreen with the cursor tone-map pipeline. GPU
completion wakes the existing observer; it never waits for scheduling or execution
on main. The completed, CPU-visible texture is copied to an immutable CGImage.
Managed textures receive a synchronization blit before that completion. The image
and current pointer position are assigned in one Core Animation transaction. A
cached transparent image represents hidden content, completed images are kept
per sprite hash, so hide/show and a return to an earlier sprite reuse them
without another GPU submission, and while a changed sprite renders the sprite on
screen keeps following the pointer with its own geometry.

The cursor's CAMetalLayer hosts images for its macOS 15-compatible HDR controls;
it never acquires or presents a drawable. This leaves the game as the only drawable
stream eligible for Metal HUD selection. Disabling the HUD on a cursor drawable
layer is insufficient: it can still affect the game's HUD scale during device
recreation. The image-hosting window stays across attachment changes.

Submitted content is tracked separately from successful completion. Each submission
owns an atomic completion result and generation; a stale callback only updates its
own result, never the newer owner's state. Callbacks retain no PE pointers or native
UI objects.
Creation, allocation, encoding and completion failures leave the latest request
pending for existing event, run-loop or present opportunities. Reentrant callbacks
cannot settle a newer request, and failures do not start immediate retry loops.

The cursor log targets record rejected uploads, visibility blockers, input routes
(at trace level), layer configuration, submitted generations, completion and failure
stages. `scripts/cursor_transaction_probe.swift <output-directory>` captures native window
pixels and asserts the final visibility after coalesced clear/show bursts; the
output directory must already exist and screen recording access must be available.
`scripts/cursor_startup_probe.swift <app> <output-directory> <x> <y>` starts a
closed app with a stationary pointer and captures the system cursor as well as
the rendered scene before and after the first movement. The capture with the
system cursor excluded distinguishes a native arrow from the software sprite.
The visible Wine probe in `windows/tests/examples/cursor_capture.rs` exercises
`SetCapture` without clipping, loading pauses and hide/show bursts. Its native
sprite and completion log must be checked separately from the game backbuffer.

## Raw pointers across the boundary need stable backing

Commands carry `u64` param fields the unix side dereferences (`setVertexBytes` ptrs, `commands_ptr` inside `PassDescriptor`, the `PassDescriptor` array itself). The backing must not move between hand-out and `unix_call` return.

A growing `Vec<u8>` silently reallocates on capacity growth, invalidating every previously-returned pointer; the unix side then dereferences freed memory, Wine's SEH shim translates SIGSEGV to `STATUS_ACCESS_VIOLATION` (`0xc0000005`), and the PE side sees `unix_call` return non-zero. Prefer `Vec<Box<[u8]>>` (one heap block per allocation) or a chunked bump allocator where chunks never move. `FrameEncoder.scratch` is the canonical example.

## `status=0xc0000005` from `unix_call` is a unix-side SIGSEGV

Wine's unix-call dispatcher wraps each handler in a SEH-translation shim. Any `0xc0000005` means the unix side crashed mid-call — the PE side's own early-return error logs (`queue retain failed`, `renderCommandEncoderWithDescriptor returned nil`, …) will **not** fire. Diagnose by instrumenting each step of the unix-side path with a log line keyed to what it's doing; the last line printed before the PE-side error is the crash site.

## Shared wire values are typed in `unix/shared/src/mtl.rs`

Any integer crossing the boundary with *symbolic* meaning — Metal enum codes (storage mode, pixel format, compare func, blend factor, primitive type, sampler filter/address, …), bitflag masks (texture usage, color write mask), stage selectors — is declared **once** in `mtl.rs` as a `#[repr(u32)]` enum (single-choice) or `bitflags!` struct (multi-bit). Param fields in `params.rs` use that type directly, *not* `u32`.

**Never** restate the encoding as a local `const`. **Never** write an integer literal at a call site or decode arm.

The three cdylibs are separate linkage units. An untyped integer is a silent-drift risk: PE adds a variant, Unix decode forgets, a `log_once_warn!` covers it. Typed fields move the contract into the shared crate; exhaustive `match` becomes a compile error the instant a variant is added. Sound because `make` / `make install` rebuild and copy all three together — the "unknown variant" UB bit pattern never appears on the wire.

How to apply:
- New thunk field with symbolic meaning → `mtl::` type. Sizes/offsets/counts/`!= 0` booleans → `u32`.
- Adding a value: extend `mtl.rs`, extend `d3d_to_metal_*` if one exists, the compiler points at every Unix-side match site.
- Bit-flag fields use `bitflags!` (`TextureUsage`, `ColorWriteMask`).
- `Command::param_a/b/c/d` carry polymorphic `u32`s whose meaning depends on `Command::cmd`. Stay `u32` on the struct; encode via `Enum::Variant as u32` in the `Command::foo` constructor and decode via `Enum::from_repr(raw)` (strum `FromRepr`) in the dispatcher — never a bare `match raw { 0 => …, 1 => …, … }`.

A `BlitCommand::CopyTextureToTexture` carries its copy depth in `depth`,
starting at z=0 at both ends. Zero keeps the original one-slice form for 2D
and cube commands; volume preservation supplies the addressed mip's full
depth. The unix side bounds-checks that depth against both live textures'
mip dimensions before encoding the copy. Array slices remain separate in
`src_slice` and `dst_slice`.

## The drawable is the layer's size, and present owns the resample

`CAMetalLayer.drawableSize` is kept at the layer's own `bounds × contentsScale`, never at the guest's back-buffer size. `macdrv::sync_drawable_size` pushes it at attach and again on the presenter before every `nextDrawable`, because the documented default is captured once and does not follow the layer: a freshly created wine metal view reports a real `bounds` beside a `0x0` `drawableSize`, and a window resize moves `bounds` without moving `drawableSize`. The one layer whose `drawableSize` is not its own geometry is a parked one, shrunk to drop its pool while no device presents through it and restored at the next attach.

That is what makes the composite pass a 1:1 copy. A drawable that is not the size of the layer's backing store gets rescaled by the compositor, on top of whatever present already did, and the phase of that second resample is not ours to control: a ratio near 1.0 shows up as the whole frame, interface included, sitting a pixel off where it was drawn. Re-syncing per present rather than per `Reset` is what covers the frames between a window resize and the guest reacting to it. The same reasoning is why `contentsGravity` is inert here rather than load-bearing.

So every back-buffer-to-drawable difference is resolved by the presenter's present buffer, by exactly one of three routes (`present_route` in `command.rs`): a 1:1 blit at matching extents, `MTLFXSpatialScaler` when the drawable is larger in both axes and the GPU has MetalFX, and the present shader's filtered stretch for everything else. Only the third covers any ratio, so it is also the backstop when a scaler declines — `MTLBlitCommandEncoder` cannot resample, and a partial copy would leave the rest of the drawable undefined.

## Adding new thunks

1. Add variant to `Thunks` enum in `mtld3d-shared` `lib.rs` (count and iteration via strum).
2. Add param struct in `mtld3d-shared` `params.rs` (`#[repr(C, align(8))]`). Field types: `u64`, `u32`, `#[repr(u32)]` enums from `mtl::`, or `bitflags!` structs from `mtl::`. Symbolic-meaning integers must use `mtl::`. `impl Thunk` with the matching code.
3. Add handler in `mtld3d-unix`, add arm to `dispatch()` (exhaustive match = compile error if forgotten).
4. Call via `unix_call(&mut params)` from `d3d9`.

## Label every Metal object created

Every `MTLDevice.new*…`, `MTLCommandQueue.commandBuffer()`, `MTLCommandBuffer.{render,blit}CommandEncoder*`, and per-stage descriptor that produces a Metal-side state object must get a `setLabel:` call before it's handed back across the boundary or used. Strings start with `mtld3d-` followed by the role and an identifying suffix — `mtld3d-tex-{tex_id:#x}`, `mtld3d-vbib-{buffer_id:#x}`, `mtld3d-frame-{submit_seq:#x}`, `mtld3d-pass-{idx}`, `mtld3d-samp-{key:#x}`, `mtld3d-backbuffer`, `mtld3d-depth`, `mtld3d-readback`, `mtld3d-mipgen`, …

Xcode GPU frame captures, Metal validation logs, and the Metal HUD display these labels everywhere they show a Metal object. Without them every handle shows up as `Buffer (8KB)` / `RenderCommandEncoder` / `Texture (BGRA8 1024×1024)`, which makes any handle-recycle / cross-device-alias / contention investigation start with "and which one is this?". With them the mapping back to a mtld3d-side identity (`TextureId` / `BufferId` / `submit_seq` / pass index / packed-bits state key) is one column in the resource browser.

How to apply:
- For *create-style* thunks (texture, buffer, sampler, DSS), the param struct on the PE side carries an `id: u64` (and a `kind` enum where one struct serves multiple roles, e.g. `BufferKind` on `CreateBufferParams`). PE side fills it from the appropriate strong-typed identifier (`tex_id.raw()`, `buffer_id.raw()`, `SamplerKey::raw()`, `DepthStencilKey::raw()`); unix side composes the label string and calls `setLabel`.
- For *per-frame* objects created entirely on the unix side (the per-frame `MTLCommandBuffer`, per-pass `MTLRenderCommandEncoder`, blit encoders, mipgen + readback transients), label inline at the create site using whatever in-scope identity disambiguates instances (`SubmitFrameParams::submit_seq`, the `pass_idx` loop variable, a static role string).
- For *descriptor-then-state* paths (`MTLRenderPipelineDescriptor`, `MTLSamplerDescriptor`, `MTLDepthStencilDescriptor`), call `setLabel` on the **descriptor** before the `newXxxStateWithDescriptor:` call — the label propagates onto the resulting state object.

Trait-import caveat: `setLabel` lives on different traits depending on the object. `MTLBuffer` / `MTLTexture` / `MTLSamplerState` / `MTLDepthStencilState` need `use objc2_metal::MTLResource;`. `MTLRenderCommandEncoder` / `MTLBlitCommandEncoder` need `use objc2_metal::MTLCommandEncoder;`. `MTLCommandBuffer` and `MTLCommandQueue` provide it on their own protocol traits, no extra import.

Cost: one `format!` + one `NSString::from_str` + one objc dispatch per create call. Negligible — paid only at object-create time (cache miss / per-frame at most). Ship unconditionally; never gate on `cfg(debug_assertions)`.

## Logging

Every crate logs via `log` + `env_logger`. All targets sit under `mtld3d::*` and `env_logger` matches by `::`-separated prefix, so `RUST_LOG=mtld3d=warn` is the single switch for the whole project; unset, everything logs at `info`. Levels: `info!` for one-shot milestones, `warn!` for unimplemented stubs and fallback paths, `error!` for unexpected internal failures, `trace!` for per-call breadcrumbs, `debug!` for routine per-call noise useful in deep debugging.

| Target                    | Scope                                                                    |
|---------------------------|--------------------------------------------------------------------------|
| `mtld3d::d3d9`            | `windows/d3d9/` + `windows/core/` (everything except `dxso` and `perf`)  |
| `mtld3d::d3d9::cursor`    | hardware cursor (HCURSOR) lifecycle, bitmap cache, wndproc               |
| `mtld3d::d3d9::display`   | fullscreen mode-set and restore, display-mode enumeration probes (trace) |
| `mtld3d::d3d9::passes`    | pass-break and pass-open probes, per-pass and per-RT shape rows (trace)  |
| `mtld3d::d3d9::state`     | every RS/TSS/SAMP write the game makes (trace)                           |
| `mtld3d::d3d9::cascade`   | shadow-map cascade summary per frame, caster writes vs samples (trace)   |
| `mtld3d::d3d9::depth`     | depth-stencil binds, per-stage depth-sampler mask, load actions (trace)  |
| `mtld3d::d3d9::tex`       | texture create, lock/unlock dirty flags, bind-time mip flush (trace)     |
| `mtld3d::d3d9::blit`      | accepted `StretchRect` blits, including the scaling render path (trace)  |
| `mtld3d::d3d9::draw`      | per-draw breadcrumb (trace)                                              |
| `mtld3d::d3d9::sampler`   | sampler-state translation (trace)                                        |
| `mtld3d::d3d9::caster`    | one row per unique shadow-caster pipeline state (trace)                  |
| `mtld3d::d3d9::decal`     | the depth bias applied per (VS, PS) pair and depth state (trace)         |
| `mtld3d::dxso`            | DXSO to MSL emitter (`trace` dumps the MSL)                              |
| `mtld3d::perf`            | 5-second averaged performance summary (`PERF=1` builds only)             |
| `mtld3d::shim`            | Wine unix-call PE shim DLL                                               |
| `mtld3d::unix`            | Metal-side `.so`                                                         |
| `mtld3d::unix::command`   | command-buffer completion/error and backbuffer allocation/view records (debug) |
| `mtld3d::unix::cursor`    | software cursor: input routes, blockers, uploads, submissions and completion    |
| `mtld3d::unix::present`   | presented-cadence probe, one row per frame (trace)                       |
| `mtld3d::unix::depth`     | comparison-sampler creation, the unix mirror of `d3d9::depth` (trace)    |

Each cdylib initializes the logger independently and idempotently; `mtld3d.so` has no owning entry point, so `d3d9.dll` dispatches a one-shot `InitLogger` thunk from its init path. Every line goes to the process's log file, `<exe>-<pid>.log` under `mtld3d-logs` beside the executable (`log.dir` moves it), never to the standard streams: a game a launcher spawned has no usable ones. `<pid>` is the macOS process id, so a launch never overwrites the log of the one before it; the directory keeps the ten newest logs and the ten newest traces, and the file appears with the first line written, so a process that logs nothing leaves nothing behind.

The Unix initialization also records CoreFoundation's loaded path, Mach-O
header address and UUID at `info` level under `mtld3d::unix`. It resolves the
address of the linked `kCFRunLoopCommonModes` export without reading a CF
object. This identifies the framework actually mapped in that process,
including an image in the dyld shared cache. Missing path or UUID information
is explicit. The record uses the same startup backlog and process log as the
build stamp; the allocating loader query never runs from a signal handler.

### Command-buffer completion and encoder errors

`RUST_LOG=mtld3d=warn,mtld3d::unix::command=debug` enables
`EncoderExecutionStatus` collection for frame, upload, synchronous readback and
creation-time texture-clear command buffers. These buffers keep retained
resource references in both modes; with the target disabled, creation uses
the ordinary `commandBuffer()` path.
Collection can add CPU, GPU and memory overhead, and logging every completion
can be verbose. Use a bounded workload when gathering diagnostics.

The frame/upload callbacks, retirement waits, CPU submission cleanup,
readback wait and diagnostic-only initialization callback log `command-buffer`
records with the actual buffer, queue and device addresses, device registry
ID and name, labels, role, sequence where known, observation site,
numeric/named status and error options. Roles come from
the constructor-owned labels; an unrecognized or missing label reports `unknown`.
Readback and initialization sequences are `unavailable`. Initialization uses
the exact `mtld3d-init-clear` label and `initialization-callback` observation
site. One buffer can appear at several sites, so correlate the addresses and
site with the sequence and log order. Addresses can be reused after release,
and log order is not a causal order across queues.
No new wait is added. The initialization callback adds scheduling and logging
work only with diagnostics enabled, so captures can alter failure frequency.
Only a recorded `Completed` status establishes successful completion of that
observed buffer; absence of an error record does not.

On failure, the following `command-buffer-error` record names the same buffer,
sequence and site and includes the signed `NSError` code, domain and description.
It checks the encoder-info array and each element's protocol conformance before
printing labels, numeric/named states and signposts in recorded order. Variable
strings are quoted and escaped to keep each record on one line. Missing errors,
missing keys, malformed payloads, empty arrays and unavailable protocol metadata
remain distinct. Encoder labels and signpost arrays are read through Foundation's
nullable key-value getter after checking `NSObject` inheritance. Nil metadata is
reported as `missing`, an empty signpost array as `empty`, and wrong value or
element classes as malformed. A conforming encoder outside `NSObject` retains
its state with object metadata reported as `unavailable-non-nsobject`.
No per-draw signposts are inserted, so existing labels can be all the driver has.
`Unknown`, `Completed`, `Affected`, `Pending` and `Faulted` remain distinct:
`Affected` does not establish that the encoder caused the error, and an encoder's
`Completed` state does not make the whole buffer successful.

The initialization callback captures no resource, PE memory or caller borrow.
Its executable lifetime depends on the D3D loading contract: `CreateDevice`
sets the used latch before a clear can be submitted, and D3D DLL detach then
self-terminates before Wine unloads its statically imported shim and Unix
image. Native unit clients compile this code into their test executable.
This is not a general contract for a client that directly loads and unloads
the shim or Unix image. A change allowing surviving D3D unload after
`CreateDevice` must revisit the callback. Process exit can end callbacks
before they log; the shutdown fence does not prove all earlier callbacks
finished, and missing records remain unobserved completions.

The target also records each successful backbuffer's base and optional view
and MSAA handles with its request device and queue. Each refused sRGB view
records the live base texture and device identities, actual base label,
dimensions, format, usage, type, mip and array sizes, and requested view
format, type, ranges and swizzle. Extra view queries run only in the enabled
refusal branch. The ordinary allocation errors carry the live allocation
device and registry ID; backbuffer failures carry the request device and
queue handles. Join these records within the process and creation interval:
addresses can be reused, and system driver errors without texture handles
cannot be matched one-to-one by timing alone. A refused optional view does
not establish a failed base allocation or memory exhaustion.

Cursor-overlay buffers and the empty teardown fence are outside this
target's construction and observation scope. A clean local run validates
construction and completion on that device; it does not demonstrate a real
error payload or establish another GPU's fault attribution.

The same debug target records synchronous encoding metadata. `render-pass`
reads the assembled descriptor just before render-encoder creation: four color
slots, depth and stencil, their actual texture and resolve bindings, subresources,
load/store/clear values and resolve filters. `commands` counts the existing
command list, including state setters; it is not a draw count. `texture-copy`
records accepted texture-to-texture endpoints, slices and validated region.
`readback-copy` records the final selected source after any readback resolve or
fallback, the actual destination Metal buffer, PE destination address/length,
and encoded offset, row pitch and image pitch. Image pitch counts block rows
for compressed textures. The existing `readback-wait` record supplies completion.
Texture extents in these records are base-level extents; `level` selects the mip.

The depth transfer behind RESZ is one of those copies. It records its two ends
as a `texture-copy` at site `depth-transfer/0`, before any of its encoders
exist: the live source and destination textures, their levels, their sample
counts and the destination region. It reaches the destination through private
depth and stencil planes rather than one `copyFromTexture:`, so that record
states the transfer it was asked for, not the operands of a single encoded
copy. When the sample-zero compute pass runs, a `depth-transfer-resample`
record follows at site `depth-transfer/1` with `sample=0`, naming the private
multisample copy the kernel samples and its stencil view, or the planes
extracted from a single-sample source, the output planes the destination copy
then inserts, the source and output regions, the kernel's four row strides
(depth in floats, stencil in bytes) and the dispatch grid and threadgroup. A
transfer whose source is already single-sampled at the destination's size needs
no compute pass and emits no such record, so its `texture-copy` stands alone.

These records include command-buffer pointer, label and queue, plus pass or
blit site, so a producer attachment can be followed through copies into a
readback destination. All additional property queries and formatting are behind
the debug filter. They establish encoding inputs, not pixel correctness or
successful encoder creation. The attachment count is fixed, but label lengths
and the number of passes and copies determine output volume. Measure lines and
bytes on the focused test before enabling this target for a full-suite capture.
Join within a process and ordered resource lifetime: pointers and per-device
sequences can be reused, and successful full-suite logs lack direct test identity.
No pixel contents are read by these records.

For a bounded manual CI run, set `e2e_filter` to
`msaa::depth_test_holds_on_a_multisampled_target` and `e2e_log` to
`mtld3d=warn,mtld3d::unix::command=debug` in the workflow dispatch form. The
existing e2e steps pass `e2e_log` as `RUST_LOG` and retain their process logs in
the e2e artifacts. An empty input preserves the ordinary logging default.

### F12: three-frame dump and GPU capture

Pressing F12 in a game records the next three frames twice over. The log gets one `[dump]` line at info level for every D3D9 event of those frames that a GPU trace cannot show: render-target, depth-stencil, viewport and scissor changes, clears, surface copies, occlusion query traffic, and every draw with the states that decide its pass shape, its shaders and its textures. At the same time a Metal GPU trace of the same frames lands beside the log file as `<exe>-<pid>-<n>.gputrace`, numbered per press; the process needs `MTL_CAPTURE_ENABLED=1` in its environment for that half, otherwise the log says so and the dump still runs. The two sides name each other: each `frame end` line carries the label of the frame's command buffer, and every dumped draw sits in a `draw N` debug group in the trace, so `gpudebug`'s `find` or Xcode's search lands on it directly.

### The Main Thread Checker

AppKit's views, windows and screens belong to the main thread, and the layer touches them from the API, encoder, submit and presenter threads only through `run_on_main_thread_sync` or the main-queue dispatches of the cursor overlay. A call that slips past that rule does not fail where it is made: it corrupts state AppKit keeps on the main thread and the process dies later, inside an autorelease pool pop in Wine's own code, with no frame of the layer on the stack. A checked `MainThreadMarker` catches the class methods that take one; it cannot catch an instance method on a view or window the code already holds, such as `-[NSView window]`, nor anything the Wine driver does.

`debug.mainThreadChecker = true` loads Apple's Main Thread Checker (`/usr/lib/libMainThreadChecker.dylib`, which lives in the dyld shared cache) from the `OpenLog` thunk, the first thunk that runs with the configuration resolved and, since `mtld3d.so` links AppKit, after the framework is in the process; inserted at launch through `DYLD_INSERT_LIBRARIES` it reports nothing under Wine. Loaded, it swizzles every AppKit method that requires the main thread and writes `Main Thread Checker: UI API called on a background thread: -[NSView window]` plus a `Thread name:` line to stderr for a call from any other thread. `make test` sets the key in `MTLD3D_CONF_TEST` and exports Apple's `MTC_CRASH_ON_REPORT=1`, under which the checker ends the process at the report, on the offending thread, so the e2e runner charges the death to the test that made the call and keeps the process's stderr with the report in it. The key is for the suite: a game runs without the checker and without the swizzle.

## Perf infrastructure

The `mtld3d::perf` summary in `windows/core/src/perf.rs` is compiled in only on a `PERF=1` build (`cfg(perf_tracking)`) and emits a multi-line report every 5 s at `info!` under `RUST_LOG=mtld3d::perf=info`. Counters group by which thread owns them (API, encoder, submit, presenter); subtimers indent under their parent. The `Submit thread` block reports `Encode+commit` (the thunk's execute less the wait) and `Present wait`, the wait for the previous present to commit; the `Present thread` block reports `Drawable wait`, still the `gpu_wait` bucket, and `Snapshots`, an event count of the presents that went out from a copy of the back buffer: none in steady state, one per read-back; `Slot waits` counts the copies that first waited for a slot, a wait on the display and the tripwire for the ring's size, 0 being the goal. Both blocks come back with the next payload, lagged one frame, and a barrier's snapshot carries over to the next sample rather than being reset. Banner shows `bottleneck=…` based on `present_block` share + `gpu_wait` vs `enc_cpu`; the four terminal buckets are echoed on a `buckets:` line for auditability. The same Info gate also enables the per-call cycle accounting — single switch. Pass / workload shape (per-pass dump, `present_texture=…` audit line, per-RT pair stats) lives on the separate `mtld3d::d3d9::passes=trace` switch — those are diagnostics, not perf metrics.

`Encode+commit` splits into three children the unix side times with `NanosSetTimer` inside `SubmitFrame`: the frame-leading blits, the replay of every pass descriptor (upload and draw, each pass's own blits included), and the frame buffer's completion-handler install plus both commits. A `resid` row takes what is left (command-buffer creation, the present settle less its wait, the upload buffer's handler, the thunk crossing), so the children add up to their parent. The `GPU` block reports `GPUEndTime - GPUStartTime` per command-buffer role (frame, upload, present) as ms per frame, with the number of buffers behind each and no peak, since a report does not line up with one frame. It is not the device's whole GPU time: snapshot copies, read-backs, creation-time clears, the cursor overlay and the shutdown fence are not counted, nor a frame or upload buffer submitted before its sequence or counters were wired. A synchronous submit behind a barrier is timed and folded like an async one, so `Encode+commit` and its children always describe the same submission. Each completion handler adds its buffer's time to the device record, and the next `SubmitFrame` of that device moves the sums into its `SubmitTimings` output and leaves zero behind, so every buffer is reported once, one submission after it finished. Buffers of one queue can overlap on the GPU, so the roles add up to busy time, not wall time, which the block's label says. All of it travels as nanoseconds in the fixed `SubmitFrameParams.timings` output; outside a `PERF=1` build, or with the perf target off, the handlers read no time and every field stays zero.

Each report identifies its owning encoder thread with `encoder=ThreadId(...)`,
which distinguishes D3D device instances without a shared counter. The interval
header gives its first and last device reset epochs. Inverse outcomes and
upload event counts are interval totals, not cumulative counters. A span
containing several epochs includes work before and after Reset. Reset flushes the old frame before advancing the API epoch.

The `inverse-view` rows partition consumed vertex draw uniform builds into
`bypass` (no active clip planes), `hit` (all sixteen view matrix bit patterns
match), and `recompute` (including the existing singular identity fallback).
Clean snapshots and dirty snapshots that do not rebuild this uniform contribute
nothing. `builds = bypass + hit + recompute`; `enabled-hit` divides `hit` by
`hit + recompute`, and is `n/a` for zero enabled builds. Each reset epoch has its
own row within a reporting interval, so reuse across frames is counted but
reuse across a cache reset is not implied. Counts saturate at `u64::MAX`; an
overflow of a count, total, or epoch sets `saturated=true` and makes the rate
`n/a`. Avoided inverse attempts relative to the original unconditional builder
are `bypass + hit`, not the number of draws or a frame-time gain.

This instrumentation adds an outcome store and a saturating counter increment
with an existing short API-state borrow per consumed build. Epoch aggregation
runs per frame; reporting formats once per interval. It adds no per-build timer,
allocation, atomic, or second matrix comparison. All fields and callsites
compile out without `PERF=1`. Use normal builds for timing comparisons.

Counter aggregation — mixing these up misreads the log:

- **Time counters** (anything ending in `ms`): per-frame averages.
- **Event counters** (passes, commands, draws, fresh, discards, wraps, …): raw window totals — divide by `frames=N` for a rate. Never average an event counter — silently rounds rare signals to zero.
- **Depth counters** (retention depth, retention KB): f64 averages, formatted `.1`.
- **Cache-size snapshots**: point-in-time at window emit, neither averaged nor summed.
- **Peak counters** (`peak …` cells): max value on any single frame in the window.

No ANSI colour anywhere: every line goes to the process's log file, and `env_logger` is told so (`WriteStyle::Never`) rather than left to auto-detect a terminal, which under Wine would be wrong in both directions.

### Buffer recycle-pool diagnostics

`PERF=1` and `RUST_LOG=mtld3d::perf=info` emit a `pagebox-pool cumulative`
line with the existing five-second performance summary. These totals cover the
process-wide pool, including all devices, and survive device resets. Multiple
devices can therefore report overlapping totals; do not add their reports.

Subtract consecutive lines to isolate a gameplay interval. Acquire attempts are
`hit + empty + oversize + disabled`; the enabled hit rate is
`hit / (hit + empty + oversize)`. Recycle attempts are
`recycle_parked + recycle_cap + recycle_oversize + recycle_disabled`.
`empty` means no retired box of the requested padded size was parked;
`oversize` means the request exceeded the largest pool class. `recycle_cap`
means the byte budget rejected a retired box. The two oversize byte fields give
cumulative requested logical bytes and the largest logical request, respectively.
They do not measure copied bytes.

All counters and their updates disappear without `PERF=1`. They share the pool's
mutex and preserve its allocation, ownership and retirement decisions. In a
performance build, early disabled and oversized outcomes also take that mutex
for accounting. Hit rate alone is not a speed measurement; correlate it with
the existing page-fault, allocation, rename, upload and frame-time rows before
changing pool limits.

### Shader and pipeline attribution

The same PERF summary appends cold-work accounting from
`windows/core/src/perf/compilation.rs`. VS and PS library misses include MSL
emission, native preparation, Metal library compilation, entry-point lookup,
and cache compression/write. Primary and no-color sibling PSO misses include
native descriptor preparation, the synchronous Metal PSO build, and recipe
compression/write. PSO cache persistence has its own nested row. Draw-path
depth-state misses are separate. Cache hits do not count as creation attempts;
a source-index miss that finds an already-prewarmed library is still a hit.
A library or pipeline build that fails is one attempt and one failure: the
source index or the pipeline cache remembers the key, its later draws are
dropped on the probe. A Reset that goes through the encoder's reset cleanup,
one that recreates the implicit surfaces, forgets the failures so each key
gets one more attempt; a Reset that keeps the back-buffer dimensions skips
that cycle and forgets nothing. A failed no-color sibling drops no draw; its
passes keep their color attachment.

Rows report total duration, ms/frame, peak summed duration on one encoder
submission, attempts (`calls`), and failures. Successful calls are attempts
minus failures. Nested rows overlap their parent totals and must not be added
to them. Resolve and pipeline remainders subtract measured children on each
submission before taking a window maximum. They include cache lookup,
bookkeeping, thunk overhead, and telemetry overhead; they are not a separate
Metal compilation phase.

Each five-second window retains at most five individual operations taking at
least 2 ms. Parent totals do not compete with their children for these slots.
Owned metadata is captured only for a retained operation and formatted with
the summary: device, encoder submission sequence, shader disk identities, and
for PSOs the vertex declaration/layout, attachments, sample count, blend/write
state, and sibling flag. `encoder_ops_same_submission` belongs to that same
submission. It does not correlate the operation with an unrelated API or
presentation peak. Pipeline builds already run on the encoder thread; a long
build there can delay encoding and eventually backpressure the API thread.

Native phase durations cross the PE/Unix boundary as nanoseconds in fixed
`repr(C)` output fields, including on failure. Unreached or disabled phases
are zero. `NanosSetTimer` measures each duration in its owning runtime; raw PE
and native ticks are never subtracted. The `TimingOutput` wrapper reserves the
same wire layout without PERF but elides its initialization, writes, and reads.
PERF callers initialize a zero fallback before crossing the boundary, so mixed
PERF/native builds are safe.
Collection and slow-event storage also compile out. The prewarm thread recreates
the deduplicated shader libraries and recorded render pipelines. Native creation
uses device-local batches capped at eight callers, including the coordinator,
and further limited by available CPUs and batch length. Regeneration and cache
appends stay serial. The library batch completes before dependent PSOs are
admitted, with one attempt per resolved PSO key in that startup. A failed worker
spawn reduces concurrency without dropping jobs. Cancellation stops new job
admission and waits for admitted calls before device cleanup; other devices keep
their own workers and stop flags. The encoder
installs their device-local handles and no-color sibling mappings before it
accepts gameplay submissions. Only the prewarm worker owns the startup channel's
sender. A failed thread spawn releases that barrier and starts the encoder cold
with persistent writes disabled, because the cache file was never validated.
Device release cancels and waits for prewarm before encoder cleanup.
Each device therefore creates its whole pipeline set again on the process's one
`MTLDevice`. The Metal HUD's "Pipeline States" figure counts pipeline-state
creations and never falls when one is released, so it rises by one set per
device recreation although no pipeline outlives its device; the per-encoder
`pipelines=` cache size in the `PERF=1` summary is the live count.
The prewarm thread logs startup compilation totals separately, plus elapsed startup time including cache I/O
and compaction. Shader identities and pipeline recipes share the translation
schema, while the container format has its own version. Each shader record also
carries a source-derived MSL emitter fingerprint. Programmable records retain
DXSO and the complete VS/PS specialization inputs; prewarm reparses them and
regenerates stale MSL before compiling libraries and dependent pipelines.
Regenerated records are appended before compaction and take precedence over
stale duplicates. Failed regeneration keeps the source for a later retry.
Fixed-function records have no retained source, so stale entries and their
dependent pipeline recipes are discarded. A stable sidecar lock
serializes reads, append operations and compaction; startup removes unreadable
tails under that lock before another process can append behind them.
Explicit `PERF=0` also overrides an inherited `MTLD3D_PERF` environment variable.

### Don't hand-roll `rdtsc()` brackets — use `perf::ApiTimer` / `CycleSetTimer` / `CycleAddTimer`

Time measurements that flow into the perf summary go through one of:

- `ApiTimer` — D3D9 vtable entry brackets, accumulates into `api_cycles_by_category[Category]`.
- `CycleAddTimer` — sub-scope inside an outer `ApiTimer`, accumulates into a `*mut u64` field (e.g. `query_wait_cycles`).
- `NanosSetTimer`: elapsed wall time in nanoseconds for native thunk outputs and cold compilation phases.
- `CycleSetTimer` — once-per-frame measurement that overwrites a `*mut u64` field (e.g. `present_block_cycles`, `op_cycles`, `submit_cycles`, `drawable_wait_cycles`, `present_wait_cycles`).

All four read `PERF_TRACKING_ENABLED`, a static `AtomicBool` latched once at
logger initialization from `log_enabled!(target: "mtld3d::perf", Level::Info)`.
A disabled helper reads no clock. The cached gate avoids a filter lookup on
every measurement. On a non-PERF build, the helpers compile to nothing
(`cfg(perf_tracking)`).

The summary itself emits at `info!` on the same target, so the user-facing switch on a `PERF=1` build is a single `RUST_LOG=mtld3d::perf=info` for both the cycle accounting and the rendered grid.

A second cached gate, `PAIR_STATS_ENABLED`, latched from `log_enabled!(target: "mtld3d::d3d9::passes", Level::Trace)`, fronts `bump_pair_stats` and the per-pass / `present_texture=…` / per-RT pair lines that `log_frame_summary` appends after the grid. Those are pass-shape and workload-shape diagnostics — they ride the same `mtld3d::d3d9::passes` target as the per-event pass-break / pass-open probes in `windows/core/src/passes.rs`, not the perf target.

The unix `.so` carries its own `PERF_TRACKING_ENABLED` + `CycleSetTimer` (in `metal/command.rs`) — each cdylib has its own `log` statics so the cache is per-runtime. Each cdylib calls its own `init_tracking_enabled` from logger init.

The two legitimate raw-`rdtsc()` use cases are (a) inside `mtld3d-core::perf` — the helpers themselves, frame/window boundary timestamps, calibration in `tsc.rs` — and (b) rdtsc as a *clock argument*, not a bracket — e.g. `BurstTracker::poll(now, …)` in shader-compile debounce.

## Debugging rendering bugs — shader/pass toolkit

Four off-by-default knobs answer "which shader on which RT produced the bad pixels":

1. **Pass × shader correlation log** — `RUST_LOG=mtld3d::d3d9=debug`. One `debug!` line per unique `(RT size, VS, PS)` triple; shaders tagged `prog 0x…` (content-hash, stable across runs) or `ff 0x…`. Implementation: `FrameEncoder::maybe_log_pass_shader` from `draw::emit_draw`.
2. **MSL dumps** — `RUST_LOG=mtld3d::dxso=trace`. Bracketed by `── VS MSL prog 0x… ──` / `── /VS MSL prog 0x… ──` (and PS).
3. **Raw DXSO bytecode dump** — `debug.bytecodeDumpDir = /tmp/mtld3d_shaders` in `mtld3d.conf`, or one-shot via `MTLD3D_CONFIG="debug.bytecodeDumpDir=/tmp/mtld3d_shaders"`. Writes raw LE `u32` token streams to `{vs|ps}_{id:x}.dxso` on `Create*Shader`. Idempotent per id.
4. **Offline disassembler** — `cd windows && cargo run --example disasm --target aarch64-apple-darwin -- /tmp/mtld3d_shaders/ps_<id>.dxso`. Prints raw tokens, parsed IR (`{:#?}` on `DxsoProgram`), and emitted MSL. Host-only, no Wine.

Typical workflow:

```sh
RUST_LOG=mtld3d=warn,mtld3d::d3d9=debug,mtld3d::dxso=trace MTLD3D_CONFIG="debug.bytecodeDumpDir=/tmp/mtld3d_shaders" ./<game>.exe > /tmp/trace.log 2>&1
```

Reproduce → grep `pass RT` for suspect ids → grep `── PS MSL <id>` for emitted MSL → `cargo run --example disasm` for deeper analysis → seed a regression test in `core/src/dxso/emit_tests.rs`.

## Debugging heap corruption — `MTLD3D_CRUMB=1` mmap breadcrumb

`unix/shared/src/crumb.rs` is a zero-I/O crash breadcrumb mapped at `Z:\tmp\mtld3d-crumb.bin` (= `/tmp/mtld3d-crumb.bin` under Wine). Probe calls compile to a single `mov [ptr], rax` when enabled and to nothing when disabled.

```sh
MTLD3D_CRUMB=1 make install   # cfg routed through build.rs
./<game>.exe                  # reproduce
xxd /tmp/mtld3d-crumb.bin     # read last-recorded state
```

`MTLD3D_CRUMB=1` is used instead of `RUSTFLAGS="--cfg mtld3d_crumb"` because cargo prefers env `RUSTFLAGS` over `[target.*.rustflags]` (does not merge), so the env approach silently drops xwin `-Lnative=…` paths. `windows/d3d9/build.rs` reads `MTLD3D_CRUMB` and emits the cfg through `cargo:rustc-cfg`, which composes correctly. The other way to add a flag without losing the config-file ones is a `--config` layer, which cargo *does* join into the existing array; that is how the Makefile's `FP=1` appends `-C force-frame-pointers=yes`.

Slot layout (8 bytes each, 128-byte map):

| Off | Writer | Meaning |
|-----|--------|---------|
| `0x00` | encoder | `(frame << 32) \| op_idx` |
| `0x08` | encoder | `Phase` tag (see `Phase` enum in `crumb.rs`) |
| `0x10` | API | `(ApiMethod << 56) \| (level << 48) \| (flags << 16)` for the last Lock/Unlock |
| `0x18` | API | pointer returned to the game from that Lock |
| `0x20` | any | `(thunk_code << 32) \| (status << 8) \| marker` — `0xEE` mid-call, `0xDD` returned |
| `0x28` | any | `unix_call` `params` pointer at entry |
| `0x30` | API | `(seq << 32) \| (tcc_max << 16) \| tcc_last` — `FfVsKey::tex_coord_count` at draw-snapshot capture |
| `0x38` | encoder | same shape — same field at `emit_vs_ff` dispatch |
| `0x40` | API | address of the captured `&FfVsKey` (PE-side closure storage) |
| `0x48` | encoder | address of the dispatched `&FfVsKey` |
| `0x50` | API | wrapping byte-sum + rotate fingerprint of all FfVsKey bytes at capture |
| `0x58` | encoder | same fingerprint at dispatch — mismatch ⇒ at least one byte changed in transit |

Adding a new probe: define under both `enabled` and `disabled` modules with matching signatures, document the new slot in this table, call directly from the suspect site (no `#[cfg]` at the call site). Single `write_volatile` so the disabled-build optimizer fully elides them. No formatting or syscalls — preserve the zero-cost-when-off contract.
