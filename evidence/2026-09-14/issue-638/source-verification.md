# Source verification

The `query.flushImmediate` arm in `windows/d3d9/src/query.rs` returns `D3D_OK` with the permissive count before `flush_current_frame_blocking` or visibility intake. The config-off path performs those operations before reporting the real result.

Direct dynamic vertex and index buffer locks treat `D3DLOCK_NOOVERWRITE` as permission to keep the existing backing. A CPU write can therefore target pages a queued draw still references. Metal's GPU resource hazard tracking cannot order that CPU write behind the queued draw.

The built-in `wow` profile in `windows/core/src/app_profile.rs` currently sets `query.flushImmediate=true`. Issue #638's statement that no shipped configuration enables the key is stale. Existing source records a measured loading-screen benefit and that the returned count is not read, but it does not establish that the poll never gates CPU-writable storage reuse.
