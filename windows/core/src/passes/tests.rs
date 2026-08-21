//! Unit tests for the render-pass state machine and its load/store optimizer.
//!
//! `PassState` is driven through synthetic frames of opaque handles, so pass breaking and
//! every load/store rule run without a GPU: clears folded into load actions versus painted
//! as quads, the `DontCare` rules with their sampler, blit and mid-frame-flush guards,
//! render scale, multiple render targets, and the `last_bound` dedup cache. A rule that
//! fires one case too wide loses pixels, so each guard gets a case that fails without it.

use mtld3d_shared::CommandType;

use super::*;

fn tex(raw: u64) -> MetalHandle<MTLTextureKind> {
    // SAFETY: tests; opaque values never dereferenced.
    unsafe { MetalHandle::new(raw) }
}

fn pso(raw: u64) -> MetalHandle<MTLRenderPipelineStateKind> {
    // SAFETY: tests; opaque values never dereferenced.
    unsafe { MetalHandle::new(raw) }
}

const BB_SIZE: (u32, u32) = (640, 480);
const BB_FORMAT: PixelFormat = PixelFormat::Bgra8Unorm;
const RT_FORMAT: PixelFormat = PixelFormat::Bgra8Unorm;

fn backbuffer() -> MetalHandle<MTLTextureKind> {
    tex(0x1000)
}
fn depth() -> MetalHandle<MTLTextureKind> {
    tex(0x2000)
}

fn fresh() -> PassState {
    let mut s = PassState::new();
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        depth_texture: depth(),
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    s
}

/// A frame rasterizing the back buffer at half the reported resolution.
fn fresh_scaled() -> PassState {
    let mut s = PassState::new();
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        depth_texture: depth(),
        depth_has_stencil: false,
        render_scale: RenderScale::from_percent(50),
        continues_frame: false,
    });
    s
}

#[test]
fn scaled_frame_binds_the_backbuffer_at_render_resolution() {
    let s = fresh_scaled();
    // D3D9 still reports 640x480; the texture is half that.
    assert_eq!(s.current_color_size, (320, 240));
    assert_eq!(s.effective_viewport(), (0, 0, 320, 240));
}

#[test]
fn scaled_viewport_and_scissor_convert_on_the_backbuffer() {
    let mut s = fresh_scaled();
    s.set_viewport(100, 50, 400, 300, 0.0, 1.0);
    assert_eq!(s.effective_viewport(), (50, 25, 200, 150));
    assert_eq!(
        s.resolved_scissor_rect(true, [100, 50, 400, 300]),
        (50, 25, 200, 150)
    );
}

#[test]
fn a_game_render_target_is_never_scaled() {
    // The game sized this texture itself, so its coordinates are already
    // in its own space and must survive untouched even though the frame
    // carries a non-default scale.
    let mut s = fresh_scaled();
    s.set_color_render_target(tex(0x3000), 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    assert_eq!(s.current_color_size, (256, 256));
    assert!(s.target_scale().is_identity());
    s.set_viewport(0, 0, 256, 256, 0.0, 1.0);
    assert_eq!(s.effective_viewport(), (0, 0, 256, 256));
    assert_eq!(
        s.resolved_scissor_rect(true, [16, 16, 64, 64]),
        (16, 16, 64, 64)
    );
}

#[test]
fn rebinding_the_backbuffer_restores_the_scale() {
    // The regression this whole design turns on: D3D9 forbids a null RT0,
    // so a game restoring the back buffer does it by binding the surface.
    // Keying the scale on handle identity keeps it applied; inferring it
    // from a null render-target pointer silently would not.
    let mut s = fresh_scaled();
    s.set_color_render_target(tex(0x3000), 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    assert!(s.target_scale().is_identity());

    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    assert!(!s.target_scale().is_identity());
    assert_eq!(s.current_color_size, (320, 240));
    s.set_viewport(0, 0, 640, 480, 0.0, 1.0);
    assert_eq!(s.effective_viewport(), (0, 0, 320, 240));
}

#[test]
fn identity_scale_leaves_every_coordinate_alone() {
    // The safety argument for shipping the default: at 100% nothing here
    // can perturb a pixel.
    let mut s = fresh();
    s.set_viewport(37, 11, 501, 293, 0.0, 1.0);
    assert_eq!(s.effective_viewport(), (37, 11, 501, 293));
    assert_eq!(
        s.resolved_scissor_rect(true, [7, 9, 123, 456]),
        (7, 9, 123, 456)
    );
    assert!(s.target_scale().is_identity());
}

fn dummy_draw() -> Command {
    // Any non-viewport command serves as a "draw" marker for bookkeeping
    // tests — the state machine only counts commands, not their kind.
    Command::draw_primitives(mtld3d_shared::mtl::PrimitiveType::Triangle, 0, 3)
}

fn unpack_scissor(cmd: &Command) -> (u32, u32, u32, u32) {
    assert_eq!(cmd.cmd, CommandType::SetScissorRect as u32);
    let x = cmd.param_a;
    // param_b/c are wire payload encoded in u64 — extract low/high u32 halves.
    let y = u32::try_from(cmd.param_b & 0xFFFF_FFFF).expect("low 32 bits fit u32");
    let w = u32::try_from(cmd.param_c >> 32).expect("high 32 bits fit u32");
    let h = u32::try_from(cmd.param_c & 0xFFFF_FFFF).expect("low 32 bits fit u32");
    (x, y, w, h)
}

#[test]
fn frame_sampled_textures_tracks_fragment_binds_in_stream_order() {
    let mut s = fresh();
    let atlas = tex(0x7E10);
    // Not sampled before any draw emitted a bind — an upload landing
    // here must NOT rename (no earlier draw reads the old content).
    assert!(!s.texture_sampled_this_frame(atlas));
    s.emit_command(Command::set_fragment_texture(atlas.raw(), 0));
    assert!(s.texture_sampled_this_frame(atlas));
    // Unrelated handle stays unsampled (a renamed-fresh texture
    // relies on exactly this).
    assert!(!s.texture_sampled_this_frame(tex(0x7E20)));
}

#[test]
fn frame_sampled_textures_clears_on_reset_frame() {
    let mut s = fresh();
    let atlas = tex(0x7E10);
    s.emit_command(Command::set_fragment_texture(atlas.raw(), 0));
    assert!(s.texture_sampled_this_frame(atlas));
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        depth_texture: depth(),
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    // Per-frame set resets — a next-frame upload before the first
    // sample goes to the live texture again.
    assert!(!s.texture_sampled_this_frame(atlas));
}

#[test]
fn frame_sampled_textures_ignores_null_bind() {
    let mut s = fresh();
    s.emit_command(Command::set_fragment_texture(0, 0));
    assert!(!s.texture_sampled_this_frame(tex(0)));
}

#[test]
fn inline_slot0_bind_forces_next_bound_vertex_buffer_reemit() {
    let mut cache = LastBoundCache::new();
    // First bind of a real VB handle reports a change and caches it.
    assert!(cache.vertex_buffer_changed(0, 0xDEAD, 0));
    // A redundant rebind of the same (handle, offset) would be skipped.
    assert!(!cache.vertex_buffer_changed(0, 0xDEAD, 0));
    // An inline slot-0 bind (setVertexBytes) clobbers the Metal binding;
    // invalidating the cache must force the next bound draw to re-emit
    // even though it targets the same (handle, offset).
    cache.invalidate_vertex_buffer();
    assert!(cache.vertex_buffer_changed(0, 0xDEAD, 0));
}

#[test]
fn vertex_buffer_slots_are_tracked_independently() {
    let mut cache = LastBoundCache::new();
    assert!(cache.vertex_buffer_changed(0, 0xDEAD, 0));
    assert!(cache.vertex_buffer_changed(1, 0xBEEF, 16));
    // Slot 1's bind leaves slot 0's cache intact, and vice versa.
    assert!(!cache.vertex_buffer_changed(0, 0xDEAD, 0));
    assert!(!cache.vertex_buffer_changed(1, 0xBEEF, 16));
    // Invalidating slot 0 (inline UP bytes) does not touch slot 1.
    cache.invalidate_vertex_buffer();
    assert!(cache.vertex_buffer_changed(0, 0xDEAD, 0));
    assert!(!cache.vertex_buffer_changed(1, 0xBEEF, 16));
    // A null-stream inline bind at slot 1 forgets only slot 1.
    cache.invalidate_vertex_buffer_slot(1);
    assert!(cache.vertex_buffer_changed(1, 0xBEEF, 16));
    assert!(!cache.vertex_buffer_changed(0, 0xDEAD, 0));
}

#[test]
fn begin_frame_starts_no_pass() {
    let s = fresh();
    assert!(s.passes().is_empty());
    assert!(s.current_pass_closed());
}

#[test]
fn first_command_opens_pass() {
    let mut s = fresh();
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 1);
    let pass = &s.passes()[0];
    assert_eq!(pass.color_texture(), backbuffer());
    assert_eq!(pass.depth_texture(), depth());
    assert_eq!(pass.viewport(), (0, 0, BB_SIZE.0, BB_SIZE.1));
    // Rule A: first use of the backbuffer + depth this frame, no
    // pending clear ⇒ DontCare. Prior contents are undefined per
    // D3D9 spec.
    assert_eq!(pass.color_load(), ColorLoad::DontCare);
    assert_eq!(pass.depth_load(), DepthLoad::DontCare);
    // First command is the implicit viewport, second is our draw.
    assert_eq!(pass.commands().len(), 2);
}

#[test]
fn set_render_target_ends_pass_on_diff() {
    let rt = tex(0x3000);
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[0].color_texture(), backbuffer());
    assert_eq!(s.passes()[1].color_texture(), rt);
    // With no explicit viewport set, the new pass falls back to the new
    // attachment size — matches D3D9 semantics where SetRenderTarget
    // implicitly resizes the viewport to the new target.
    assert_eq!(s.passes()[1].viewport(), (0, 0, 256, 256));
}

#[test]
fn set_render_target_same_handle_no_break() {
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 1);
}

#[test]
fn set_render_target_subresource_breaks_on_slice_or_level_change() {
    let rt = tex(0x3000);
    let mut s = fresh();
    s.set_color_render_target_subresource(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY, (0, 0));
    s.emit_command(dummy_draw());
    s.set_color_render_target_subresource(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY, (1, 0));
    s.emit_command(dummy_draw());
    s.set_color_render_target_subresource(rt, 128, 128, RT_FORMAT, RenderScale::IDENTITY, (1, 1));
    s.emit_command(dummy_draw());

    assert_eq!(s.passes().len(), 3);
    assert_eq!(
        (s.passes()[0].color_slice(), s.passes()[0].color_level()),
        (0, 0)
    );
    assert_eq!(
        (s.passes()[1].color_slice(), s.passes()[1].color_level()),
        (1, 0)
    );
    assert_eq!(
        (s.passes()[2].color_slice(), s.passes()[2].color_level()),
        (1, 1)
    );
}

#[test]
fn ordinary_render_target_binding_uses_base_subresource() {
    let rt = tex(0x3000);
    let mut s = fresh();
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());

    assert_eq!(s.passes().len(), 1);
    assert_eq!(s.passes()[0].color_slice(), 0);
    assert_eq!(s.passes()[0].color_level(), 0);
}

#[test]
fn mid_pass_color_clear_returns_emit_quad_outcome() {
    // D3D9's `Clear` is viewport-clipped and can fire mid-render.
    // Metal has no in-encoder Clear primitive, so a mid-pass Clear
    // returns `ColorClearOutcome::EmitQuad` so the encoder layer
    // (which owns the clear-quad pipeline cache) can emit a
    // scissored fullscreen-triangle draw. The pass does NOT break:
    // breaking on Clear would open a new encoder with
    // `loadAction = Clear` which wipes the full attachment under
    // Metal's full-attachment Clear semantics, deleting all prior
    // tile draws (the failure mode for sub-rect Clears into a
    // shared shadow/tile atlas).
    let mut s = fresh();
    s.emit_command(dummy_draw());
    let outcome = s.clear_color(1, 2, 3, 4);
    s.emit_command(dummy_draw());
    assert!(matches!(outcome, ColorClearOutcome::EmitQuad { .. }));
    assert_eq!(
        s.passes().len(),
        1,
        "pass should not break on mid-pass Clear; encoder emits a clear-quad inline"
    );
}

#[test]
fn clear_before_any_draw_merges_into_first_pass() {
    let mut s = fresh();
    s.clear_color(5, 6, 7, 8);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 1);
    assert_eq!(
        s.passes()[0].color_load(),
        ColorLoad::Clear {
            r: 5,
            g: 6,
            b: 7,
            a: 8
        }
    );
}

#[test]
fn clear_amends_empty_pass_in_place() {
    // Pass open with only the viewport command → Clear amends the load
    // action directly instead of ending the pass.
    let mut s = fresh();
    s.ensure_pass_open();
    assert_eq!(s.passes().len(), 1);
    s.clear_color(9, 9, 9, 9);
    assert_eq!(s.passes().len(), 1, "empty pass should not be broken");
    assert_eq!(
        s.passes()[0].color_load(),
        ColorLoad::Clear {
            r: 9,
            g: 9,
            b: 9,
            a: 9
        }
    );
}

#[test]
fn depth_change_triggers_pass_break() {
    let other_depth = tex(0x4000);
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.set_depth_stencil_attachment(other_depth, false, false);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[0].depth_texture(), depth());
    assert_eq!(s.passes()[1].depth_texture(), other_depth);
}

#[test]
fn viewport_applied_to_new_pass_start() {
    let rt = tex(0x3000);
    let mut s = fresh();
    s.set_viewport(0, 0, 320, 240, 0.0, 1.0);
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 128, 128, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 2);
    // Both passes use the 320x240 viewport (sticky). The first command
    // of each pass is the viewport set.
    assert_eq!(s.passes()[0].viewport(), (0, 0, 320, 240));
    assert_eq!(s.passes()[1].viewport(), (0, 0, 320, 240));
}

#[test]
fn first_use_each_rt_is_dontcare() {
    let rt = tex(0x3000);
    // Rule A — every rt's first use in a frame, with no pending
    // clear, gets DontCare. Backbuffer is first-use in pass A;
    // rt is first-use in pass B. Depth is shared so it's
    // first-use in A and re-use (Load) in B.
    let mut s = fresh();
    s.emit_command(dummy_draw()); // pass A on backbuffer() + depth()
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw()); // pass B on rt + depth()
    assert_eq!(s.passes()[0].color_load(), ColorLoad::DontCare);
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::DontCare);
    assert_eq!(s.passes()[1].color_load(), ColorLoad::DontCare);
    // depth() is seen-already (pass A used it), so Load this time.
    assert_eq!(s.passes()[1].depth_load(), DepthLoad::Load);
}

#[test]
fn region_clear_as_first_touch_loads_instead_of_dontcare() {
    // A `Clear(pRects)` opening the frame's first backbuffer pass:
    // the rect quads cover only the rects, so the pass must open with
    // `Load`, not Rule A's first-use `DontCare` — `DontCare` would
    // present undefined tile memory outside the rects.
    let mut s = fresh();
    s.begin_region_color_clear();
    assert_eq!(s.passes()[0].color_load(), ColorLoad::Load);
}

#[test]
fn region_clear_after_pending_full_clear_keeps_the_clear() {
    // `Clear(NULL, white)` then `Clear(rects, red)`: the pending
    // whole-RT clear must land under the rect quads, per the D3D9
    // spec (white everywhere outside the rects).
    let mut s = fresh();
    s.clear_color(10, 20, 30, 40);
    s.begin_region_color_clear();
    assert!(matches!(
        s.passes()[0].color_load(),
        ColorLoad::Clear { .. }
    ));
}

#[test]
fn region_depth_clear_as_first_touch_loads_instead_of_dontcare() {
    // The depth mirror of `region_clear_as_first_touch_loads_instead_of_
    // dontcare`: the rect quads cover only the rects, so the pass opens
    // with `Load` on both planes.
    let mut s = fresh();
    let target = s.begin_region_depth_stencil_clear();
    assert!(target.is_some());
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::Load);
    assert_eq!(s.passes()[0].stencil_load(), StencilLoad::Load);
}

#[test]
fn region_depth_clear_after_pending_full_clear_keeps_the_clear() {
    // `Clear(NULL, 1.0)` then `Clear(rects, 0.0)`: the pending whole-
    // attachment depth clear lands under the rect quads.
    let mut s = fresh();
    let z = f32::to_bits(1.0);
    s.clear_depth(z);
    s.begin_region_depth_stencil_clear();
    assert!(matches!(
        s.passes()[0].depth_load(),
        DepthLoad::Clear { value } if value == z
    ));
}

#[test]
fn region_depth_clear_without_depth_attachment_is_noop() {
    let mut s = fresh();
    s.set_depth_stencil_attachment(MetalHandle::NULL, false, false);
    assert!(s.begin_region_depth_stencil_clear().is_none());
    assert!(s.passes().is_empty());
}

#[test]
fn mid_pass_depth_clear_returns_emit_quad_outcome() {
    // Depth mirror of `mid_pass_color_clear_returns_emit_quad_outcome`.
    let mut s = fresh();
    s.emit_command(dummy_draw());
    let z = f32::to_bits(0.5);
    let outcome = s.clear_depth(z);
    s.emit_command(dummy_draw());
    assert!(matches!(outcome, DepthClearOutcome::EmitQuad { value, .. } if value == z));
    assert_eq!(s.passes().len(), 1);
}

#[test]
fn reset_frame_drops_pending_clears() {
    let mut s = fresh();
    s.clear_color(1, 2, 3, 4);
    assert!(s.pending_color_clear().is_some());
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        depth_texture: depth(),
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    assert!(s.pending_color_clear().is_none());
    assert!(s.passes().is_empty());
}

#[test]
fn clear_then_rt_switch_materializes_old_target() {
    let rt = tex(0x3000);
    // D3D9 semantic: Clear applies to the bound rt at call time. If the
    // game clears the rt and switches target without drawing, the old
    // rt must still receive the clear.
    let mut s = fresh();
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.clear_color(1, 2, 3, 4);
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 2);
    // Old rt got the clear
    assert_eq!(s.passes()[0].color_texture(), rt);
    assert_eq!(
        s.passes()[0].color_load(),
        ColorLoad::Clear {
            r: 1,
            g: 2,
            b: 3,
            a: 4
        }
    );
    // Backbuffer pass is first-use this frame (the synthesised
    // clear pass ran on rt, not on backbuffer()), no pending clear ⇒
    // Rule A flips to DontCare.
    assert_eq!(s.passes()[1].color_texture(), backbuffer());
    assert_eq!(s.passes()[1].color_load(), ColorLoad::DontCare);
}

#[test]
fn flush_pending_clears_is_noop_when_empty() {
    let mut s = fresh();
    s.flush_pending_clears();
    assert!(s.passes().is_empty());
}

#[test]
fn flush_pending_clears_materializes_pass() {
    let mut s = fresh();
    s.clear_color(7, 8, 9, 10);
    s.flush_pending_clears();
    assert_eq!(s.passes().len(), 1);
    assert_eq!(
        s.passes()[0].color_load(),
        ColorLoad::Clear {
            r: 7,
            g: 8,
            b: 9,
            a: 10
        }
    );
    // Pass is closed so a subsequent draw opens a new pass.
    assert!(s.current_pass_closed());
}

#[test]
fn multiple_rt_swaps_produce_multiple_passes() {
    let rt = tex(0x3000);
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 3);
    assert_eq!(s.passes()[0].color_texture(), backbuffer());
    assert_eq!(s.passes()[1].color_texture(), rt);
    assert_eq!(s.passes()[2].color_texture(), backbuffer());
}

#[test]
fn color_format_propagates_per_pass() {
    const OTHER_FORMAT: PixelFormat = PixelFormat::Rgba16Float;
    let rt = tex(0x3000);
    // Format the pass opens with is what was current at pass-open
    // time. Pipelines created during each pass key on this value,
    // so distinct rt formats must yield distinct Pass.color_format.
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 256, 256, OTHER_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes()[0].color_format(), BB_FORMAT);
    assert_eq!(s.passes()[1].color_format(), OTHER_FORMAT);
    assert_eq!(s.current_color_format(), OTHER_FORMAT);
}

#[test]
fn emit_scissor_enabled_uses_game_rect() {
    let mut s = fresh();
    s.set_viewport(0, 0, 640, 480, 0.0, 1.0);
    s.emit_scissor(true, [10, 20, 200, 150]);
    let cmds = s.passes()[0].commands();
    // [0] = implicit viewport, [1] = our scissor
    assert_eq!(unpack_scissor(&cmds[1]), (10, 20, 200, 150));
}

#[test]
fn emit_scissor_disabled_falls_back_to_viewport() {
    let mut s = fresh();
    s.set_viewport(5, 7, 320, 240, 0.0, 1.0);
    // test_enable = false → stored rect ignored, viewport used
    s.emit_scissor(false, [10, 20, 200, 150]);
    let cmds = s.passes()[0].commands();
    assert_eq!(unpack_scissor(&cmds[1]), (5, 7, 320, 240));
}

#[test]
fn emit_scissor_zero_rect_falls_back_to_viewport() {
    let mut s = fresh();
    s.set_viewport(0, 0, 640, 480, 0.0, 1.0);
    // SetScissorRect was never called (scissor_rect = [0; 4]) but the
    // game turned the test on anyway → fall back to viewport so Metal
    // doesn't clip to an empty rect.
    s.emit_scissor(true, [0, 0, 0, 0]);
    let cmds = s.passes()[0].commands();
    assert_eq!(unpack_scissor(&cmds[1]), (0, 0, 640, 480));
}

#[test]
fn emit_scissor_reemit_updates_per_draw() {
    // Our architecture re-emits scissor every draw (no dirty
    // tracking). Two draws with different states produce two commands.
    let mut s = fresh();
    s.set_viewport(0, 0, 640, 480, 0.0, 1.0);
    s.emit_scissor(true, [10, 20, 200, 150]);
    s.emit_scissor(false, [0, 0, 0, 0]);
    let cmds = s.passes()[0].commands();
    // [0] viewport, [1] first scissor, [2] second scissor
    assert_eq!(unpack_scissor(&cmds[1]), (10, 20, 200, 150));
    assert_eq!(unpack_scissor(&cmds[2]), (0, 0, 640, 480));
}

#[test]
fn emit_scissor_without_viewport_uses_rt_size() {
    // No SetViewport call → PassState falls back to the color-size
    // fallback at pass-open (also used as viewport fallback).
    let mut s = fresh();
    s.emit_scissor(false, [10, 20, 30, 40]);
    let cmds = s.passes()[0].commands();
    assert_eq!(unpack_scissor(&cmds[1]), (0, 0, BB_SIZE.0, BB_SIZE.1));
}

#[test]
fn set_viewport_dedups_redundant_reemit_within_pass() {
    let mut s = fresh();
    // Opens the pass; the pass-open viewport (RT-size fallback,
    // depth range 0..1) is the first command.
    s.emit_command(dummy_draw());
    let n0 = s.passes()[0].commands().len();
    // Re-setting the value the pass already opened with is a no-op —
    // re-emitting it is the Xcode "already bound" redundant bind.
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    assert_eq!(
        s.passes()[0].commands().len(),
        n0,
        "redundant viewport must not re-emit",
    );
    // A genuine x/y/w/h change re-emits once.
    s.set_viewport(0, 0, 320, 240, 0.0, 1.0);
    assert_eq!(s.passes()[0].commands().len(), n0 + 1);
    // Re-setting that same value is again a no-op.
    s.set_viewport(0, 0, 320, 240, 0.0, 1.0);
    assert_eq!(s.passes()[0].commands().len(), n0 + 1);
    // A depth-range-only change (same x/y/w/h) must still re-emit —
    // the z-range is part of the bind.
    s.set_viewport(0, 0, 320, 240, 0.0, 0.5);
    assert_eq!(
        s.passes()[0].commands().len(),
        n0 + 2,
        "depth-range-only change must re-emit",
    );
}

fn dummy_blit() -> BlitCommand {
    BlitCommand::copy_texture_to_texture_full_mip(0xAA, 0xBB, 0, 64, 64)
}

#[test]
fn pending_blit_drains_into_next_pass() {
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.push_pending_leading_blit(dummy_blit());
    // Next pass open inherits the queued blit.
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[0].leading_blits().len(), 0);
    assert_eq!(s.passes()[1].leading_blits().len(), 1);
    // Pending queue is empty after the drain.
    let mut s2 = s;
    assert!(s2.take_pending_leading_blits().is_empty());
}

#[test]
fn trailing_pending_blit_survives_via_take() {
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.push_pending_leading_blit(dummy_blit());
    // No follow-up draw — pending blit stays in the queue for
    // `submit` to drain into a synthetic blit-only pass.
    let trailing = s.take_pending_leading_blits();
    assert_eq!(trailing.len(), 1);
}

#[test]
fn fresh_pass_has_no_counting_visibility() {
    let mut s = fresh();
    s.emit_command(dummy_draw());
    assert!(!s.passes()[0].has_counting_visibility());
}

#[test]
fn counting_visibility_latches_flag() {
    let mut s = fresh();
    s.emit_command(Command::set_visibility_result_mode(
        VisibilityResultMode::Counting,
        0,
    ));
    assert!(s.passes()[0].has_counting_visibility());
}

#[test]
fn disabled_only_does_not_flip_flag() {
    // End-of-query tail: the encoder emits `Disabled` with
    // `active_count == 0`. No counter is written in this pass, so
    // the buffer must not be attached.
    let mut s = fresh();
    s.emit_command(Command::set_visibility_result_mode(
        VisibilityResultMode::Disabled,
        0,
    ));
    assert!(!s.passes()[0].has_counting_visibility());
}

#[test]
fn non_visibility_commands_do_not_flip_flag() {
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.emit_command(Command::set_cull_mode(mtld3d_shared::mtl::CullMode::None));
    assert!(!s.passes()[0].has_counting_visibility());
}

#[test]
fn counting_then_disabled_stays_latched() {
    // BEGIN then END within one pass: Counting arms, Disabled
    // closes — the counter was written, so the flag must stay set
    // for the submit path to keep the buffer attached.
    let mut s = fresh();
    s.emit_command(Command::set_visibility_result_mode(
        VisibilityResultMode::Counting,
        0,
    ));
    s.emit_command(Command::set_visibility_result_mode(
        VisibilityResultMode::Disabled,
        8,
    ));
    assert!(s.passes()[0].has_counting_visibility());
}

#[test]
fn pass_break_clears_flag_for_new_pass() {
    let rt = tex(0x3000);
    // A Counting pass followed by a rendertarget switch must not
    // bleed the flag into the next pass — each pass tracks its
    // own attachments independently.
    let mut s = fresh();
    s.emit_command(Command::set_visibility_result_mode(
        VisibilityResultMode::Counting,
        0,
    ));
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 2);
    assert!(s.passes()[0].has_counting_visibility());
    assert!(!s.passes()[1].has_counting_visibility());
}

// ── LastBoundCache ──

#[test]
fn last_bound_first_call_reports_changed() {
    let mut c = LastBoundCache::new();
    assert!(c.fragment_sampler_changed(0, 0xAAAA));
    assert!(c.fragment_texture_changed(0, 0xBBBB));
    assert!(c.pipeline_changed(0xCCCC));
    assert!(c.depth_stencil_changed(0xDDDD));
    assert!(c.cull_mode_changed(CullMode::Back));
}

#[test]
fn last_bound_repeat_value_is_unchanged() {
    let mut c = LastBoundCache::new();
    c.fragment_sampler_changed(2, 0xAAAA);
    assert!(!c.fragment_sampler_changed(2, 0xAAAA));
    c.fragment_texture_changed(2, 0xBBBB);
    assert!(!c.fragment_texture_changed(2, 0xBBBB));
    c.pipeline_changed(0xCCCC);
    assert!(!c.pipeline_changed(0xCCCC));
    c.depth_stencil_changed(0xDDDD);
    assert!(!c.depth_stencil_changed(0xDDDD));
    c.cull_mode_changed(CullMode::Front);
    assert!(!c.cull_mode_changed(CullMode::Front));
}

#[test]
fn last_bound_different_value_is_changed() {
    let mut c = LastBoundCache::new();
    c.fragment_sampler_changed(0, 0xAAAA);
    assert!(c.fragment_sampler_changed(0, 0xBEEF));
    c.cull_mode_changed(CullMode::None);
    assert!(c.cull_mode_changed(CullMode::Back));
}

#[test]
fn last_bound_cull_none_distinct_from_unset() {
    // CullMode::None is value 0, but a freshly-reset cache must still
    // report "changed" on the first call — otherwise the first draw of
    // a pass that wants None cull would silently inherit whatever
    // Metal's default is. The Option<CullMode> sentinel guards this.
    let mut c = LastBoundCache::new();
    assert!(c.cull_mode_changed(CullMode::None));
    assert!(!c.cull_mode_changed(CullMode::None));
}

#[test]
fn last_bound_stages_are_independent() {
    let mut c = LastBoundCache::new();
    c.fragment_sampler_changed(3, 0xAAAA);
    c.fragment_texture_changed(3, 0xBBBB);
    assert!(c.fragment_sampler_changed(4, 0xAAAA));
    assert!(c.fragment_texture_changed(4, 0xBBBB));
    assert!(!c.fragment_sampler_changed(3, 0xAAAA));
    assert!(!c.fragment_texture_changed(3, 0xBBBB));
}

#[test]
fn last_bound_reset_clears_everything() {
    let mut c = LastBoundCache::new();
    c.fragment_sampler_changed(0, 0xAAAA);
    c.fragment_texture_changed(1, 0xBBBB);
    c.pipeline_changed(0xCCCC);
    c.depth_stencil_changed(0xDDDD);
    c.cull_mode_changed(CullMode::Back);
    c.vs_constants_changed(&[1, 2, 3, 4]);
    c.ps_constants_changed(&[5, 6, 7, 8]);
    c.ps_alpha_ref_changed(&[9, 10, 11, 12]);
    c.ps_fog_color_changed(&[13, 14, 15, 16]);
    c.vertex_buffer_changed(0, 0xEEEE, 32);
    c.scissor_rect_changed((1, 2, 3, 4));
    c.blend_color_changed(0xFF11_2233);
    c.reset();
    assert!(c.fragment_sampler_changed(0, 0xAAAA));
    assert!(c.fragment_texture_changed(1, 0xBBBB));
    assert!(c.pipeline_changed(0xCCCC));
    assert!(c.depth_stencil_changed(0xDDDD));
    assert!(c.cull_mode_changed(CullMode::Back));
    assert!(c.vs_constants_changed(&[1, 2, 3, 4]));
    assert!(c.ps_constants_changed(&[5, 6, 7, 8]));
    assert!(c.ps_alpha_ref_changed(&[9, 10, 11, 12]));
    assert!(c.ps_fog_color_changed(&[13, 14, 15, 16]));
    assert!(c.vertex_buffer_changed(0, 0xEEEE, 32));
    assert!(c.scissor_rect_changed((1, 2, 3, 4)));
    assert!(c.blend_color_changed(0xFF11_2233));
}

#[test]
fn last_bound_inline_bytes_dedup() {
    let mut c = LastBoundCache::new();
    assert!(c.ps_constants_changed(&[1, 2, 3, 4]));
    assert!(!c.ps_constants_changed(&[1, 2, 3, 4]));
    assert!(c.ps_constants_changed(&[1, 2, 3, 5]));
    assert!(c.ps_constants_changed(&[1, 2, 3])); // length change
    assert!(!c.ps_constants_changed(&[1, 2, 3]));
}

#[test]
fn last_bound_inline_bytes_slots_are_independent() {
    let mut c = LastBoundCache::new();
    c.vs_constants_changed(&[1; 16]);
    c.ps_constants_changed(&[2; 16]);
    c.ps_alpha_ref_changed(&[3; 4]);
    c.ps_fog_color_changed(&[4; 16]);
    // Identical content in a different slot must still report changed
    // (slot 13 hasn't seen this payload yet).
    assert!(!c.vs_constants_changed(&[1; 16]));
    assert!(!c.ps_constants_changed(&[2; 16]));
    assert!(!c.ps_alpha_ref_changed(&[3; 4]));
    assert!(!c.ps_fog_color_changed(&[4; 16]));
}

#[test]
fn last_bound_inline_bytes_reset_keeps_capacity() {
    let mut c = LastBoundCache::new();
    c.ps_constants_changed(&[0xAB; 256]);
    let cap_before = c.ps_constants.capacity();
    c.reset();
    assert_eq!(c.ps_constants.len(), 0);
    assert_eq!(c.ps_constants.capacity(), cap_before);
}

#[test]
fn last_bound_vertex_buffer_dedup() {
    let mut c = LastBoundCache::new();
    assert!(c.vertex_buffer_changed(0, 0xAAAA, 0));
    assert!(!c.vertex_buffer_changed(0, 0xAAAA, 0));
    // Same handle, different offset → changed.
    assert!(c.vertex_buffer_changed(0, 0xAAAA, 64));
    // Same offset, different handle → changed.
    assert!(c.vertex_buffer_changed(0, 0xBBBB, 64));
}

#[test]
fn last_bound_scissor_dedup() {
    let mut c = LastBoundCache::new();
    // First emit always goes through — fresh encoder has no scissor.
    assert!(c.scissor_rect_changed((0, 0, 640, 480)));
    assert!(!c.scissor_rect_changed((0, 0, 640, 480)));
    // Any tuple field different → changed.
    assert!(c.scissor_rect_changed((10, 0, 640, 480)));
    assert!(c.scissor_rect_changed((10, 20, 640, 480)));
    assert!(c.scissor_rect_changed((10, 20, 700, 480)));
    assert!(c.scissor_rect_changed((10, 20, 700, 500)));
    // Reset → first emit goes through again even with the same rect.
    let rect = (10, 20, 700, 500);
    assert!(!c.scissor_rect_changed(rect));
    c.reset();
    assert!(c.scissor_rect_changed(rect));
}

#[test]
fn last_bound_blend_color_dedup() {
    let mut c = LastBoundCache::new();
    // Default `0xFFFF_FFFF` is the post-reset value; matches Metal's
    // own default, so a first-call with the default reports unchanged.
    assert!(!c.blend_color_changed(0xFFFF_FFFF));
    assert!(c.blend_color_changed(0xFF80_2040));
    assert!(!c.blend_color_changed(0xFF80_2040));
    assert!(c.blend_color_changed(0xFFFF_FFFF));
}

#[test]
fn clear_quad_pipeline_change_forces_caster_reemit() {
    // The CSM cascade-atlas shadow-flicker case: all four cascades
    // render in one pass, each preceded by a mid-pass depth clear-quad. If
    // the clear-quad routes its own pipeline/DSS through the cache (as it
    // must), a later caster with the SAME pipeline/DSS as a prior cascade
    // is forced to re-emit — it does not stale-skip and inherit the
    // clear-quad's always-compare depth state.
    let mut c = LastBoundCache::new();
    let (p_caster, d_caster) = (0xCA57, 0x0D55);
    let (p_clear, d_clear) = (0xC1EA, 0xC1DD);
    // Cascade 0 caster binds its pipeline + depth-stencil.
    assert!(c.pipeline_changed(p_caster));
    assert!(c.depth_stencil_changed(d_caster));
    // Mid-pass clear-quad advances the cache to its own state.
    assert!(c.pipeline_changed(p_clear));
    assert!(c.depth_stencil_changed(d_clear));
    // Cascade 1 caster: identical to cascade 0 → must re-emit, not skip.
    assert!(c.pipeline_changed(p_caster));
    assert!(c.depth_stencil_changed(d_caster));
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "pipeline cache desync")]
fn debug_guard_catches_pipeline_bypass() {
    // Reproduce the clear-quad bug class: a pipeline emitted DIRECTLY onto
    // the encoder without going through `pipeline_changed`. The shadow
    // records the emit; the cache stays stale; the in-sync check fires.
    let cache = LastBoundCache::new();
    let mut shadow = DebugBoundShadow::default();
    shadow.record(&Command::set_render_pipeline_state(0xDEAD_BEEF));
    cache.debug_assert_in_sync(&shadow);
}

#[cfg(debug_assertions)]
#[test]
fn debug_guard_in_sync_across_every_tracked_slot() {
    // Each tracked slot driven through the real gate→emit→shadow cycle must
    // leave cache and shadow agreeing — proving the decode/re-pack round
    // trips (scissor packing, depth-bias bits, vertex-buffer offset
    // widening, cull discriminant) and that the guard is free of false
    // positives on correct usage. A fresh cache reports every first bind as
    // changed, so each gate must return `true`.
    let mut cache = LastBoundCache::new();
    let mut shadow = DebugBoundShadow::default();

    assert!(cache.pipeline_changed(0x9001));
    shadow.record(&Command::set_render_pipeline_state(0x9001));
    assert!(cache.depth_stencil_changed(0x9002));
    shadow.record(&Command::set_depth_stencil_state(0x9002));
    assert!(cache.cull_mode_changed(CullMode::Back));
    shadow.record(&Command::set_cull_mode(CullMode::Back));
    assert!(cache.fragment_texture_changed(3, 0x7E10));
    shadow.record(&Command::set_fragment_texture(0x7E10, 3));
    assert!(cache.fragment_sampler_changed(3, 0x5A77));
    shadow.record(&Command::set_fragment_sampler_state(0x5A77, 3));
    assert!(cache.vertex_buffer_changed(0, 0xBEEF, 0x40));
    shadow.record(&Command::set_vertex_buffer(0xBEEF, 0x40, 0));
    assert!(cache.scissor_rect_changed((7, 9, 1024, 768)));
    shadow.record(&Command::set_scissor_rect(7, 9, 1024, 768));
    assert!(cache.depth_bias_changed(-1e-4, -1.5));
    shadow.record(&Command::set_depth_bias(-1e-4, -1.5));

    // Every gate fired and recorded its matching emit: cache == encoder.
    cache.debug_assert_in_sync(&shadow);
}

#[cfg(debug_assertions)]
#[test]
fn debug_guard_in_sync_after_inline_slot0_invalidate() {
    // An inline slot-0 bind (UP geometry / clear-quad) clobbers the real
    // vertex buffer. The cache invalidates; the shadow must mirror that or
    // the next in-sync check false-positives.
    let mut cache = LastBoundCache::new();
    let mut shadow = DebugBoundShadow::default();

    assert!(cache.vertex_buffer_changed(0, 0xBEEF, 0x10));
    shadow.record(&Command::set_vertex_buffer(0xBEEF, 0x10, 0));
    cache.debug_assert_in_sync(&shadow);

    // Inline slot-0 bind, then invalidate — the encoder's emit order.
    shadow.record(&Command::set_vertex_bytes_at(0xA000, 4, 0));
    cache.invalidate_vertex_buffer();
    cache.debug_assert_in_sync(&shadow);

    // The next bound draw re-binds the same buffer and stays in sync.
    assert!(cache.vertex_buffer_changed(0, 0xBEEF, 0x10));
    shadow.record(&Command::set_vertex_buffer(0xBEEF, 0x10, 0));
    cache.debug_assert_in_sync(&shadow);
}

#[cfg(debug_assertions)]
#[test]
fn debug_guard_tracks_every_vertex_stream_slot() {
    // A second stream bound at slot 1 is mirrored by the shadow, and a
    // null-stream inline bind there is forgotten by both sides; a
    // uniform-slot inline bind (above the stream slots) touches neither.
    let mut cache = LastBoundCache::new();
    let mut shadow = DebugBoundShadow::default();

    assert!(cache.vertex_buffer_changed(0, 0xBEEF, 0x10));
    shadow.record(&Command::set_vertex_buffer(0xBEEF, 0x10, 0));
    assert!(cache.vertex_buffer_changed(1, 0xCAFE, 0x20));
    shadow.record(&Command::set_vertex_buffer(0xCAFE, 0x20, 1));
    cache.debug_assert_in_sync(&shadow);

    shadow.record(&Command::set_vertex_bytes_at(0xA000, 16, 1));
    cache.invalidate_vertex_buffer_slot(1);
    cache.debug_assert_in_sync(&shadow);

    shadow.record(&Command::set_vertex_bytes_at(
        0xB000,
        16,
        mtld3d_shared::mtl::VS_POS_FIXUP_SLOT,
    ));
    cache.debug_assert_in_sync(&shadow);
}

// ── Rule A: first-use LoadAction::DontCare ────────────────────

#[test]
fn rule_a_fresh_frame_clear_color_only_keeps_clear() {
    // Pending color clear must beat the first-use DontCare branch;
    // depth still falls through to DontCare since no depth clear
    // was issued.
    let mut s = fresh();
    s.clear_color(1, 2, 3, 4);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 1);
    assert_eq!(
        s.passes()[0].color_load(),
        ColorLoad::Clear {
            r: 1,
            g: 2,
            b: 3,
            a: 4
        }
    );
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::DontCare);
}

#[test]
fn rule_a_same_rt_after_pass_break_is_load() {
    let rt = tex(0x3000);
    // backbuffer() → rt → backbuffer(): the third pass re-uses
    // backbuffer(), which was already seen in pass 0, so it gets
    // Load (Rule A would let DontCare slip otherwise).
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 3);
    assert_eq!(s.passes()[2].color_texture(), backbuffer());
    assert_eq!(s.passes()[2].color_load(), ColorLoad::Load);
}

#[test]
fn rule_a_reset_frame_re_arms_dontcare() {
    let mut s = fresh();
    s.emit_command(dummy_draw());
    assert_eq!(s.passes()[0].color_load(), ColorLoad::DontCare);
    // Next frame: same backbuffer is "first use again" because the
    // seen set was reset.
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        depth_texture: depth(),
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 1);
    assert_eq!(s.passes()[0].color_load(), ColorLoad::DontCare);
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::DontCare);
}

#[test]
fn rule_a_first_use_stencil_is_dontcare_and_later_use_loads() {
    // The stencil plane lives in the depth texture, so it takes the
    // first-use DontCare under the depth predicate. A second pass on the
    // same texture in the frame loads: games carry stencil across passes.
    let ds = tex(0x3300);
    let rt = tex(0x3000);
    let mut s = fresh();
    s.set_depth_stencil_attachment(ds, false, true);
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::DontCare);
    assert_eq!(s.passes()[0].stencil_load(), StencilLoad::DontCare);
    assert_eq!(s.passes()[1].depth_load(), DepthLoad::Load);
    assert_eq!(s.passes()[1].stencil_load(), StencilLoad::Load);
}

#[test]
fn rule_a_pending_stencil_clear_beats_dontcare() {
    let ds = tex(0x3300);
    let mut s = fresh();
    s.set_depth_stencil_attachment(ds, false, true);
    assert_eq!(s.clear_stencil(5), StencilClearOutcome::Folded);
    s.emit_command(dummy_draw());
    assert_eq!(
        s.passes()[0].stencil_load(),
        StencilLoad::Clear { value: 5 }
    );
    // Depth had no pending clear, so it still takes the discard.
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::DontCare);
}

#[test]
fn rule_a_reset_frame_re_arms_stencil_dontcare() {
    let rt = tex(0x3000);
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes()[1].stencil_load(), StencilLoad::Load);
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        depth_texture: depth(),
        depth_has_stencil: true,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 1);
    assert_eq!(s.passes()[0].stencil_load(), StencilLoad::DontCare);
}

#[test]
fn rule_a_reverts_stencil_dontcare_when_depth_sampled_later() {
    // A texture declared sampleable never gets the discard up front; this
    // is the other route, a depth-stencil texture the frame samples
    // without having declared it. The stencil plane is reverted with the
    // depth plane, since the sampler reads the texture both live in.
    let ds = tex(0x3300);
    let mut s = fresh();
    s.set_depth_stencil_attachment(ds, false, true);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes()[0].stencil_load(), StencilLoad::DontCare);
    s.set_depth_stencil_attachment(depth(), false, false);
    s.emit_command(Command::set_fragment_texture(ds.raw(), 0));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_load_actions();
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::Load);
    assert_eq!(s.passes()[0].stencil_load(), StencilLoad::Load);
}

#[test]
fn rule_a_leading_blit_to_rt_forces_load() {
    let rt_src = tex(0x3000);
    let rt_dst = tex(0x4000);
    // StretchRect lands between two passes and writes to the
    // pass's destination rt. The blit's output must survive into
    // the pass — first-use DontCare would discard it.
    let mut s = fresh();
    // First pass on backbuffer() so rt_dst is first-use when it opens.
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt_dst, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.push_pending_leading_blit(BlitCommand::copy_texture_to_texture_full_mip(
        rt_src.raw(),
        rt_dst.raw(),
        0,
        256,
        256,
    ));
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[1].color_texture(), rt_dst);
    assert_eq!(s.passes()[1].color_load(), ColorLoad::Load);
}

// ── Rule B: last-use depth/stencil StoreAction::DontCare ──────

#[test]
fn rule_b_single_pass_depth_store_is_dontcare() {
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 1);
    assert_eq!(s.passes()[0].depth_store(), StoreAction::DontCare);
    // Color store is left as Store — the HDR present pass or next
    // frame's reads still need it.
    assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
}

#[test]
fn rule_b_three_passes_same_depth_only_last_is_dontcare() {
    let rt_a = tex(0x3000);
    let rt_b = tex(0x4000);
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt_a, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt_b, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 3);
    // All three share depth(); only the last pass's depth_store is DontCare.
    assert_eq!(s.passes()[0].depth_store(), StoreAction::Store);
    assert_eq!(s.passes()[1].depth_store(), StoreAction::Store);
    assert_eq!(s.passes()[2].depth_store(), StoreAction::DontCare);
}

#[test]
fn rule_b_alternating_depth_each_gets_last_use_dontcare() {
    let d1 = depth();
    let d2 = tex(0x9000);
    // Two depth textures alternating: d1, d2, d1, d2. Last d1 is
    // pass 2; last d2 is pass 3. Both should be DontCare; the
    // earlier passes (0, 1) keep Store.
    let mut s = fresh();
    // Pass 0: backbuffer() + d1
    s.emit_command(dummy_draw());
    // Pass 1: backbuffer() + d2
    s.set_depth_stencil_attachment(d2, false, false);
    s.emit_command(dummy_draw());
    // Pass 2: backbuffer() + d1
    s.set_depth_stencil_attachment(d1, false, false);
    s.emit_command(dummy_draw());
    // Pass 3: backbuffer() + d2
    s.set_depth_stencil_attachment(d2, false, false);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 4);
    assert_eq!(s.passes()[0].depth_store(), StoreAction::Store);
    assert_eq!(s.passes()[1].depth_store(), StoreAction::Store);
    assert_eq!(s.passes()[2].depth_store(), StoreAction::DontCare);
    assert_eq!(s.passes()[3].depth_store(), StoreAction::DontCare);
}

// ── Rule C: next-pass-clear color StoreAction::DontCare ───────

#[test]
fn rule_c_single_pass_color_store_is_store() {
    // No "next pass" → final pass's color contents must survive
    // (backbuffer Present and persistent RTs read them).
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 1);
    assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
}

#[test]
fn rule_c_distinct_rts_no_next_use_keep_store() {
    let rt = tex(0x3000);
    // Pass 0 backbuffer(), pass 1 rt, pass 2 backbuffer() — neither rt
    // is followed by another pass with the SAME color rt (backbuffer()'s
    // re-use at pass 2 has color_load=Load, not Clear). Rule C does
    // not fire for any pass here. Rule D, however, flips pass 1's
    // rt (non-backbuffer, last use, not sampled) to DontCare.
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
    assert_eq!(s.passes()[1].color_store(), StoreAction::DontCare);
    assert_eq!(s.passes()[2].color_store(), StoreAction::Store);
}

#[test]
fn rule_c_next_pass_clears_same_rt_flips_store() {
    let rt = tex(0x3000);
    // Pass 0 backbuffer(), pass 1 rt with clear, pass 2 backbuffer() with
    // clear. backbuffer() pass 0's next use is pass 2 (clear) → Rule C
    // flips. rt pass 1 has no next use → Rule C keeps Store, then
    // Rule D (non-backbuffer last-use) flips to DontCare.
    // backbuffer() pass 2 is the last pass for backbuffer() → exempt
    // from Rule D, keeps Store (Present reads it).
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.clear_color(1, 2, 3, 4);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.clear_color(5, 6, 7, 8);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 3);
    assert_eq!(s.passes()[0].color_texture(), backbuffer());
    assert_eq!(s.passes()[0].color_store(), StoreAction::DontCare);
    assert_eq!(s.passes()[1].color_texture(), rt);
    assert_eq!(s.passes()[1].color_store(), StoreAction::DontCare);
    assert_eq!(s.passes()[2].color_texture(), backbuffer());
    assert_eq!(s.passes()[2].color_store(), StoreAction::Store);
}

#[test]
fn rule_c_next_pass_loads_same_rt_keeps_store() {
    let rt = tex(0x3000);
    // Pass 0 backbuffer(), pass 1 backbuffer() (no clear → Load). Pass 0's
    // contents must survive — pass 1 reads them via Load.
    let mut s = fresh();
    s.emit_command(dummy_draw());
    // Force a pass break with no pending clear: bounce rt then back.
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 3);
    assert_eq!(s.passes()[0].color_texture(), backbuffer());
    assert_eq!(s.passes()[2].color_texture(), backbuffer());
    // Pass 2's color_load is Load (no clear was queued), so pass 0
    // MUST keep its store.
    assert_eq!(s.passes()[2].color_load(), ColorLoad::Load);
    assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
}

#[test]
fn rule_c_csm_cluster_intra_frame_stores_drop() {
    let rt_a = tex(0x3000);
    let rt_b = tex(0x4000);
    // WoW CSM-style frame shape: scene-on-backbuffer → rt_a clear →
    // rt_b clear → rt_a clear → UI-on-backbuffer (loads). Each
    // intra-frame cascade pass's color store is redundant because
    // the next pass touching the same rt begins with Clear. The
    // first backbuffer() pass's store stays because the UI pass loads.
    let mut s = fresh();
    // Pass 0: scene on backbuffer()
    s.emit_command(dummy_draw());
    // Pass 1: rt_a cascade A1, cleared on entry
    s.set_color_render_target(rt_a, 1024, 512, RT_FORMAT, RenderScale::IDENTITY);
    s.clear_color(0, 0, 0, 0);
    s.emit_command(dummy_draw());
    // Pass 2: rt_b cascade B1, cleared on entry
    s.set_color_render_target(rt_b, 1024, 512, RT_FORMAT, RenderScale::IDENTITY);
    s.clear_color(0, 0, 0, 0);
    s.emit_command(dummy_draw());
    // Pass 3: rt_a cascade A2, cleared on entry
    s.set_color_render_target(rt_a, 1024, 512, RT_FORMAT, RenderScale::IDENTITY);
    s.clear_color(0, 0, 0, 0);
    s.emit_command(dummy_draw());
    // Pass 4: UI on backbuffer() (loads scene)
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 5);
    // Pass 0 backbuffer() → next backbuffer() use is pass 4, which Loads → keep Store.
    assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
    // Pass 1 rt_a → next rt_a use is pass 3 Clear → Rule C flips.
    assert_eq!(s.passes()[1].color_store(), StoreAction::DontCare);
    // Pass 2 rt_b → no next rt_b use → Rule D (non-backbuffer last-use)
    // flips to DontCare since rt_b is never sampled.
    assert_eq!(s.passes()[2].color_store(), StoreAction::DontCare);
    // Pass 3 rt_a → no next rt_a use → Rule D flips.
    assert_eq!(s.passes()[3].color_store(), StoreAction::DontCare);
    // Pass 4 backbuffer() → last in frame, exempt from Rule D (Present reads it).
    assert_eq!(s.passes()[4].color_store(), StoreAction::Store);
}

#[test]
fn rule_c_color_walk_independent_of_depth_walk() {
    // Two passes share backbuffer() + depth(), second pass starts with a
    // color clear. Rule C flips pass 0's color store, Rule B flips
    // pass 1's depth store, and pass 0's depth keeps Store (Rule B
    // only flips the LAST pass per depth texture).
    let rt = tex(0x3000);
    let mut s = fresh();
    s.emit_command(dummy_draw());
    // Force a pass break with a pending color clear on backbuffer().
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.clear_color(1, 1, 1, 1);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 3);
    // Pass 0 backbuffer() → next backbuffer() is pass 2 Clear → flip color.
    assert_eq!(s.passes()[0].color_store(), StoreAction::DontCare);
    // All three share depth(); only pass 2's depth_store flips.
    assert_eq!(s.passes()[0].depth_store(), StoreAction::Store);
    assert_eq!(s.passes()[1].depth_store(), StoreAction::Store);
    assert_eq!(s.passes()[2].depth_store(), StoreAction::DontCare);
}

// ── Sampler-aware exemptions (CSM sampling) ──────────────────────

#[test]
fn rule_b_keeps_store_when_depth_sampled_later() {
    let cascade_depth = tex(0x9000);
    let cascade_color = tex(0x3000);
    // Cascade sampling: cascade depth is written in pass 0, sampled
    // by the scene PS in pass 1. Rule B must NOT flip pass 0's
    // depth_store to DontCare or the scene's `sample_compare` reads
    // tile memory that was never preserved to VRAM.
    let mut s = fresh();
    // Pass 0: cascade caster pass — write into cascade_depth.
    s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT, RenderScale::IDENTITY);
    s.set_depth_stencil_attachment(cascade_depth, false, false);
    s.clear_depth(f32::to_bits(1.0));
    s.emit_command(dummy_draw());
    // Pass 1: scene pass — different rt+depth, sample cascade_depth.
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_depth_stencil_attachment(depth(), false, false);
    s.emit_command(Command::set_fragment_texture(cascade_depth.raw(), 4));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 2);
    // Pass 0 is the last (only) pass that depth-attaches cascade_depth;
    // without sampler awareness Rule B would flip Store→DontCare and
    // discard the caster depth before the scene PS samples it.
    assert_eq!(s.passes()[0].depth_texture(), cascade_depth);
    assert_eq!(s.passes()[0].depth_store(), StoreAction::Store);
    // Pass 1's depth (the scene depth) is never sampled this frame,
    // so the normal Rule B optimisation still applies there.
    assert_eq!(s.passes()[1].depth_texture(), depth());
    assert_eq!(s.passes()[1].depth_store(), StoreAction::DontCare);
}

#[test]
fn rule_a_reverts_dontcare_when_color_sampled_later() {
    let rt = tex(0x4000);
    // Pass 0 first-attaches a fresh color rt (Rule A: Load=DontCare),
    // pass 1 samples that same rt as a fragment texture. The eager
    // DontCare must be reverted at finalize so the sampler reads the
    // pass-0 content, not undefined tile memory.
    let mut s = fresh();
    // Bounce off backbuffer() first so the next pass-open is first-use of rt.
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    // First-use ⇒ Rule A flipped Load=DontCare eagerly.
    assert_eq!(s.passes()[1].color_load(), ColorLoad::DontCare);
    // Bounce back to backbuffer() and sample rt.
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(Command::set_fragment_texture(rt.raw(), 0));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_load_actions();
    // finalize_load_actions must revert pass 1's eager DontCare.
    assert_eq!(s.passes()[1].color_load(), ColorLoad::Load);
}

// ── Rule G: depth-only strip for clear-only passes ────────────

#[test]
fn rule_g_strips_color_from_clear_only_pass_with_wasted_color() {
    let cascade_color = tex(0x3000);
    let cascade_d0 = tex(0x9000);
    let cascade_d1 = tex(0x9100);
    // Cascade-init clear-only pass: cascade_color (Clear), depth
    // sampled by scene (so depth Store stays Store via Rule B).
    // Rule C flips color Store=DontCare because the next pass on
    // cascade_color also begins with Clear. Rule G should then
    // strip the color attachment so the pass becomes depth-only.
    let mut s = fresh();
    // Pass 0: cascade-color + cascade_d0, clear-only (no draws).
    s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT, RenderScale::IDENTITY);
    s.set_depth_stencil_attachment(cascade_d0, false, false);
    s.clear_color(1, 2, 3, 4);
    s.clear_depth(f32::to_bits(1.0));
    // Pass 1: same cascade_color but different depth. cascade_d0
    // is sampled in the scene pass later.
    s.set_depth_stencil_attachment(cascade_d1, false, false);
    s.clear_color(1, 2, 3, 4);
    s.clear_depth(f32::to_bits(1.0));
    s.emit_command(dummy_draw());
    // Scene pass samples cascade_d0 so its Store must stay.
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_depth_stencil_attachment(depth(), false, false);
    s.emit_command(Command::set_fragment_texture(cascade_d0.raw(), 4));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    s.strip_dead_color_in_clear_only_passes();
    s.cull_dead_clear_only_passes();
    // The cascade-d0 clear-only pass should now be depth-only:
    // color_texture stripped, depth_texture preserved.
    let stripped = s
        .passes()
        .iter()
        .find(|p| p.depth_texture() == cascade_d0)
        .expect("cascade_d0 pass must remain");
    assert_eq!(
        stripped.color_texture(),
        MetalHandle::NULL,
        "color stripped"
    );
    assert_eq!(stripped.depth_store(), StoreAction::Store);
}

// ── Rule F: dead clear-only pass culling ──────────────────────

#[test]
fn rule_f_culls_pass_where_both_stores_become_dontcare() {
    let cascade_color = tex(0x3000);
    let cascade_depth = tex(0x9000);
    // Pass 0: cascade_color (Clear) + cascade_depth (Clear), no
    // draws. cascade_depth is NEVER sampled this frame, so Rule B
    // flips depth Store=DontCare. cascade_color is non-backbuffer,
    // not sampled, last-use → Rule D flips color Store=DontCare.
    // Both Stores DontCare + no draws + no blits → Rule F culls.
    let mut s = fresh();
    s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT, RenderScale::IDENTITY);
    s.set_depth_stencil_attachment(cascade_depth, false, false);
    s.clear_color(1, 2, 3, 4);
    s.clear_depth(f32::to_bits(1.0));
    // No draws, no blits — pure clear-only pass.
    // Switch back to BB so this is the last cascade frame use.
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_depth_stencil_attachment(depth(), false, false);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    s.cull_dead_clear_only_passes();
    // The cascade clear-only pass should be gone; only the BB
    // scene pass remains.
    assert_eq!(s.passes().len(), 1);
    assert_eq!(s.passes()[0].color_texture(), backbuffer());
}

#[test]
fn rule_f_keeps_pass_where_depth_is_sampled() {
    let cascade_color = tex(0x3000);
    let cascade_depth = tex(0x9000);
    // Same as above but cascade_depth IS sampled by the scene pass
    // — Rule B keeps its Store=Store, so the cascade pass still
    // performs observable work (depth clear lands in VRAM for the
    // sampler). Rule F must NOT cull.
    let mut s = fresh();
    s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT, RenderScale::IDENTITY);
    s.set_depth_stencil_attachment(cascade_depth, false, false);
    s.clear_color(1, 2, 3, 4);
    s.clear_depth(f32::to_bits(1.0));
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_depth_stencil_attachment(depth(), false, false);
    s.emit_command(Command::set_fragment_texture(cascade_depth.raw(), 4));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    s.cull_dead_clear_only_passes();
    // Cascade clear-only pass stays — depth Store must commit to
    // VRAM for the scene's sample_compare to read it.
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[0].depth_texture(), cascade_depth);
    assert_eq!(s.passes()[0].depth_store(), StoreAction::Store);
}

// ── Rule E: clear-only pass coalescing ────────────────────────

#[test]
fn rule_e_bb_clear_coalesces_into_scene_pass() {
    let other_rt = tex(0x3000);
    // The canonical WoW frame pattern that produced spurious BB
    // clear passes: Clear(BB) → SetRT(other) → … → SetRT(BB) →
    // Draw. The clear-only BB pass should fold into the scene
    // pass's color_load.
    let mut s = fresh();
    s.clear_color(7, 7, 7, 7);
    // Switch rt — currently materialises a spurious BB clear pass.
    s.set_color_render_target(other_rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    // Come back to BB and draw.
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();
    // Three passes pre-coalesce: BB clear-only, other_rt draw, BB
    // draw. Post-coalesce: two — other_rt, BB-with-Clear-load.
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[0].color_texture(), other_rt);
    assert_eq!(s.passes()[1].color_texture(), backbuffer());
    assert!(matches!(
        s.passes()[1].color_load(),
        ColorLoad::Clear {
            r: 7,
            g: 7,
            b: 7,
            a: 7
        }
    ));
}

#[test]
fn a_stencil_clear_paints_once_the_plane_is_in_use() {
    // Metal's loadAction covers the whole attachment. Once the frame has
    // drawn into the depth-stencil texture, folding a later clear into a
    // load action would wipe those tiles, so the decision has to be a
    // scissored quad instead. This is the shadow-volume shape: clear,
    // draw, then clear again under a narrowed viewport.
    let ds = tex(0x3300);
    let mut s = fresh();
    s.set_depth_stencil_attachment(ds, false, true);

    // Nothing drawn yet: folding is observationally identical.
    assert_eq!(s.clear_stencil(5), StencilClearOutcome::Folded);

    s.ensure_pass_open();
    s.emit_command(dummy_draw());

    // The plane now carries this frame's content, so it must be painted.
    assert!(
        matches!(
            s.clear_stencil(7),
            StencilClearOutcome::EmitQuad { value: 7, .. }
        ),
        "a clear over an in-use plane must be a quad, not a load action"
    );
}

#[test]
fn a_stencil_clear_under_a_zero_area_viewport_is_a_no_op() {
    // Under identity scale a zero viewport means "unset" and reads as the
    // whole attachment, so the degenerate case only arises when the
    // game's viewport rounds to nothing at render resolution. D3D9 clears
    // nothing for a zero-area viewport. The fall-through this replaces
    // folded a whole-attachment clear into a pass that already held
    // draws, ahead of them.
    let ds = tex(0x3300);
    let mut s = fresh_scaled();
    s.set_depth_stencil_attachment(ds, false, true);
    s.emit_command(dummy_draw());
    let before = s.passes()[0].stencil_load();
    s.set_viewport(1, 1, 1, 1, 0.0, 1.0);
    assert_eq!(s.effective_viewport(), (1, 1, 0, 0));

    assert_eq!(s.clear_stencil(7), StencilClearOutcome::NoOp);
    assert_eq!(s.passes().len(), 1);
    assert!(!s.current_pass_closed(), "the live pass stays open");
    assert_eq!(
        s.passes()[0].stencil_load(),
        before,
        "a pass with draws keeps its load action"
    );
    assert!(s.pending_stencil_clear.is_none());
}

#[test]
fn a_stencil_clear_under_a_zero_area_viewport_opens_no_pass() {
    // The same degenerate viewport between passes: nothing to paint, so
    // no encoder is opened for it either.
    let ds = tex(0x3300);
    let mut s = fresh_scaled();
    s.set_depth_stencil_attachment(ds, false, true);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.set_viewport(1, 1, 1, 1, 0.0, 1.0);

    assert_eq!(s.clear_stencil(7), StencilClearOutcome::NoOp);
    assert_eq!(s.passes().len(), 1);
    assert!(s.current_pass_closed());
    assert!(s.pending_stencil_clear.is_none());
}

#[test]
fn a_depth_clear_under_a_zero_area_viewport_is_a_no_op() {
    // Depth twin of the stencil case: a viewport that rounds to nothing
    // at render resolution used to paint a zero-size quad, paying the
    // pipeline and state switches around it for no pixels.
    let mut s = fresh_scaled();
    s.emit_command(dummy_draw());
    let before = s.passes()[0].depth_load();
    s.set_viewport(1, 1, 1, 1, 0.0, 1.0);
    assert_eq!(s.effective_viewport(), (1, 1, 0, 0));

    assert_eq!(s.clear_depth(f32::to_bits(1.0)), DepthClearOutcome::NoOp);
    assert_eq!(s.passes().len(), 1);
    assert!(!s.current_pass_closed(), "the live pass stays open");
    assert_eq!(s.passes()[0].depth_load(), before);
    assert!(s.pending_depth_clear.is_none());
}

#[test]
fn a_depth_clear_under_a_zero_area_viewport_opens_no_pass() {
    let mut s = fresh_scaled();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.set_viewport(1, 1, 1, 1, 0.0, 1.0);

    assert_eq!(s.clear_depth(f32::to_bits(1.0)), DepthClearOutcome::NoOp);
    assert_eq!(s.passes().len(), 1);
    assert!(s.current_pass_closed());
    assert!(s.pending_depth_clear.is_none());
}

#[test]
fn a_depth_clear_with_no_depth_attachment_is_a_no_op() {
    let mut s = fresh();
    s.set_depth_stencil_attachment(MetalHandle::NULL, false, false);

    assert_eq!(s.clear_depth(f32::to_bits(1.0)), DepthClearOutcome::NoOp);
    assert!(s.pending_depth_clear.is_none());
    assert!(s.passes().is_empty());
}

#[test]
fn a_stencil_clear_with_no_depth_attachment_is_a_no_op() {
    // Nothing is attached, so there is nothing to fold or paint; stashing
    // would clear whatever texture the next pass happens to attach.
    let mut s = fresh();
    s.set_depth_stencil_attachment(MetalHandle::NULL, false, false);

    assert_eq!(s.clear_stencil(1), StencilClearOutcome::NoOp);
    assert!(s.pending_stencil_clear.is_none());
    assert!(s.passes().is_empty());
}

#[test]
fn depth_and_stencil_clears_over_draws_paint_matching_quads() {
    // Clear(ZBUFFER | STENCIL) asks the two chains in turn and paints one
    // quad when both answer EmitQuad over the same rect, which they do
    // because neither call changes the state the other reads.
    let ds = tex(0x3300);
    let mut s = fresh();
    s.set_depth_stencil_attachment(ds, false, true);
    s.emit_command(dummy_draw());

    let depth = s.clear_depth(f32::to_bits(1.0));
    let stencil = s.clear_stencil(1);
    let DepthClearOutcome::EmitQuad { viewport: dvp, .. } = depth else {
        panic!("depth over draws must paint, got {depth:?}");
    };
    let StencilClearOutcome::EmitQuad { viewport: svp, .. } = stencil else {
        panic!("stencil over draws must paint, got {stencil:?}");
    };
    assert_eq!(dvp, svp);
    assert_eq!(s.passes().len(), 1, "both quads land in the live pass");
}

#[test]
fn depth_and_stencil_clears_under_a_counting_query_share_the_fresh_pass() {
    // With a visibility query armed the depth chain ends the pass and
    // reopens one with Load; the stencil chain then finds that fresh pass
    // with no draws and paints into it as well. The pass count is what
    // the encoder uses to know its binding cache must start over.
    let ds = tex(0x3300);
    let mut s = fresh();
    s.set_depth_stencil_attachment(ds, false, true);
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    s.emit_command(dummy_draw());
    s.emit_command(Command::set_visibility_result_mode(
        mtld3d_shared::mtl::VisibilityResultMode::Counting,
        0,
    ));

    let depth = s.clear_depth(f32::to_bits(1.0));
    let stencil = s.clear_stencil(1);
    assert!(matches!(depth, DepthClearOutcome::EmitQuad { .. }));
    assert!(matches!(stencil, StencilClearOutcome::EmitQuad { .. }));
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[1].depth_load(), DepthLoad::Load);
    assert_eq!(s.passes()[1].stencil_load(), StencilLoad::Load);
}

#[test]
fn rule_e_carries_the_stencil_clear_into_the_merge_target() {
    // Same shape as the colour case, but the clear-only pass carries a
    // stencil clear. Folding only colour and depth would delete the pass
    // and the stencil clear with it, leaving the plane holding the
    // previous frame's values.
    let other_rt = tex(0x3100);
    let ds = tex(0x3200);
    let mut s = fresh();
    s.set_depth_stencil_attachment(ds, false, true);
    s.clear_color(1, 2, 3, 4);
    s.clear_stencil(0x2A);
    s.set_color_render_target(other_rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();

    // Either outcome is correct as long as the clear is still there: the
    // fold moves it into the target, and a refused fold leaves the
    // clear-only pass standing.
    let survivors = s.passes();
    assert!(
        survivors
            .iter()
            .any(|p| matches!(p.stencil_load(), StencilLoad::Clear { value: 0x2A })),
        "the stencil clear must survive coalescing"
    );
}

#[test]
fn rule_e_aborts_when_intervening_pass_samples_target() {
    let rt = tex(0x4000);
    // If something between the clear-only pass and the candidate
    // merge target SAMPLES the texture, moving the Clear past it
    // would change the read; the merge must be rejected.
    let mut s = fresh();
    // Pass 0: clear-only on rt.
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.clear_color(1, 2, 3, 4);
    // Force the pending clear to materialise by hopping rt
    // (combined flush).
    s.set_color_render_target(tex(0x5000), 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(Command::set_fragment_texture(rt.raw(), 0));
    s.emit_command(dummy_draw());
    // Re-attach rt and draw. Without the read at 0x5000 this would
    // be a valid merge target, but the intervening sample disables it.
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let before = s.passes().len();
    s.coalesce_clear_only_passes();
    // Coalesce must not delete the clear-only pass.
    assert_eq!(s.passes().len(), before);
    // rt's clear-only pass is still there with its Clear load action.
    let cleared = s
        .passes()
        .iter()
        .find(|p| p.color_texture() == rt && matches!(p.color_load(), ColorLoad::Clear { .. }))
        .expect("clear-only rt pass must remain");
    let cmds = cleared.commands();
    let has_draw = cmds.iter().any(|c| {
        c.cmd == mtld3d_shared::CommandType::DrawPrimitives as u32
            || c.cmd == mtld3d_shared::CommandType::DrawIndexedPrimitives as u32
    });
    assert!(!has_draw);
}

#[test]
fn rule_d_non_backbuffer_color_last_use_is_dontcare() {
    let cascade_color = tex(0x3000);
    // CSM cascade color is a placeholder for the depth-only caster
    // pass and is never sampled. Rule D must flip its Store to
    // DontCare so the 16 MB writeback doesn't hit VRAM. Backbuffer
    // color in the next pass must stay Store (Present consumes it).
    let mut s = fresh();
    // Pass 0: cascade caster pass — color is junk, depth gets work.
    s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    // Pass 1: scene pass on backbuffer.
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[0].color_texture(), cascade_color);
    assert_eq!(s.passes()[0].color_store(), StoreAction::DontCare);
    // Backbuffer Present needs the pixels — exempt from Rule D.
    assert_eq!(s.passes()[1].color_texture(), backbuffer());
    assert_eq!(s.passes()[1].color_store(), StoreAction::Store);
}

#[test]
fn rule_d_keeps_store_when_color_sampled_later() {
    let rt = tex(0x4000);
    // A non-backbuffer color rt that IS sampled by a later pass
    // must preserve its content; Rule D must NOT flip Store to
    // DontCare for it.
    let mut s = fresh();
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(Command::set_fragment_texture(rt.raw(), 0));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[0].color_texture(), rt);
    assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
}

#[test]
fn cascade_init_sequence_collapses_to_one_pass() {
    let cascade_color = tex(0x3000);
    let cascade_depth = tex(0x9000);
    // WoW's typical cascade-init sequence is
    //   SetRT(C) → Clear(TARGET) → SetDST(D) → Clear(ZBUFFER) → Draw.
    // The pending color clear when SetDST fires applies to the
    // *unchanged* color rt C, so it must survive the depth-attach
    // switch and combine with the depth clear on the next pass.
    // Without the split flush, this would produce a spurious
    // 1-cmd clear-only pass for C with the still-old depth.
    let mut s = fresh();
    s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT, RenderScale::IDENTITY);
    s.clear_color(1, 2, 3, 4);
    s.set_depth_stencil_attachment(cascade_depth, false, false);
    s.clear_depth(f32::to_bits(1.0));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    // One pass — no spurious clear-only pass dropped between the two
    // clears.
    assert_eq!(s.passes().len(), 1);
    assert_eq!(s.passes()[0].color_texture(), cascade_color);
    assert_eq!(s.passes()[0].depth_texture(), cascade_depth);
    // Both clears land on the single pass's load actions.
    assert!(matches!(
        s.passes()[0].color_load(),
        ColorLoad::Clear {
            r: 1,
            g: 2,
            b: 3,
            a: 4
        }
    ));
    assert!(matches!(
        s.passes()[0].depth_load(),
        DepthLoad::Clear { .. }
    ));
}

#[test]
fn pending_color_clear_survives_depth_attach_change() {
    let d2 = tex(0x9000);
    // Narrow assertion: when only the depth attachment changes and
    // a color clear is pending, the clear stays pending (does not
    // materialise into a spurious pass).
    let mut s = fresh();
    s.clear_color(7, 7, 7, 7);
    s.set_depth_stencil_attachment(d2, false, false);
    // No draws yet — the pending color clear should still be
    // pending on the same (unchanged) color rt.
    assert!(s.passes().is_empty());
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 1);
    assert!(matches!(
        s.passes()[0].color_load(),
        ColorLoad::Clear {
            r: 7,
            g: 7,
            b: 7,
            a: 7
        }
    ));
}

#[test]
fn rule_c_skips_color_store_dontcare_when_sampled_between() {
    let rt = tex(0x5000);
    // Pass 0 writes rt, pass 1 samples rt, pass 2 re-clears rt.
    // Rule C would naively flip pass 0's color_store to DontCare
    // because the next consumer (pass 2) begins with Clear — but
    // pass 1 in between samples rt, so the content must survive to
    // VRAM. Sampler-aware Rule C keeps pass 0 Store.
    let mut s = fresh();
    // Pass 0: write to rt.
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    // Pass 1: sample rt into backbuffer().
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(Command::set_fragment_texture(rt.raw(), 0));
    s.emit_command(dummy_draw());
    // Pass 2: clear+rewrite rt.
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.clear_color(0, 0, 0, 0);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 3);
    assert_eq!(s.passes()[0].color_texture(), rt);
    // Without the sampler check, pass 0's color_store would be
    // DontCare (next consumer at pass 2 begins with Clear).
    // Sampler-aware Rule C keeps Store because pass 1 reads rt.
    assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
}

// ── Rule H — strip color from passes-with-draws where every draw
// ── ran with COLORWRITEENABLE = 0. Side-map of with-color → no-color
// ── pipeline handles is supplied by the caller (built by the
// ── FrameEncoder at draw time from zero-mask snapshots).

const PSO_WITH: u64 = 0xAAAA_1111;
const PSO_NO_COLOR: u64 = 0xBBBB_2222;

fn set_pso(handle: u64) -> Command {
    Command::set_render_pipeline_state(handle)
}

#[test]
fn rule_h_strips_color_when_all_draws_have_writemask_zero() {
    let mut s = fresh();
    // Five zero-mask draws into the backbuffer + depth pass.
    for _ in 0..5 {
        s.note_draw_color_write_mask(0);
        s.emit_command(set_pso(PSO_WITH));
        s.emit_command(dummy_draw());
    }
    s.end_current_pass("test");
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    s.strip_color_from_no_color_draw_passes(&alt);
    let pass = &s.passes()[0];
    assert_eq!(
        pass.color_texture(),
        MetalHandle::NULL,
        "color attachment stripped"
    );
    assert_eq!(pass.color_load(), ColorLoad::DontCare);
    assert_eq!(pass.color_store(), StoreAction::DontCare);
    // Every SetPSO in the pass now binds the no-color variant.
    let pso_handles: Vec<u64> = pass
        .commands()
        .iter()
        .filter(|c| c.cmd == CommandType::SetRenderPipelineState as u32)
        .map(|c| c.param_b)
        .collect();
    assert!(!pso_handles.is_empty(), "test setup emitted SetPSO");
    assert!(
        pso_handles.iter().all(|h| *h == PSO_NO_COLOR),
        "every SetPSO rewritten: {pso_handles:?}"
    );
}

#[test]
fn rule_h_keeps_color_when_any_draw_writes_color() {
    let mut s = fresh();
    for _ in 0..4 {
        s.note_draw_color_write_mask(0);
        s.emit_command(set_pso(PSO_WITH));
        s.emit_command(dummy_draw());
    }
    // One non-zero-mask draw flips the pass's tag.
    s.note_draw_color_write_mask(0xF);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    s.strip_color_from_no_color_draw_passes(&alt);
    let pass = &s.passes()[0];
    assert_eq!(pass.color_texture(), backbuffer(), "color attachment kept");
    assert!(pass.color_writes_observed());
    // SetPSO handles preserved unchanged.
    assert!(
        pass.commands()
            .iter()
            .filter(|c| c.cmd == CommandType::SetRenderPipelineState as u32)
            .all(|c| c.param_b == PSO_WITH),
        "no rewrite on color-writing pass"
    );
}

#[test]
fn rule_h_skipped_without_depth_attachment() {
    let mut s = fresh();
    // Detach depth so the candidate pass has color but no depth —
    // stripping color would produce an encoder with zero
    // attachments, which Metal rejects.
    s.set_depth_stencil_attachment(MetalHandle::NULL, false, false);
    s.note_draw_color_write_mask(0);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    s.strip_color_from_no_color_draw_passes(&alt);
    let pass = &s.passes()[0];
    assert_eq!(
        pass.color_texture(),
        backbuffer(),
        "no-depth pass must keep color"
    );
}

#[test]
fn rule_h_skipped_for_clear_only_pass() {
    // A pass with zero draws is Rule G's territory, not Rule H's.
    // Rule H must leave it alone so finalize-time invariants hold.
    let mut s = fresh();
    s.clear_color(0, 0, 0, 0);
    s.flush_pending_clears();
    let alt: FxHashMap<u64, MetalHandle<MTLRenderPipelineStateKind>> = FxHashMap::default();
    s.strip_color_from_no_color_draw_passes(&alt);
    let pass = &s.passes()[0];
    // Color still attached — Rule H bailed because the pass had
    // no draw commands.
    assert_eq!(pass.color_texture(), backbuffer());
}

#[test]
fn rule_h_aborts_strip_on_missing_alt_handle() {
    // A zero-mask draw bound PSO_WITH but the side-map is empty
    // (would mean the FrameEncoder skipped the dual-build path —
    // an upstream bug). The rule must keep the color attachment
    // intact rather than bind a with-color pipeline against a
    // depth-only render pass descriptor.
    let mut s = fresh();
    s.note_draw_color_write_mask(0);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let alt: FxHashMap<u64, MetalHandle<MTLRenderPipelineStateKind>> = FxHashMap::default();
    s.strip_color_from_no_color_draw_passes(&alt);
    let pass = &s.passes()[0];
    assert_eq!(
        pass.color_texture(),
        backbuffer(),
        "missing alt-handle → no strip"
    );
    assert_eq!(
        pass.commands()
            .iter()
            .find(|c| c.cmd == CommandType::SetRenderPipelineState as u32)
            .map(|c| c.param_b),
        Some(PSO_WITH),
        "no rewrite on aborted strip"
    );
}

#[test]
fn rule_h_strips_color_with_self_mapped_depth_clear_quad() {
    const PSO_CASTER: u64 = 0xAAAA_1111;
    const PSO_CASTER_NO_COLOR: u64 = 0xBBBB_2222;
    const PSO_CLEAR_QUAD_DEPTH: u64 = 0xCCCC_3333;
    // Cascade caster pass: per-tile depth clear-quad SetPSO +
    // zero-mask caster SetPSO + draws. The depth clear-quad
    // pipeline is built `has_color: false` and self-maps in
    // `no_color_pipeline_alt` (encoder.rs); Rule H must strip
    // color cleanly without firing the side-map-miss warn.

    let mut s = fresh();
    for _ in 0..3 {
        s.emit_command(set_pso(PSO_CLEAR_QUAD_DEPTH));
        s.emit_command(dummy_draw());
        s.note_draw_color_write_mask(0);
        s.emit_command(set_pso(PSO_CASTER));
        s.emit_command(dummy_draw());
    }
    s.end_current_pass("test");

    let mut alt = FxHashMap::default();
    alt.insert(PSO_CASTER, pso(PSO_CASTER_NO_COLOR));
    alt.insert(PSO_CLEAR_QUAD_DEPTH, pso(PSO_CLEAR_QUAD_DEPTH));
    s.strip_color_from_no_color_draw_passes(&alt);

    let pass = &s.passes()[0];
    assert_eq!(
        pass.color_texture(),
        MetalHandle::NULL,
        "color attachment stripped"
    );
    let pso_handles: Vec<u64> = pass
        .commands()
        .iter()
        .filter(|c| c.cmd == CommandType::SetRenderPipelineState as u32)
        .map(|c| c.param_b)
        .collect();
    assert!(
        pso_handles.contains(&PSO_CASTER_NO_COLOR),
        "caster rewritten to no-color sibling: {pso_handles:?}"
    );
    assert!(
        pso_handles.contains(&PSO_CLEAR_QUAD_DEPTH),
        "self-mapped depth clear-quad preserved: {pso_handles:?}"
    );
    assert!(
        !pso_handles.contains(&PSO_CASTER),
        "caster with-color handle replaced: {pso_handles:?}"
    );
}

#[test]
fn rule_h_keeps_back_buffer_color_clear_quad_beside_zero_mask_draws() {
    const PSO_CLEAR_QUAD_COLOR: u64 = 0xCAFE_BABE;
    // A cross-pass colour clear-quad on the back buffer shares a pass with
    // a zero-mask draw (a depth clear-quad, say). The back buffer is
    // presented, so the clear is observable and the pass keeps its colour.
    let mut s = fresh();
    let start = s.open_color_clear_quad_block();
    s.emit_command(set_pso(PSO_CLEAR_QUAD_COLOR));
    s.emit_command(dummy_draw());
    s.close_color_clear_quad_block(start);
    s.note_draw_color_write_mask(0);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    s.strip_color_from_no_color_draw_passes(&alt);
    let pass = &s.passes()[0];
    assert_eq!(pass.color_texture(), backbuffer(), "colour kept");
    assert_eq!(pass.color_clear_quad_ranges().len(), 1);
}

#[test]
fn rule_h_keeps_color_clear_quad_a_later_pass_loads() {
    const PSO_CLEAR_QUAD_COLOR: u64 = 0xCAFE_BABE;
    // Pass 0 clears an offscreen target with a colour clear-quad beside
    // zero-mask draws; pass 1 reattaches the target with `Load` and draws
    // colour, so it observes the clear. Pass 0 keeps its colour; a third
    // pass of pure zero-mask draws on the target, loading it but stripped
    // itself, is not an observer.
    let mut s = fresh();
    let atlas = tex(0x3000);
    s.set_color_render_target(atlas, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    let start = s.open_color_clear_quad_block();
    s.emit_command(set_pso(PSO_CLEAR_QUAD_COLOR));
    s.emit_command(dummy_draw());
    s.close_color_clear_quad_block(start);
    s.note_draw_color_write_mask(0);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.note_draw_color_write_mask(0xF);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.note_draw_color_write_mask(0);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    s.strip_color_from_no_color_draw_passes(&alt);
    assert_eq!(s.passes()[0].color_texture(), atlas, "observed clear kept");
    assert_eq!(
        s.passes()[1].color_texture(),
        atlas,
        "colour-writing pass kept"
    );
    assert_eq!(
        s.passes()[2].color_texture(),
        MetalHandle::NULL,
        "trailing zero-mask pass stripped"
    );
}

#[test]
fn rule_h_strips_color_and_clear_quad_when_only_zero_mask_draws_plus_clear_quad() {
    const PSO_CLEAR_QUAD_COLOR: u64 = 0xCAFE_BABE;
    // Cascade caster pass shape: WoW issued mid-pass `Clear` on
    // the cascade color atlas (e.g. per-tile clear), which the
    // encoder folded into a cross-pass color clear-quad. The rest
    // of the pass is zero-mask caster draws. Rule H must strip
    // the color attachment AND drain the clear-quad's commands so
    // the resulting depth-only descriptor doesn't try to bind a
    // color-output clear-quad pipeline. The atlas is a texture of its
    // own that nothing later in the frame observes.
    let mut s = fresh();
    s.set_color_render_target(tex(0x3000), 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    // Color clear-quad block — 6 commands, none of which should
    // tag `color_writes_observed`.
    let start = s.open_color_clear_quad_block();
    s.emit_command(set_pso(PSO_CLEAR_QUAD_COLOR));
    s.emit_command(dummy_draw());
    s.close_color_clear_quad_block(start);
    // Two zero-mask caster draws.
    for _ in 0..2 {
        s.note_draw_color_write_mask(0);
        s.emit_command(set_pso(PSO_WITH));
        s.emit_command(dummy_draw());
    }
    s.end_current_pass("test");
    // Only the real caster needs a side-map entry; clear-quad
    // PSOs are removed wholesale and don't need to resolve.
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    s.strip_color_from_no_color_draw_passes(&alt);

    let pass = &s.passes()[0];
    assert_eq!(
        pass.color_texture(),
        MetalHandle::NULL,
        "color attachment stripped"
    );
    assert_eq!(pass.color_load(), ColorLoad::DontCare);
    assert_eq!(pass.color_store(), StoreAction::DontCare);
    assert!(
        pass.color_clear_quad_ranges().is_empty(),
        "clear-quad ranges drained after strip"
    );
    // Caster SetPSO rewritten; clear-quad SetPSO gone.
    let pso_handles: Vec<u64> = pass
        .commands()
        .iter()
        .filter(|c| c.cmd == CommandType::SetRenderPipelineState as u32)
        .map(|c| c.param_b)
        .collect();
    assert!(
        !pso_handles.contains(&PSO_CLEAR_QUAD_COLOR),
        "color clear-quad SetPSO removed: {pso_handles:?}"
    );
    assert!(
        !pso_handles.contains(&PSO_WITH),
        "caster with-color handle replaced: {pso_handles:?}"
    );
    assert!(
        pso_handles.iter().all(|h| *h == PSO_NO_COLOR),
        "every surviving SetPSO is the no-color variant: {pso_handles:?}"
    );
}

#[test]
fn rule_h_keeps_color_clear_quad_when_real_color_writing_draw_present() {
    const PSO_CLEAR_QUAD_COLOR: u64 = 0xCAFE_BABE;
    // Same shape as above but one real draw writes color
    // (`COLORWRITEENABLE != 0`). The clear-quad output is now
    // load-bearing for that draw's blend, so Rule H must skip the
    // pass entirely — both the attachment AND the clear-quad
    // commands must survive untouched.
    let mut s = fresh();
    let start = s.open_color_clear_quad_block();
    s.emit_command(set_pso(PSO_CLEAR_QUAD_COLOR));
    s.emit_command(dummy_draw());
    s.close_color_clear_quad_block(start);
    // One real color-writing draw.
    s.note_draw_color_write_mask(0xF);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");

    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    s.strip_color_from_no_color_draw_passes(&alt);

    let pass = &s.passes()[0];
    assert_eq!(
        pass.color_texture(),
        backbuffer(),
        "real color-writing draw keeps the attachment"
    );
    assert!(pass.color_writes_observed());
    assert_eq!(
        pass.color_clear_quad_ranges().len(),
        1,
        "clear-quad range preserved"
    );
    let pso_handles: Vec<u64> = pass
        .commands()
        .iter()
        .filter(|c| c.cmd == CommandType::SetRenderPipelineState as u32)
        .map(|c| c.param_b)
        .collect();
    assert!(
        pso_handles.contains(&PSO_CLEAR_QUAD_COLOR),
        "clear-quad SetPSO preserved: {pso_handles:?}"
    );
    assert!(
        pso_handles.contains(&PSO_WITH),
        "real-draw SetPSO not rewritten: {pso_handles:?}"
    );
}

#[test]
fn rule_h_skipped_when_pass_has_only_color_clear_quad_no_real_draws() {
    const PSO_CLEAR_QUAD_COLOR: u64 = 0xCAFE_BABE;
    // A pass with ONLY a color clear-quad and no real draw is not
    // Rule H's territory — it leaves the pass intact (Rule F /
    // Rule G handle the clear-only shape elsewhere). The
    // clear-quad's color writes are kept; if they're wasted,
    // upstream rules cull the pass.
    let mut s = fresh();
    let start = s.open_color_clear_quad_block();
    s.emit_command(set_pso(PSO_CLEAR_QUAD_COLOR));
    s.emit_command(dummy_draw());
    s.close_color_clear_quad_block(start);
    s.end_current_pass("test");

    let alt: FxHashMap<u64, MetalHandle<MTLRenderPipelineStateKind>> = FxHashMap::default();
    s.strip_color_from_no_color_draw_passes(&alt);

    let pass = &s.passes()[0];
    assert_eq!(
        pass.color_texture(),
        backbuffer(),
        "clear-quad-only pass left alone by Rule H"
    );
    assert_eq!(pass.color_clear_quad_ranges().len(), 1);
}

// ── Clear-quad mid-pass Clear translation ─────────────────────

/// A shared shadow tile-atlas pattern.
///
/// Open a single pass on a cascade depth texture, then for each of N
/// tiles emit `set_viewport(tile_N) + clear_depth(1.0) + draw`.
/// Under Metal's full-attachment Clear semantics, breaking a pass
/// per tile would emit N separate passes each `loadAction = Clear`,
/// wiping the prior tile's draws; the clear-quad path instead keeps
/// one pass open and returns N `EmitQuad` outcomes the encoder
/// layer translates into scissored fullscreen-triangle draws.
#[test]
fn wow_tile_atlas_clears_emit_inline_quad_not_pass_break() {
    const TILE_COUNT: u32 = 9;
    let mut s = fresh();
    // Establish a pass open on the depth attachment with one draw,
    // so the first per-tile Clear arrives at a "has work" pass.
    s.emit_command(dummy_draw());
    let z = f32::to_bits(1.0);
    let mut quad_outcomes: u32 = 0;
    for tile in 0..TILE_COUNT {
        let x = (tile % 3) * 683;
        let y = (tile / 3) * 683;
        s.set_viewport(x, y, 683, 683, 0.0, 1.0);
        match s.clear_depth(z) {
            DepthClearOutcome::EmitQuad {
                value, viewport, ..
            } => {
                assert_eq!(value, z);
                assert_eq!(viewport, (x, y, 683, 683));
                quad_outcomes += 1;
            }
            DepthClearOutcome::Folded | DepthClearOutcome::NoOp => {
                panic!("tile {tile} clear should have returned EmitQuad");
            }
        }
        s.emit_command(dummy_draw());
    }
    assert_eq!(
        quad_outcomes, TILE_COUNT,
        "every tile-clear must emit a quad outcome"
    );
    assert_eq!(
        s.passes().len(),
        1,
        "single pass should survive the entire tile sequence"
    );
}

/// Color mirror of the depth tile-atlas test.
#[test]
fn wow_color_clear_mid_pass_returns_emit_quad_per_tile() {
    let mut s = fresh();
    s.emit_command(dummy_draw());
    let outcome = s.clear_color(0x11, 0x22, 0x33, 0x44);
    assert!(matches!(
        outcome,
        ColorClearOutcome::EmitQuad {
            rgba: (0x11, 0x22, 0x33, 0x44),
            ..
        }
    ));
    assert_eq!(s.passes().len(), 1);
}

/// First Clear in a pass still folds into the pass's load action.
///
/// The pass has only the implicit viewport command, no draws —
/// Metal's `loadAction = Clear` is the cheap path here. Quad
/// emission only kicks in once real work has been added.
#[test]
fn first_depth_clear_in_pass_folds_into_load_action() {
    let mut s = fresh();
    s.ensure_pass_open();
    let z = f32::to_bits(1.0);
    let outcome = s.clear_depth(z);
    assert_eq!(outcome, DepthClearOutcome::Folded);
    assert_eq!(s.passes().len(), 1);
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::Clear { value: z });
}

/// Two mid-pass Clears with different values both emit their own quad outcome.
///
/// The encoder will materialise both with their distinct depths in
/// the same encoder.
#[test]
fn distinct_depth_clear_values_in_same_pass_each_emit_quad() {
    let mut s = fresh();
    s.emit_command(dummy_draw());
    let z1 = f32::to_bits(0.5);
    let z2 = f32::to_bits(0.75);
    let o1 = s.clear_depth(z1);
    s.emit_command(dummy_draw());
    let o2 = s.clear_depth(z2);
    s.emit_command(dummy_draw());
    assert!(matches!(o1, DepthClearOutcome::EmitQuad { value, .. } if value == z1));
    assert!(matches!(o2, DepthClearOutcome::EmitQuad { value, .. } if value == z2));
    assert_eq!(s.passes().len(), 1);
}

/// Cross-pass case: a tile sequence where each tile is its own pass.
///
/// A fresh `SetRenderTarget` between tiles breaks the pass. First
/// tile's `Clear` lands as `pending_depth_clear` and the pass opens
/// with `loadAction = Clear` for the full attachment (correct —
/// first use of the texture). Second tile's `Clear` hits a CLOSED
/// pass on a depth texture *already seen* this frame — folding into
/// a fresh `loadAction = Clear` would let Metal wipe the first
/// tile's draws. The fix opens the second pass with
/// `loadAction = Load` and returns `EmitQuad` so the encoder layer
/// emits a scissored clear-quad inside the new pass.
#[test]
fn cross_pass_depth_clear_uses_load_plus_quad() {
    let mut s = fresh();
    // Tile 0: open the first pass on depth(); Clear folds into load
    // action; a draw lands in the pass; we end the pass (e.g. a
    // SetRenderTarget switch).
    let z = f32::to_bits(1.0);
    s.set_viewport(0, 0, 683, 683, 0.0, 1.0);
    assert_eq!(s.clear_depth(z), DepthClearOutcome::Folded);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 1);
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::Clear { value: z });
    s.end_current_pass("test_color_rt_switch");

    // Tile 1: Clear arrives on the same depth(). Depth is already
    // in seen_depth_rts → cross-pass case fires. Pass opens with
    // load=Load (preserving tile 0's content) and the outcome is
    // EmitQuad so the encoder emits a scissored clear-quad.
    s.set_viewport(683, 0, 683, 683, 0.0, 1.0);
    let outcome = s.clear_depth(z);
    assert!(
        matches!(
            outcome,
            DepthClearOutcome::EmitQuad { value, viewport, .. }
                if value == z && viewport == (683, 0, 683, 683)
        ),
        "cross-pass clear must return EmitQuad, got {outcome:?}",
    );
    assert_eq!(s.passes().len(), 2);
    assert_eq!(
        s.passes()[1].depth_load(),
        DepthLoad::Load,
        "tile 1 pass must use Load so Metal preserves tile 0",
    );
}

/// Sampleable shadow maps must keep `Store` even when not sampled in this frame.
///
/// The receiver may sample them on a future frame (cascade-3
/// rotations etc.). The `is_sampleable` flag on
/// `set_depth_stencil_attachment` covers the bootstrap-frame gap
/// that the persist-`seen_sampled` fix alone can't close for
/// rarely-sampled cascades.
#[test]
fn sampleable_depth_keeps_store_even_when_never_sampled() {
    let cascade_depth = tex(0xCAFE_5000);
    let mut s = fresh();
    s.set_depth_stencil_attachment(cascade_depth, /* is_sampleable */ true, false);
    s.emit_command(dummy_draw());
    s.finalize_store_actions(false);
    let cascade_pass = s
        .passes()
        .iter()
        .find(|p| p.depth_texture() == cascade_depth)
        .expect("cascade pass present");
    assert_eq!(
        cascade_pass.depth_store(),
        StoreAction::Store,
        "sampleable depth must keep Store even without a sample in seen_sampled",
    );
}

/// Non-sampleable depth still gets the Rule B optimization when no sample lands on it.
///
/// Non-sampleable means a standalone `CreateDepthStencilSurface`,
/// e.g. the backbuffer's z; the sample has to be absent for this
/// frame. Guards against the sampleable-flag fix accidentally
/// over-conservatively keeping Store on every depth attachment.
#[test]
fn non_sampleable_depth_still_gets_rule_b_dontcare() {
    let rt_depth = tex(0xCAFE_6000);
    let mut s = fresh();
    s.set_depth_stencil_attachment(
        rt_depth, /* is_sampleable */ false, /* has_stencil */ false,
    );
    s.emit_command(dummy_draw());
    s.finalize_store_actions(false);
    let rt_pass = s
        .passes()
        .iter()
        .find(|p| p.depth_texture() == rt_depth)
        .expect("rt pass present");
    assert_eq!(
        rt_pass.depth_store(),
        StoreAction::DontCare,
        "non-sampleable depth never sampled → Rule B optimization preserved",
    );
}

/// A cascade rebound with a stale `is_sampleable=false` is a no-op.
///
/// Neither the pass break nor the loss of Rule B's exemption fires.
///
/// `GetDepthStencilSurface` returns a surface with `parent_texture=null`,
/// so a save/restore cycle re-binds the same cascade handle through the
/// `Eager` path with `is_sampleable=false`. The sticky resolve against
/// `seen_sampleable_depth_textures` keeps the pass open and keeps Rule B's
/// keep-Store exemption in force.
#[test]
fn cascade_rebind_with_stale_sampleable_flag_is_a_no_op() {
    let cascade_depth = tex(0xCAFE_7000);
    let mut s = fresh();
    // First bind as sampleable, draw into it, then rebind the SAME handle
    // with a stale is_sampleable=false (the GetDepthStencilSurface path).
    s.set_depth_stencil_attachment(cascade_depth, true, false);
    s.emit_command(dummy_draw());
    let passes_before = s.passes().len();
    s.set_depth_stencil_attachment(cascade_depth, false, false);
    assert!(
        !s.current_pass_closed(),
        "a rebind of a known-sampleable cascade must not break the pass",
    );
    assert_eq!(
        s.passes().len(),
        passes_before,
        "no new pass opened by the rebind",
    );
    assert!(
        s.current_depth_is_sampleable(),
        "the sampleable flag stays set through the stale rebind",
    );
    s.emit_command(dummy_draw());
    s.finalize_store_actions(false);
    let cascade_pass = s
        .passes()
        .iter()
        .find(|p| p.depth_texture() == cascade_depth)
        .expect("cascade pass present");
    assert_eq!(
        cascade_pass.depth_store(),
        StoreAction::Store,
        "Rule B keeps Store through the stale rebind",
    );
}

/// Visibility-counting passes fall back to the legacy pass-break path.
///
/// Emitting a clear-quad mid-pass would falsely increment the
/// per-pass fragment counter; until proper save/restore of
/// `SetVisibilityResultMode` lands, the safe behaviour is to end
/// the pass on Clear-with-work as before.
#[test]
fn clear_depth_with_visibility_query_active_falls_back_to_pass_break() {
    let mut s = fresh();
    s.emit_command(dummy_draw());
    // Activate visibility counting on the current pass.
    s.emit_command(Command::set_visibility_result_mode(
        mtld3d_shared::mtl::VisibilityResultMode::Counting,
        0,
    ));
    let z = f32::to_bits(1.0);
    let outcome = s.clear_depth(z);
    assert_eq!(
        outcome,
        DepthClearOutcome::Folded,
        "visibility-active Clear must fall back to legacy pass-break (not EmitQuad)"
    );
}

/// A slot binding sized like the back buffer.
fn slot(texture: MetalHandle<MTLTextureKind>, size: (u32, u32)) -> ExtraColorSlot {
    ExtraColorSlot {
        texture,
        subresource: 0,
        size: (0, 0),
        logical_size: size,
        format: PixelFormat::R8Unorm,
        scale: RenderScale::IDENTITY,
        has_alpha: false,
    }
}

#[test]
fn extra_target_joins_the_pass_when_it_matches_rt0() {
    let mut s = fresh();
    s.set_extra_color_render_target(1, Some(slot(tex(0x3000), BB_SIZE)));
    assert_eq!(s.extra_present_mask(), 0b001);
    let attachments = s.extra_color_attachments();
    assert_eq!(attachments.present_mask, 0b001);
    assert_eq!(attachments.formats[0], PixelFormat::R8Unorm);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let pass = &s.passes()[0];
    assert_eq!(pass.extra_color()[0].texture(), tex(0x3000));
    assert!(!pass.extra_color()[1].is_bound());
    // First use under a covering viewport: Rule A, like render target 0.
    assert_eq!(pass.extra_color()[0].load(), ColorLoad::DontCare);
    assert_eq!(pass.extra_color()[0].store(), StoreAction::Store);
}

#[test]
fn mismatched_extra_target_stays_out_of_the_pass() {
    let mut s = fresh();
    s.set_extra_color_render_target(2, Some(slot(tex(0x3000), (128, 128))));
    assert_eq!(s.extra_present_mask(), 0);
    assert!(s.has_extra_color_targets());
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert!(!s.passes()[0].extra_color()[1].is_bound());
    // Rebinding render target 0 at the extra's size brings it in.
    s.set_color_render_target(tex(0x4000), 128, 128, BB_FORMAT, RenderScale::IDENTITY);
    assert_eq!(s.extra_present_mask(), 0b010);
}

#[test]
fn binding_an_extra_target_breaks_the_pass_and_rebinding_it_does_not() {
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.set_extra_color_render_target(1, Some(slot(tex(0x3000), BB_SIZE)));
    assert!(s.current_pass_closed());
    s.emit_command(dummy_draw());
    s.set_extra_color_render_target(1, Some(slot(tex(0x3000), BB_SIZE)));
    assert!(!s.current_pass_closed(), "same binding is a no-op");
    s.set_extra_color_render_target(1, None);
    assert!(s.current_pass_closed(), "unbinding ends the pass");
    assert_eq!(s.extra_present_mask(), 0);
}

#[test]
fn pending_clear_lands_on_every_present_target() {
    let mut s = fresh();
    s.set_extra_color_render_target(1, Some(slot(tex(0x3000), BB_SIZE)));
    s.set_extra_color_render_target(3, Some(slot(tex(0x3001), BB_SIZE)));
    assert_eq!(s.clear_color(1, 2, 3, 4), ColorClearOutcome::Folded);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let pass = &s.passes()[0];
    let clear = ColorLoad::Clear {
        r: 1,
        g: 2,
        b: 3,
        a: 4,
    };
    assert_eq!(pass.color_load(), clear);
    assert_eq!(pass.extra_color()[0].load(), clear);
    assert!(!pass.extra_color()[1].is_bound());
    assert_eq!(pass.extra_color()[2].load(), clear);
}

#[test]
fn clear_with_work_in_a_multi_target_pass_emits_one_quad() {
    // The quad writes every colour target of the pass, so the pass stays
    // open and nothing breaks per target.
    let mut s = fresh();
    s.set_extra_color_render_target(1, Some(slot(tex(0x3000), BB_SIZE)));
    s.emit_command(dummy_draw());
    assert!(matches!(
        s.clear_color(1, 2, 3, 4),
        ColorClearOutcome::EmitQuad { .. }
    ));
    assert!(!s.current_pass_closed());
    assert_eq!(s.passes().len(), 1);
}

#[test]
fn clear_after_a_target_was_drawn_opens_a_load_pass_for_every_target() {
    let mut s = fresh();
    s.set_extra_color_render_target(1, Some(slot(tex(0x3000), BB_SIZE)));
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    s.emit_command(dummy_draw());
    // A depth change ends the pass; the extra target has content now.
    s.set_depth_stencil_attachment(tex(0x5000), false, false);
    assert!(matches!(
        s.clear_color(1, 2, 3, 4),
        ColorClearOutcome::EmitQuad { .. }
    ));
    let pass = &s.passes()[1];
    assert_eq!(pass.color_load(), ColorLoad::Load);
    assert_eq!(pass.extra_color()[0].load(), ColorLoad::Load);
}

#[test]
fn take_and_restore_round_trip_the_binding_set() {
    let mut s = fresh();
    s.set_color_rt_has_alpha(false);
    s.set_extra_color_render_target(2, Some(slot(tex(0x3000), BB_SIZE)));
    s.emit_command(dummy_draw());
    let saved = s.take_color_attachments();
    assert!(s.current_pass_closed(), "taking the extras ends the pass");
    assert_eq!(s.extra_present_mask(), 0);
    assert!(!s.has_extra_color_targets());
    assert!(saved.slot(0).is_some());
    assert!(saved.slot(1).is_none());
    assert!(saved.extra_matches_rt0(2));
    // Bind slot 2 alone as render target 0, as the clear bracket does.
    let target = saved.slot(2).expect("slot 2 bound");
    s.set_color_render_target_subresource(
        target.texture,
        target.logical_size.0,
        target.logical_size.1,
        target.format,
        target.scale,
        (0, 0),
    );
    s.set_color_rt_has_alpha(target.has_alpha);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.restore_color_attachments(saved);
    assert_eq!(s.current_color_texture(), backbuffer());
    assert_eq!(s.current_color_format(), BB_FORMAT);
    assert!(!s.current_color_rt_has_alpha());
    assert_eq!(s.extra_present_mask(), 0b010);
    assert_eq!(s.current_depth_texture(), depth());
}

#[test]
fn depth_clear_in_a_multi_target_pass_stays_a_quad() {
    // The depth clear-quad declares the extra targets with an empty
    // write mask, so it runs inside the live pass like on a single
    // target.
    let mut s = fresh();
    s.set_extra_color_render_target(1, Some(slot(tex(0x3000), BB_SIZE)));
    s.emit_command(dummy_draw());
    assert!(matches!(
        s.clear_depth(f32::to_bits(0.5)),
        DepthClearOutcome::EmitQuad {
            has_color: true,
            ..
        }
    ));
    assert!(!s.current_pass_closed());
    assert_eq!(s.passes().len(), 1);
}

#[test]
fn rule_c_and_d_apply_per_attachment() {
    // Pass 0: backbuffer + rt_a (slot 1). Pass 1: rt_a alone as render
    // target 0 with a clear, then rt_b (slot 2) never used again.
    let rt_a = tex(0x3000);
    let rt_b = tex(0x3001);
    let mut s = fresh();
    s.set_extra_color_render_target(1, Some(slot(rt_a, BB_SIZE)));
    s.set_extra_color_render_target(2, Some(slot(rt_b, BB_SIZE)));
    s.emit_command(dummy_draw());
    s.set_extra_color_render_target(1, None);
    s.set_extra_color_render_target(2, None);
    s.set_color_render_target(rt_a, BB_SIZE.0, BB_SIZE.1, BB_FORMAT, RenderScale::IDENTITY);
    s.clear_color(1, 2, 3, 4);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    let first = &s.passes()[0];
    // rt_a's next use clears it: Rule C flips the slot-1 store.
    assert_eq!(first.extra_color()[0].store(), StoreAction::DontCare);
    // rt_b is never used again and is not the backbuffer: Rule D.
    assert_eq!(first.extra_color()[1].store(), StoreAction::DontCare);
    // The backbuffer keeps its store for Present.
    assert_eq!(first.color_store(), StoreAction::Store);
}

#[test]
fn rule_h_strips_every_target_when_nothing_writes_colour() {
    let mut s = fresh();
    s.set_extra_color_render_target(1, Some(slot(tex(0x3000), BB_SIZE)));
    s.note_draw_color_write_mask(0);
    s.emit_command(Command::set_render_pipeline_state(0x77));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let mut alt = FxHashMap::default();
    // SAFETY: tests; opaque value never dereferenced.
    alt.insert(0x77u64, unsafe { MetalHandle::new(0x78) });
    s.strip_color_from_no_color_draw_passes(&alt);
    let pass = &s.passes()[0];
    assert_eq!(pass.color_texture(), MetalHandle::NULL);
    assert!(!pass.extra_color()[0].is_bound(), "extras go with target 0");
    assert_eq!(pass.extra_present_mask(), 0);
}

#[test]
fn rule_e_merges_a_clear_only_pass_into_the_same_target_set() {
    // Clear the set, bind another target (a clear-only pass materialises),
    // come back to the set and draw: the clear folds into the draw pass.
    let rt_a = tex(0x3000);
    let mut s = fresh();
    s.set_extra_color_render_target(1, Some(slot(rt_a, BB_SIZE)));
    s.clear_color(1, 2, 3, 4);
    s.set_extra_color_render_target(1, None);
    s.set_color_render_target(
        tex(0x4000),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_extra_color_render_target(1, Some(slot(rt_a, BB_SIZE)));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 3);
    s.coalesce_clear_only_passes();
    assert_eq!(s.passes().len(), 2, "clear-only pass folded");
    let merged = &s.passes()[1];
    let clear = ColorLoad::Clear {
        r: 1,
        g: 2,
        b: 3,
        a: 4,
    };
    assert_eq!(merged.color_load(), clear);
    assert_eq!(merged.extra_color()[0].load(), clear);
}

#[test]
fn rule_e_refuses_a_different_target_set() {
    // The clear-only pass carries {backbuffer, rt_a}; the next pass on
    // the backbuffer carries {backbuffer} alone, so the set does not
    // match and the clear stays where it was.
    let rt_a = tex(0x3000);
    let mut s = fresh();
    s.set_extra_color_render_target(1, Some(slot(rt_a, BB_SIZE)));
    s.clear_color(1, 2, 3, 4);
    s.set_extra_color_render_target(1, None);
    s.set_color_render_target(
        tex(0x4000),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();
    assert_eq!(s.passes().len(), 3, "different set, no merge");
}

#[test]
fn rule_g_strips_only_the_dead_extra_and_rule_f_needs_every_store_dead() {
    let rt_a = tex(0x3000);
    let rt_b = tex(0x3001);
    let mut s = fresh();
    s.set_extra_color_render_target(1, Some(slot(rt_a, BB_SIZE)));
    s.set_extra_color_render_target(2, Some(slot(rt_b, BB_SIZE)));
    s.clear_color(1, 2, 3, 4);
    s.flush_pending_clears();
    // rt_a is read back later, so its store survives; rt_b's is dead.
    s.note_color_read_back(rt_a);
    s.finalize_store_actions(false);
    s.strip_dead_color_in_clear_only_passes();
    assert_eq!(s.passes().len(), 1);
    let pass = &s.passes()[0];
    assert!(
        pass.extra_color()[0].is_bound(),
        "read-back target keeps its store"
    );
    assert!(!pass.extra_color()[1].is_bound(), "dead extra stripped");
    s.cull_dead_clear_only_passes();
    assert_eq!(s.passes().len(), 1, "a live store keeps the pass");
}

#[test]
fn mid_frame_flush_keeps_every_colour_store() {
    // Two clear-only passes on two targets; the first is read back, which
    // flushes the frame. The second target is read back afterwards, so
    // its last-use store must survive the flush (Rule D off) and Rule F
    // must keep its pass.
    let rt_a = tex(0x3000);
    let rt_b = tex(0x3001);
    let mut s = fresh();
    s.set_color_render_target(rt_a, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.clear_color(1, 2, 3, 4);
    s.set_color_render_target(rt_b, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.clear_color(1, 2, 3, 4);
    s.flush_pending_clears();
    s.note_color_read_back(rt_a);
    s.finalize_store_actions(true);
    s.cull_dead_clear_only_passes();
    assert_eq!(
        s.passes().len(),
        2,
        "both clear-only passes survive a readback flush"
    );
    assert_eq!(s.passes()[1].color_store(), StoreAction::Store);
    // At a real frame end the unread target's store is elided as before.
    s.finalize_store_actions(false);
    assert_eq!(s.passes()[1].color_store(), StoreAction::DontCare);
}

// ── Blits in the read/write model ─────────────────────────────

fn copy_blit(src: MetalHandle<MTLTextureKind>, dst: MetalHandle<MTLTextureKind>) -> BlitCommand {
    BlitCommand::copy_texture_to_texture_full_mip(src.raw(), dst.raw(), 0, 64, 64)
}

#[test]
fn rule_d_keeps_store_when_a_stretch_rect_reads_the_target() {
    // Render into rt, then copy rt to the backbuffer after the pass: the
    // copy reads rt from device memory, so its last-use store must stay.
    let rt = tex(0x3000);
    let mut s = fresh();
    s.set_color_render_target(rt, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.push_pending_leading_blit(copy_blit(rt, backbuffer()));
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 1);
    assert_eq!(s.passes()[0].color_texture(), rt);
    assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
    // The blit is still queued for the trailing blit-only pass.
    assert_eq!(s.take_pending_leading_blits().len(), 1);
}

#[test]
fn rule_c_keeps_store_when_a_blit_reads_between_write_and_clear() {
    // rt written in pass 0, copied out by a blit that pass 1 carries, then
    // cleared in pass 2. The next-clear rule must not discard pass 0's
    // store: the copy reads it.
    let rt = tex(0x3000);
    let other = tex(0x5000);
    let mut s = fresh();
    s.set_color_render_target(rt, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.push_pending_leading_blit(copy_blit(rt, other));
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.clear_color(1, 2, 3, 4);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 3);
    assert_eq!(s.passes()[1].leading_blits().len(), 1);
    assert_eq!(
        s.passes()[2].color_load(),
        ColorLoad::Clear {
            r: 1,
            g: 2,
            b: 3,
            a: 4
        }
    );
    s.finalize_store_actions(false);
    assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
}

#[test]
fn rule_a_loads_a_target_written_by_a_blit_in_an_earlier_pass() {
    // A copy into rt_x is queued while rt_y is bound, so it lands in
    // rt_y's pass. rt_x's own first pass must still Load the copy.
    let rt_src = tex(0x3000);
    let rt_x = tex(0x4000);
    let rt_y = tex(0x5000);
    let mut s = fresh();
    s.set_color_render_target(rt_y, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.push_pending_leading_blit(copy_blit(rt_src, rt_x));
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt_x, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[0].color_texture(), rt_y);
    assert_eq!(s.passes()[0].leading_blits().len(), 1);
    assert_eq!(s.passes()[0].color_load(), ColorLoad::DontCare);
    assert_eq!(s.passes()[1].color_texture(), rt_x);
    assert_eq!(s.passes()[1].leading_blits().len(), 0);
    assert_eq!(s.passes()[1].color_load(), ColorLoad::Load);
}

#[test]
fn blit_written_set_resets_with_the_frame() {
    let rt_src = tex(0x3000);
    let rt_x = tex(0x4000);
    let mut s = fresh();
    s.push_pending_leading_blit(copy_blit(rt_src, rt_x));
    s.take_pending_leading_blits();
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        depth_texture: depth(),
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    s.set_color_render_target(rt_x, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes()[0].color_load(), ColorLoad::DontCare);
}

#[test]
fn rule_e_keeps_a_clear_only_pass_that_carries_leading_blits() {
    // A copy is queued, then Clear(rt_x) goes pending, and SetRT(rt_y)
    // materialises it as a clear-only pass that drains the copy. The later
    // Load pass on rt_x would be a merge target; merging would drop the
    // copy with the pass.
    let rt_x = tex(0x3000);
    let rt_y = tex(0x4000);
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.push_pending_leading_blit(dummy_blit());
    s.set_color_render_target(rt_x, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    assert_eq!(s.clear_color(1, 2, 3, 4), ColorClearOutcome::Folded);
    s.set_color_render_target(rt_y, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt_x, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 4);
    assert_eq!(s.passes()[1].color_texture(), rt_x);
    assert_eq!(s.passes()[1].leading_blits().len(), 1);
    assert_eq!(s.passes()[3].color_load(), ColorLoad::Load);
    s.coalesce_clear_only_passes();
    assert_eq!(
        s.passes().len(),
        4,
        "a pass with leading blits is never merged away"
    );
    assert_eq!(s.passes()[1].leading_blits().len(), 1);
    assert_eq!(s.passes()[3].color_load(), ColorLoad::Load);
}

#[test]
fn rule_e_aborts_when_an_intervening_blit_writes_the_target() {
    // Clear(rt_x) materialises as pass 0, a copy rt_y -> rt_x is queued
    // after pass 1, and pass 2 attaches rt_x with Load and carries that
    // copy. The clear is ordered before the copy, so it must not move
    // into pass 2's load action.
    let rt_x = tex(0x3000);
    let rt_y = tex(0x4000);
    let mut s = fresh();
    s.set_color_render_target(rt_x, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    assert_eq!(s.clear_color(1, 2, 3, 4), ColorClearOutcome::Folded);
    s.set_color_render_target(rt_y, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.push_pending_leading_blit(copy_blit(rt_y, rt_x));
    s.set_color_render_target(rt_x, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 3);
    assert_eq!(s.passes()[2].leading_blits().len(), 1);
    assert_eq!(s.passes()[2].color_load(), ColorLoad::Load);
    s.coalesce_clear_only_passes();
    assert_eq!(s.passes().len(), 3, "the clear stays ahead of the copy");
    assert_eq!(s.passes()[2].color_load(), ColorLoad::Load);
}

// ── B3: a mid-frame flush is not a frame end ─────────────────

#[test]
fn mid_frame_flush_keeps_the_depth_store() {
    // A depth-tested pass, then a mid-frame flush (a readback): the depth
    // surface may still be tested against in the continuation, so Rule B
    // must not discard its store at the flush. A real Present still elides
    // it (the TBDR depth-store optimisation).
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(true);
    assert_eq!(
        s.passes()[0].depth_store(),
        StoreAction::Store,
        "depth store survives a mid-frame flush",
    );
    s.finalize_store_actions(false);
    assert_eq!(
        s.passes()[0].depth_store(),
        StoreAction::DontCare,
        "depth store is still elided at a real Present",
    );
}

#[test]
fn continuation_loads_targets_drawn_before_the_flush() {
    // Draw to the backbuffer + depth, then a mid-frame flush. The
    // continuation's first pass on the same attachments must Load
    // (preserving the pre-flush pixels), not open first-use `DontCare`.
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        depth_texture: depth(),
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: true,
    });
    s.emit_command(dummy_draw());
    assert_eq!(
        s.passes()[0].color_load(),
        ColorLoad::Load,
        "continuation loads the backbuffer drawn before the flush",
    );
    assert_eq!(
        s.passes()[0].depth_load(),
        DepthLoad::Load,
        "continuation loads the depth surface too",
    );
}

#[test]
fn a_real_present_still_dontcares_first_use() {
    // The contrast to `continuation_loads_targets_drawn_before_the_flush`:
    // a real Present (continues_frame false) clears the seen sets, so the
    // next frame's first use of the backbuffer is `DontCare` again.
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        depth_texture: depth(),
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    s.emit_command(dummy_draw());
    assert_eq!(s.passes()[0].color_load(), ColorLoad::DontCare);
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::DontCare);
}

#[test]
fn a_clear_after_a_flush_folds_instead_of_a_scissored_quad() {
    // The frame continues past a mid-frame flush, so the backbuffer stays
    // "seen" for the load rules — but a Clear issued after the flush must
    // still fold to a full loadAction=Clear, not the cross-pass scissored
    // quad, because the pre-flush content is safely in VRAM. A sub-rect
    // viewport is set to expose the bug: the quad would clip to it.
    // (This is the `test_viewport` conformance shape: readback mid-frame,
    // then Clear under a viewport.)
    let mut s = fresh();
    s.set_viewport(0, 0, 640, 480, 0.0, 1.0);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        depth_texture: depth(),
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: true,
    });
    s.set_viewport(100, 100, 64, 64, 0.0, 1.0);
    assert_eq!(
        s.clear_color(1, 2, 3, 4),
        ColorClearOutcome::Folded,
        "a full clear after a flush folds, even though the target is still seen this frame",
    );
    // And the depth clear on the same sequence folds too, not a quad.
    assert_eq!(
        s.clear_depth(f32::to_bits(1.0)),
        DepthClearOutcome::Folded,
        "depth clear after a flush folds as well",
    );
}

#[test]
fn a_clear_within_one_segment_still_paints_a_quad() {
    // The contrast: within one submission segment (no flush), a second
    // clear of a target already drawn takes the cross-pass quad path so a
    // full loadAction=Clear cannot wipe the earlier content.
    let mut s = fresh();
    s.set_viewport(0, 0, 683, 683, 0.0, 1.0);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.set_viewport(683, 0, 683, 683, 0.0, 1.0);
    assert!(
        matches!(
            s.clear_color(1, 2, 3, 4),
            ColorClearOutcome::EmitQuad { .. }
        ),
        "a cross-pass clear inside one segment still scissors a quad",
    );
}
