//! Unit tests for the render-pass state machine and its load/store optimizer.
//!
//! `PassState` is driven through synthetic frames of opaque handles, so pass breaking and
//! every load/store rule run without a GPU: clears folded into load actions versus painted
//! as quads, the `DontCare` rules with their sampler, blit and mid-frame-flush guards,
//! render scale, multiple render targets, and the `last_bound` dedup cache. A rule that
//! fires one case too wide loses pixels, so each guard gets a case that fails without it.

use mtld3d_shared::{
    CommandType,
    mtl::{IndexType, PrimitiveType},
};

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
/// The back buffer's sRGB twin view, as the device supplies it every frame.
fn backbuffer_srgb() -> MetalHandle<MTLTextureKind> {
    tex(0x1001)
}

fn fresh() -> PassState {
    let mut s = PassState::new();
    reset_test_frame(&mut s);
    s
}

fn reset_test_frame(s: &mut PassState) {
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: BB_SIZE,
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
}

/// A frame rasterizing the back buffer at half the reported resolution.
fn fresh_scaled() -> PassState {
    let mut s = PassState::new();
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: (BB_SIZE.0 / 2, BB_SIZE.1 / 2),
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

fn sampler_binds(texture: u64) -> [(CommandType, Command); 2] {
    [
        (
            CommandType::SetFragmentTexture,
            Command::set_fragment_texture(texture, 0),
        ),
        (
            CommandType::SetVertexTexture,
            Command::set_vertex_texture(texture, 0),
        ),
    ]
}

fn every_draw_command() -> [(&'static str, Command); 3] {
    [
        (
            "non-indexed",
            Command::draw_primitives(PrimitiveType::Triangle, 0, 3),
        ),
        (
            "bound indices",
            Command::draw_indexed_primitives(
                PrimitiveType::Triangle,
                3,
                IndexType::UInt16,
                0xA000,
                0,
                0,
                1,
            ),
        ),
        (
            "inline or generated indices",
            Command::draw_indexed_primitives_up(
                PrimitiveType::Triangle,
                3,
                IndexType::UInt16,
                0xB000,
                6,
                1,
            ),
        ),
    ]
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
fn frame_sampled_textures_tracks_sampler_binds_in_stream_order() {
    for (stage, bind) in sampler_binds(0x7E10) {
        let mut s = fresh();
        let atlas = tex(0x7E10);
        // Not sampled before any draw emitted a bind: an upload landing
        // here must not rename because no earlier draw reads the old content.
        assert!(!s.texture_sampled_this_frame(atlas), "{stage:?}");
        s.emit_command(bind);
        assert!(s.texture_sampled_this_frame(atlas), "{stage:?}");
        // Unrelated handle stays unsampled (a renamed-fresh texture
        // relies on exactly this).
        assert!(!s.texture_sampled_this_frame(tex(0x7E20)), "{stage:?}");
    }
}

#[test]
fn frame_sampled_textures_clears_on_reset_frame() {
    for (stage, bind) in sampler_binds(0x7E10) {
        let mut s = fresh();
        let atlas = tex(0x7E10);
        s.emit_command(bind);
        assert!(s.texture_sampled_this_frame(atlas), "{stage:?}");
        s.reset_frame(&FrameReset {
            backbuffer: backbuffer(),
            backbuffer_srgb: backbuffer_srgb(),
            backbuffer_msaa: MetalHandle::NULL,
            backbuffer_msaa_srgb: MetalHandle::NULL,
            backbuffer_sample_count: 1,
            backbuffer_size: BB_SIZE,
            backbuffer_format: BB_FORMAT,
            backbuffer_contents: BackbufferContents::Undefined,
            depth_texture: depth(),
            depth_size: BB_SIZE,
            depth_has_stencil: false,
            render_scale: RenderScale::IDENTITY,
            continues_frame: false,
        });
        // Per-frame set resets: a next-frame upload before the first
        // sample goes to the live texture again.
        assert!(!s.texture_sampled_this_frame(atlas), "{stage:?}");
    }
}

#[test]
fn frame_sampled_textures_ignores_null_bind() {
    for (stage, bind) in sampler_binds(0) {
        let mut s = fresh();
        s.emit_command(bind);
        assert!(!s.texture_sampled_this_frame(tex(0)), "{stage:?}");
    }
}

#[test]
fn inline_slot0_bind_forces_next_bound_vertex_buffer_reemit() {
    let mut cache = LastBoundCache::new();
    // First bind of a real VB handle reports a change and caches it.
    assert_eq!(
        cache.vertex_buffer_changed(0, 0xDEAD, 0, 1),
        VertexBufferBind::Changed
    );
    // A redundant rebind of the same (handle, offset, generation) is skipped.
    assert_eq!(
        cache.vertex_buffer_changed(0, 0xDEAD, 0, 1),
        VertexBufferBind::Same
    );
    // An inline slot-0 bind (setVertexBytes) clobbers the Metal binding;
    // invalidating the cache must force the next bound draw to re-emit
    // even though it targets the same (handle, offset).
    cache.invalidate_vertex_buffer();
    assert_eq!(
        cache.vertex_buffer_changed(0, 0xDEAD, 0, 1),
        VertexBufferBind::Changed
    );
}

#[test]
fn reused_handle_over_another_generation_is_rebound() {
    let mut cache = LastBoundCache::new();
    assert_eq!(
        cache.vertex_buffer_changed(0, 0xDEAD, 0, 1),
        VertexBufferBind::Changed
    );
    // Same object address and offset, but a wrapper over a different
    // allocation: the bind goes out again and the cache moves on.
    assert_eq!(
        cache.vertex_buffer_changed(0, 0xDEAD, 0, 2),
        VertexBufferBind::ReusedHandle
    );
    assert_eq!(
        cache.vertex_buffer_changed(0, 0xDEAD, 0, 2),
        VertexBufferBind::Same
    );
}

#[test]
fn vertex_buffer_slots_are_tracked_independently() {
    let mut cache = LastBoundCache::new();
    assert_eq!(
        cache.vertex_buffer_changed(0, 0xDEAD, 0, 1),
        VertexBufferBind::Changed
    );
    assert_eq!(
        cache.vertex_buffer_changed(1, 0xBEEF, 16, 1),
        VertexBufferBind::Changed
    );
    // Slot 1's bind leaves slot 0's cache intact, and vice versa.
    assert_eq!(
        cache.vertex_buffer_changed(0, 0xDEAD, 0, 1),
        VertexBufferBind::Same
    );
    assert_eq!(
        cache.vertex_buffer_changed(1, 0xBEEF, 16, 1),
        VertexBufferBind::Same
    );
    // Invalidating slot 0 (inline UP bytes) does not touch slot 1.
    cache.invalidate_vertex_buffer();
    assert_eq!(
        cache.vertex_buffer_changed(0, 0xDEAD, 0, 1),
        VertexBufferBind::Changed
    );
    assert_eq!(
        cache.vertex_buffer_changed(1, 0xBEEF, 16, 1),
        VertexBufferBind::Same
    );
    // A null-stream inline bind at slot 1 forgets only slot 1.
    cache.invalidate_vertex_buffer_slot(1);
    assert_eq!(
        cache.vertex_buffer_changed(1, 0xBEEF, 16, 1),
        VertexBufferBind::Changed
    );
    assert_eq!(
        cache.vertex_buffer_changed(0, 0xDEAD, 0, 1),
        VertexBufferBind::Same
    );
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
    s.set_color_render_target_subresource(
        rt,
        &TargetExtent::whole(RenderScale::IDENTITY, (256, 256)),
        RT_FORMAT,
        (0, 0),
    );
    s.emit_command(dummy_draw());
    s.set_color_render_target_subresource(
        rt,
        &TargetExtent::whole(RenderScale::IDENTITY, (256, 256)),
        RT_FORMAT,
        (1, 0),
    );
    s.emit_command(dummy_draw());
    s.set_color_render_target_subresource(
        rt,
        &TargetExtent::whole(RenderScale::IDENTITY, (128, 128)),
        RT_FORMAT,
        (1, 1),
    );
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
    s.set_depth_stencil_attachment(other_depth, BB_SIZE, false, false);
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
fn first_use_colour_dontcare_is_the_back_buffer_alone() {
    let rt = tex(0x3000);
    // Rule A: the back buffer's first use in a frame, with no pending
    // clear, gets DontCare, since `Present` left it undefined. A game
    // render target keeps last frame's contents, so its first use loads.
    // Depth is shared so it's first-use in A and re-use (Load) in B.
    let mut s = fresh();
    s.emit_command(dummy_draw()); // pass A on backbuffer() + depth()
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw()); // pass B on rt + depth()
    assert_eq!(s.passes()[0].color_load(), ColorLoad::DontCare);
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::DontCare);
    assert_eq!(s.passes()[1].color_load(), ColorLoad::Load);
    // depth() is seen-already (pass A used it), so Load this time.
    assert_eq!(s.passes()[1].depth_load(), DepthLoad::Load);
}

/// A frame on the default surfaces whose back buffer starts with `contents`.
fn reset_frame_with_backbuffer_contents(s: &mut PassState, contents: BackbufferContents) {
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: contents,
        depth_texture: depth(),
        depth_size: BB_SIZE,
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
}

#[test]
fn first_use_colour_dontcare_needs_the_discard_swap_effect() {
    // Under COPY or FLIP the back buffer keeps its contents across
    // `Present`, so a game redrawing part of the frame without clearing
    // relies on the rest surviving: its first use loads. The depth plane
    // is unaffected.
    let mut s = PassState::new();
    reset_frame_with_backbuffer_contents(&mut s, BackbufferContents::Preserved);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes()[0].color_texture(), backbuffer());
    assert_eq!(s.passes()[0].color_load(), ColorLoad::Load);
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::DontCare);

    // The next frame under DISCARD takes the first-use `DontCare` again.
    reset_frame_with_backbuffer_contents(&mut s, BackbufferContents::Undefined);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes()[0].color_load(), ColorLoad::DontCare);
}

#[test]
fn only_the_discard_swap_effect_leaves_the_back_buffer_undefined() {
    assert!(matches!(
        BackbufferContents::from_swap_effect(mtld3d_types::D3DSWAPEFFECT_DISCARD),
        BackbufferContents::Undefined
    ));
    for swap_effect in [
        mtld3d_types::D3DSWAPEFFECT_FLIP,
        mtld3d_types::D3DSWAPEFFECT_COPY,
    ] {
        assert!(matches!(
            BackbufferContents::from_swap_effect(swap_effect),
            BackbufferContents::Preserved
        ));
    }
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
    s.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
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
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: BB_SIZE,
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
fn rebinding_the_same_target_with_a_new_format_breaks_the_pass() {
    const OTHER_FORMAT: PixelFormat = PixelFormat::Rgba16Float;
    let rt = tex(0x3000);
    // Handle and subresource unchanged, pixel format not. The open pass
    // froze the old format in its attachment descriptor, so it has to
    // close: otherwise the next draw builds a pipeline declaring the new
    // format against a pass attaching the old one.
    let mut s = fresh();
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 256, 256, OTHER_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());

    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[0].color_texture(), rt);
    assert_eq!(s.passes()[1].color_texture(), rt);
    assert_eq!(s.passes()[0].color_format(), RT_FORMAT);
    assert_eq!(s.passes()[1].color_format(), OTHER_FORMAT);
}

#[test]
fn rebinding_the_same_target_with_a_new_extent_breaks_the_pass() {
    let rt = tex(0x3000);
    // The extent is frozen at pass open too: it sizes the attachment and
    // decides Rule A's full-attachment-write predicate.
    let mut s = fresh();
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 128, 128, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());

    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[0].color_size(), (256, 256));
    assert_eq!(s.passes()[1].color_size(), (128, 128));
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
    c.vs_draw_changed(&[1, 2, 3, 4]);
    c.ps_draw_changed(&[5, 6, 7, 8]);
    c.ps_alpha_ref_changed(&[9, 10, 11, 12]);
    c.ps_fog_color_changed(&[13, 14, 15, 16]);
    c.vertex_buffer_changed(0, 0xEEEE, 32, 1);
    c.scissor_rect_changed((1, 2, 3, 4));
    c.blend_color_changed(0xFF11_2233);
    c.reset();
    assert!(c.fragment_sampler_changed(0, 0xAAAA));
    assert!(c.fragment_texture_changed(1, 0xBBBB));
    assert!(c.pipeline_changed(0xCCCC));
    assert!(c.depth_stencil_changed(0xDDDD));
    assert!(c.cull_mode_changed(CullMode::Back));
    assert!(c.vs_draw_changed(&[1, 2, 3, 4]));
    assert!(c.ps_draw_changed(&[5, 6, 7, 8]));
    assert!(c.ps_alpha_ref_changed(&[9, 10, 11, 12]));
    assert!(c.ps_fog_color_changed(&[13, 14, 15, 16]));
    assert_eq!(
        c.vertex_buffer_changed(0, 0xEEEE, 32, 1),
        VertexBufferBind::Changed
    );
    assert!(c.scissor_rect_changed((1, 2, 3, 4)));
    assert!(c.blend_color_changed(0xFF11_2233));
}

#[test]
fn last_bound_inline_bytes_dedup() {
    let mut c = LastBoundCache::new();
    assert!(c.ps_draw_changed(&[1, 2, 3, 4]));
    assert!(!c.ps_draw_changed(&[1, 2, 3, 4]));
    assert!(c.ps_draw_changed(&[1, 2, 3, 5]));
    assert!(c.ps_draw_changed(&[1, 2, 3])); // length change
    assert!(!c.ps_draw_changed(&[1, 2, 3]));
}

#[test]
fn last_bound_inline_bytes_slots_are_independent() {
    let mut c = LastBoundCache::new();
    c.vs_draw_changed(&[1; 16]);
    c.ps_draw_changed(&[2; 16]);
    c.ps_alpha_ref_changed(&[3; 4]);
    c.ps_fog_color_changed(&[4; 16]);
    // Identical content in a different slot must still report changed
    // (slot 13 hasn't seen this payload yet).
    assert!(!c.vs_draw_changed(&[1; 16]));
    assert!(!c.ps_draw_changed(&[2; 16]));
    assert!(!c.ps_alpha_ref_changed(&[3; 4]));
    assert!(!c.ps_fog_color_changed(&[4; 16]));
}

#[test]
fn last_bound_inline_bytes_reset_keeps_capacity() {
    let mut c = LastBoundCache::new();
    c.ps_draw_changed(&[0xAB; 256]);
    let cap_before = c.ps_draw.capacity();
    c.reset();
    assert_eq!(c.ps_draw.len(), 0);
    assert_eq!(c.ps_draw.capacity(), cap_before);
}

#[test]
fn last_bound_vertex_buffer_dedup() {
    let mut c = LastBoundCache::new();
    assert_eq!(
        c.vertex_buffer_changed(0, 0xAAAA, 0, 1),
        VertexBufferBind::Changed
    );
    assert_eq!(
        c.vertex_buffer_changed(0, 0xAAAA, 0, 1),
        VertexBufferBind::Same
    );
    // Same handle, different offset → changed.
    assert_eq!(
        c.vertex_buffer_changed(0, 0xAAAA, 64, 1),
        VertexBufferBind::Changed
    );
    // Same offset, different handle → changed.
    assert_eq!(
        c.vertex_buffer_changed(0, 0xBBBB, 64, 1),
        VertexBufferBind::Changed
    );
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
    // A fresh encoder blends with zero, so the D3D9 default opaque white is
    // a change on a pass's first draw and zero is not.
    assert!(!c.blend_color_changed(FRESH_BLEND_COLOR));
    assert!(c.blend_color_changed(0xFFFF_FFFF));
    assert!(!c.blend_color_changed(0xFFFF_FFFF));
    assert!(c.blend_color_changed(0xFF80_2040));
    assert!(!c.blend_color_changed(0xFF80_2040));
    c.reset();
    assert!(c.blend_color_changed(0xFFFF_FFFF), "reset returns to zero");
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
    assert!(cache.triangle_fill_mode_changed(TriangleFillMode::Lines));
    shadow.record(&Command::set_triangle_fill_mode(TriangleFillMode::Lines));
    assert!(cache.fragment_texture_changed(3, 0x7E10));
    shadow.record(&Command::set_fragment_texture(0x7E10, 3));
    assert!(cache.fragment_sampler_changed(3, 0x5A77));
    shadow.record(&Command::set_fragment_sampler_state(0x5A77, 3));
    assert_eq!(
        cache.vertex_buffer_changed(0, 0xBEEF, 0x40, 1),
        VertexBufferBind::Changed
    );
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

    assert_eq!(
        cache.vertex_buffer_changed(0, 0xBEEF, 0x10, 1),
        VertexBufferBind::Changed
    );
    shadow.record(&Command::set_vertex_buffer(0xBEEF, 0x10, 0));
    cache.debug_assert_in_sync(&shadow);

    // Inline slot-0 bind, then invalidate — the encoder's emit order.
    shadow.record(&Command::set_vertex_bytes_at(0xA000, 4, 0));
    cache.invalidate_vertex_buffer();
    cache.debug_assert_in_sync(&shadow);

    // The next bound draw re-binds the same buffer and stays in sync.
    assert_eq!(
        cache.vertex_buffer_changed(0, 0xBEEF, 0x10, 1),
        VertexBufferBind::Changed
    );
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

    assert_eq!(
        cache.vertex_buffer_changed(0, 0xBEEF, 0x10, 1),
        VertexBufferBind::Changed
    );
    shadow.record(&Command::set_vertex_buffer(0xBEEF, 0x10, 0));
    assert_eq!(
        cache.vertex_buffer_changed(1, 0xCAFE, 0x20, 1),
        VertexBufferBind::Changed
    );
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
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: BB_SIZE,
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
fn rule_a_depth_wider_than_the_rt_loads_on_first_use() {
    // D3D9 only requires the depth-stencil surface to be at least as large as
    // the render target, so a larger one is legal. Under a viewport that covers
    // render target 0 exactly, the pass cannot write the depth surface outside
    // that area, so its first use in the frame must Load rather than discard
    // what the surface holds there. The stencil plane rides the same texture.
    let ds = tex(0x3300);
    let mut s = fresh();
    s.set_depth_stencil_attachment(ds, (1024, 1024), false, true);
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 1);
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::Load);
    assert_eq!(s.passes()[0].stencil_load(), StencilLoad::Load);
    // The back buffer is judged against its own extent and keeps the discard.
    assert_eq!(s.passes()[0].color_load(), ColorLoad::DontCare);
}

#[test]
fn rule_a_depth_matching_the_rt_still_discards_on_first_use() {
    // The counterpart to the oversized case: a depth surface the viewport does
    // cover keeps Rule A's first-use discard, so measuring the depth plane
    // against its own extent costs nothing in the common shape.
    let rt = tex(0x3000);
    let ds = tex(0x3300);
    let mut s = fresh();
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.set_depth_stencil_attachment(ds, (256, 256), false, true);
    s.set_viewport(0, 0, 256, 256, 0.0, 1.0);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 1);
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::DontCare);
    assert_eq!(s.passes()[0].stencil_load(), StencilLoad::DontCare);
}

#[test]
fn rule_a_first_use_stencil_is_dontcare_and_later_use_loads() {
    // The stencil plane lives in the depth texture, so it takes the
    // first-use DontCare under the depth predicate. A second pass on the
    // same texture in the frame loads: games carry stencil across passes.
    let ds = tex(0x3300);
    let rt = tex(0x3000);
    let mut s = fresh();
    s.set_depth_stencil_attachment(ds, BB_SIZE, false, true);
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
    s.set_depth_stencil_attachment(ds, BB_SIZE, false, true);
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
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: BB_SIZE,
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
    s.set_depth_stencil_attachment(ds, BB_SIZE, false, true);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes()[0].stencil_load(), StencilLoad::DontCare);
    s.set_depth_stencil_attachment(depth(), BB_SIZE, false, false);
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
    s.set_depth_stencil_attachment(d2, BB_SIZE, false, false);
    s.emit_command(dummy_draw());
    // Pass 2: backbuffer() + d1
    s.set_depth_stencil_attachment(d1, BB_SIZE, false, false);
    s.emit_command(dummy_draw());
    // Pass 3: backbuffer() + d2
    s.set_depth_stencil_attachment(d2, BB_SIZE, false, false);
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
    // not fire for any pass here, and pass 1's rt keeps its store at its
    // last use: a later frame may read it.
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
    assert_eq!(s.passes()[1].color_store(), StoreAction::Store);
    assert_eq!(s.passes()[2].color_store(), StoreAction::Store);
}

#[test]
fn rule_c_next_pass_clears_same_rt_flips_store() {
    let rt = tex(0x3000);
    // Pass 0 backbuffer(), pass 1 rt with clear, pass 2 backbuffer() with
    // clear. backbuffer() pass 0's next use is pass 2 (clear) → Rule C
    // flips. rt pass 1 has no next use → Rule C keeps Store, and it
    // stays: a later frame may read it. backbuffer() pass 2 is the last
    // pass for backbuffer() and keeps Store (Present reads it).
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
    assert_eq!(s.passes()[1].color_store(), StoreAction::Store);
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
    // Pass 2 rt_b → no next rt_b use → keeps Store for a later frame.
    assert_eq!(s.passes()[2].color_store(), StoreAction::Store);
    // Pass 3 rt_a → no next rt_a use → keeps Store for a later frame.
    assert_eq!(s.passes()[3].color_store(), StoreAction::Store);
    // Pass 4 backbuffer() → last in frame, keeps Store (Present reads it).
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
    s.set_depth_stencil_attachment(cascade_depth, BB_SIZE, false, false);
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
    s.set_depth_stencil_attachment(depth(), BB_SIZE, false, false);
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
fn rule_a_colour_target_sampled_later_loads_on_first_use() {
    let rt = tex(0x4000);
    // Pass 1 first-attaches a colour rt, pass 2 samples that same rt as a
    // fragment texture. A game target keeps its contents across frames, so
    // Rule A never discards its load, and finalize has nothing to revert.
    let mut s = fresh();
    // Bounce off backbuffer() first so the next pass-open is first-use of rt.
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes()[1].color_load(), ColorLoad::Load);
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
    s.set_depth_stencil_attachment(cascade_d0, BB_SIZE, false, false);
    s.clear_color(1, 2, 3, 4);
    s.clear_depth(f32::to_bits(1.0));
    // Pass 1: same cascade_color but different depth. cascade_d0
    // is sampled in the scene pass later.
    s.set_depth_stencil_attachment(cascade_d1, BB_SIZE, false, false);
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
    s.set_depth_stencil_attachment(depth(), BB_SIZE, false, false);
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
    let other_depth = tex(0x9100);
    // Pass 0: cascade_color (Clear) + cascade_depth (Clear), no
    // draws. cascade_depth is NEVER sampled this frame, so Rule B
    // flips depth Store=DontCare. The next pass on cascade_color
    // begins with a Clear, so Rule C flips color Store=DontCare.
    // Both Stores DontCare + no draws + no blits → Rule F culls.
    let mut s = fresh();
    s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT, RenderScale::IDENTITY);
    s.set_depth_stencil_attachment(cascade_depth, BB_SIZE, false, false);
    s.clear_color(1, 2, 3, 4);
    s.clear_depth(f32::to_bits(1.0));
    // No draws, no blits — pure clear-only pass. The next pass clears
    // cascade_color again under a different depth surface and draws.
    s.set_depth_stencil_attachment(other_depth, BB_SIZE, false, false);
    s.clear_color(5, 6, 7, 8);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_depth_stencil_attachment(depth(), BB_SIZE, false, false);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    s.cull_dead_clear_only_passes();
    // The cascade clear-only pass should be gone; the redraw of
    // cascade_color and the BB scene pass remain.
    assert!(
        s.passes()
            .iter()
            .all(|p| p.depth_texture() != cascade_depth),
        "the dead clear-only pass is culled",
    );
    assert_eq!(s.passes().len(), 2);
}

#[test]
fn rule_f_keeps_a_last_use_clear_only_colour_pass() {
    let cascade_color = tex(0x3000);
    let cascade_depth = tex(0x9000);
    // A clear-only pass whose colour target is not touched again this
    // frame. Its depth is never sampled, so Rule B discards the depth
    // store, but the cleared colour is the target's content for a later
    // frame, which may sample it. The colour store stays and Rule F
    // must keep the pass.
    let mut s = fresh();
    s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT, RenderScale::IDENTITY);
    s.set_depth_stencil_attachment(cascade_depth, BB_SIZE, false, false);
    s.clear_color(1, 2, 3, 4);
    s.clear_depth(f32::to_bits(1.0));
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_depth_stencil_attachment(depth(), BB_SIZE, false, false);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    s.cull_dead_clear_only_passes();
    let kept = s
        .passes()
        .iter()
        .find(|p| p.color_texture() == cascade_color)
        .expect("the clear-only colour pass is kept");
    assert_eq!(kept.color_store(), StoreAction::Store);
    assert_eq!(kept.depth_store(), StoreAction::DontCare);
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
    s.set_depth_stencil_attachment(cascade_depth, BB_SIZE, false, false);
    s.clear_color(1, 2, 3, 4);
    s.clear_depth(f32::to_bits(1.0));
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_depth_stencil_attachment(depth(), BB_SIZE, false, false);
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
fn every_draw_command_survives_the_complete_pass_rule_sequence() {
    for (name, draw) in every_draw_command() {
        let target = tex(0x4000);
        let mut s = fresh();
        s.set_color_render_target(
            target,
            BB_SIZE.0,
            BB_SIZE.1,
            BB_FORMAT,
            RenderScale::IDENTITY,
        );
        s.clear_color(1, 2, 3, 4);
        s.note_draw_color_write_mask(0xF);
        s.emit_command(draw);
        s.set_color_render_target(
            tex(0x5000),
            BB_SIZE.0,
            BB_SIZE.1,
            BB_FORMAT,
            RenderScale::IDENTITY,
        );
        s.note_draw_color_write_mask(0xF);
        s.emit_command(dummy_draw());
        s.set_color_render_target(
            target,
            BB_SIZE.0,
            BB_SIZE.1,
            BB_FORMAT,
            RenderScale::IDENTITY,
        );
        s.note_draw_color_write_mask(0xF);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");

        assert_eq!(s.passes().len(), 3, "{name}: before pass rules");
        s.coalesce_clear_only_passes();
        assert_eq!(s.passes().len(), 3, "{name}: Rule E keeps the draw pass");
        s.finalize_load_actions();
        s.finalize_store_actions(false);
        s.strip_dead_color_in_clear_only_passes();
        s.strip_color_from_no_color_draw_passes(&FxHashMap::default());
        s.cull_dead_clear_only_passes();

        assert_eq!(
            s.passes().len(),
            3,
            "{name}: Rules F through H keep the pass"
        );
        assert!(
            matches!(
                s.passes()[0].color_load(),
                ColorLoad::Clear {
                    r: 1,
                    g: 2,
                    b: 3,
                    a: 4
                }
            ),
            "{name}: the clear remains ordered before the draw",
        );
        assert!(
            s.passes()[0]
                .commands()
                .iter()
                .any(|command| command.cmd == draw.cmd),
            "{name}: the draw command remains in the first pass",
        );
        assert_eq!(
            s.passes()[2].color_load(),
            ColorLoad::Load,
            "{name}: the later draw loads the first pass's contribution",
        );
    }
}

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
    s.set_depth_stencil_attachment(ds, BB_SIZE, false, true);

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
    s.set_depth_stencil_attachment(ds, BB_SIZE, false, true);
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
fn a_zero_area_viewport_covers_neither_attachment() {
    // The same degenerate viewport between passes. It is a strict sub-region
    // of both attachments, so a whole-target clear under it takes the region
    // path, whose clip leaves nothing to paint, rather than folding a
    // full-attachment clear.
    let ds = tex(0x3300);
    let mut s = fresh_scaled();
    s.set_depth_stencil_attachment(ds, BB_SIZE, false, true);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.set_viewport(1, 1, 1, 1, 0.0, 1.0);
    assert_eq!(s.effective_viewport(), (1, 1, 0, 0));

    assert!(!s.viewport_covers_color_attachment());
    assert!(!s.viewport_covers_depth_attachment());
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
fn a_depth_clear_with_no_depth_attachment_is_a_no_op() {
    let mut s = fresh();
    s.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);

    assert_eq!(s.clear_depth(f32::to_bits(1.0)), DepthClearOutcome::NoOp);
    assert!(s.pending_depth_clear.is_none());
    assert!(s.passes().is_empty());
}

#[test]
fn a_stencil_clear_with_no_depth_attachment_is_a_no_op() {
    // Nothing is attached, so there is nothing to fold or paint; stashing
    // would clear whatever texture the next pass happens to attach.
    let mut s = fresh();
    s.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);

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
    s.set_depth_stencil_attachment(ds, BB_SIZE, false, true);
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
fn depth_and_stencil_clears_under_a_counting_query_end_the_pass_and_fold() {
    // With a visibility query armed the depth chain ends the pass rather than
    // paint a quad the query would count; the covering clear then waits for
    // the next pass, and the stencil chain, finding no pass open, waits with
    // it. Both planes open that pass with a Clear load and no quad at all.
    let ds = tex(0x3300);
    let mut s = fresh();
    s.set_depth_stencil_attachment(ds, BB_SIZE, false, true);
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    s.emit_command(dummy_draw());
    s.emit_command(Command::set_visibility_result_mode(
        mtld3d_shared::mtl::VisibilityResultMode::Counting,
        0,
    ));

    let z = f32::to_bits(1.0);
    assert_eq!(s.clear_depth(z), DepthClearOutcome::Folded);
    assert_eq!(s.clear_stencil(1), StencilClearOutcome::Folded);
    assert!(s.current_pass_closed(), "the counted pass ended first");
    assert_eq!(s.passes().len(), 1);

    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[1].depth_load(), DepthLoad::Clear { value: z });
    assert_eq!(
        s.passes()[1].stencil_load(),
        StencilLoad::Clear { value: 1 }
    );
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
    s.set_depth_stencil_attachment(ds, BB_SIZE, false, true);
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

/// A depth clear on one mip level never folds into a pass on another level.
#[test]
fn rule_e_keeps_a_depth_clear_off_another_level_of_the_same_texture() {
    let ds = tex(0x3300);
    let half = (BB_SIZE.0 / 2, BB_SIZE.1 / 2);
    let mut s = fresh();
    s.set_depth_stencil_attachment_level(ds, 0, BB_SIZE, false, true);
    s.clear_depth(f32::to_bits(0.5));
    s.clear_stencil(0x2A);
    // Rebinding to level 1 materialises the clear-only pass on level 0.
    s.set_depth_stencil_attachment_level(ds, 1, half, false, true);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();

    let passes = s.passes();
    assert_eq!(passes.len(), 2, "the level 0 clear-only pass stands");
    assert_eq!(passes[0].depth_level(), 0);
    assert_eq!(
        passes[0].depth_load(),
        DepthLoad::Clear {
            value: f32::to_bits(0.5)
        }
    );
    assert_eq!(passes[0].stencil_load(), StencilLoad::Clear { value: 0x2A });
    assert_eq!(passes[1].depth_level(), 1);
    assert_eq!(passes[1].depth_load(), DepthLoad::Load);
    assert_eq!(passes[1].stencil_load(), StencilLoad::Load);
}

/// A pass on another mip level is not a consumer, so the clear folds past it.
#[test]
fn rule_e_folds_a_depth_clear_past_another_level_into_its_own_level() {
    let ds = tex(0x3300);
    let half = (BB_SIZE.0 / 2, BB_SIZE.1 / 2);
    let mut s = fresh();
    s.set_depth_stencil_attachment_level(ds, 0, BB_SIZE, false, false);
    s.clear_depth(f32::to_bits(0.5));
    s.set_depth_stencil_attachment_level(ds, 1, half, false, false);
    s.emit_command(dummy_draw());
    s.set_depth_stencil_attachment_level(ds, 0, BB_SIZE, false, false);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();

    let passes = s.passes();
    assert_eq!(passes.len(), 2, "the clear-only pass folds away");
    assert_eq!(passes[0].depth_level(), 1);
    assert_eq!(passes[0].depth_load(), DepthLoad::Load);
    assert_eq!(passes[1].depth_level(), 0);
    assert_eq!(
        passes[1].depth_load(),
        DepthLoad::Clear {
            value: f32::to_bits(0.5)
        }
    );
}

#[test]
fn rule_e_aborts_when_intervening_pass_samples_target() {
    for (stage, bind) in sampler_binds(0x4000) {
        let rt = tex(0x4000);
        // If something between the clear-only pass and the candidate
        // merge target samples the texture, moving the Clear past it
        // would change the read; the merge must be rejected.
        let mut s = fresh();
        // Pass 0: clear-only on rt.
        s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
        s.clear_color(1, 2, 3, 4);
        // Force the pending clear to materialise by hopping rt
        // (combined flush).
        s.set_color_render_target(tex(0x5000), 256, 256, RT_FORMAT, RenderScale::IDENTITY);
        s.emit_command(bind);
        s.emit_command(dummy_draw());
        // Re-attach rt and draw. Without the read at 0x5000 this would
        // be a valid merge target, but the intervening sample disables it.
        s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        let before = s.passes().len();
        s.coalesce_clear_only_passes();
        // Coalesce must not delete the clear-only pass.
        assert_eq!(s.passes().len(), before, "{stage:?}");
        // rt's clear-only pass is still there with its Clear load action.
        let cleared = s
            .passes()
            .iter()
            .find(|p| p.color_texture() == rt && matches!(p.color_load(), ColorLoad::Clear { .. }))
            .expect("clear-only rt pass must remain");
        let cmds = cleared.commands();
        let has_draw = cmds.iter().any(Command::is_draw);
        assert!(!has_draw, "{stage:?}");
    }
}

#[test]
fn rule_e_aborts_when_intervening_pass_samples_target_through_srgb_twin() {
    for (stage, bind) in sampler_binds(0x4001) {
        let rt = tex(0x4000);
        let twin = tex(0x4001);
        let mut s = fresh();
        s.register_srgb_twin(twin, rt);
        s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
        s.clear_color(1, 2, 3, 4);
        s.set_color_render_target(tex(0x5000), 256, 256, RT_FORMAT, RenderScale::IDENTITY);
        s.emit_command(bind);
        s.emit_command(dummy_draw());
        s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        let before = s.passes().len();
        s.coalesce_clear_only_passes();
        assert_eq!(
            s.passes().len(),
            before,
            "{stage:?}: a sampler bind through the sRGB twin reads the base target"
        );
    }
}

#[test]
fn rule_e_aborts_when_only_an_intervening_leading_blit_reads_the_target() {
    // A blit source is no sampler bind, so the target is unmarked for this
    // submission; the command scan may be skipped but the leading blits of
    // the intervening pass must still be read.
    let rt = tex(0x4000);
    let mut s = fresh();
    s.set_color_render_target(rt, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.clear_color(1, 2, 3, 4);
    s.set_color_render_target(tex(0x5000), 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.push_pending_leading_blit(copy_blit(rt, tex(0x6000)));
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert!(!s.texture_sampled_this_frame(rt));
    let before = s.passes().len();
    s.coalesce_clear_only_passes();
    assert_eq!(
        s.passes().len(),
        before,
        "the copy reads the cleared target"
    );
    assert!(matches!(
        s.passes()[0].color_load(),
        ColorLoad::Clear { .. }
    ));
}

/// The pass scans skip a target no sampler bind marked in this submission.
///
/// A bind that bypassed `emit_command` breaks the premise, and the debug
/// build checks the skipped scan against the full one, so reaching that
/// check at all shows the scan was skipped.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "which no bind this submission marked")]
fn pass_scan_skips_a_target_no_bind_marked() {
    let rt = tex(0x4000);
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.passes[0]
        .commands
        .push(Command::set_fragment_texture(rt.raw(), 0));
    pass_samples_texture(
        &s.passes[0],
        rt,
        &s.texture_view_to_base,
        &s.frame_sampled_textures,
    );
}

#[test]
fn unsampled_colour_target_keeps_its_contents_into_the_next_frame() {
    let portrait = tex(0x3000);
    // A render target the frame clears and draws into but never samples:
    // the next frame is the first to read it. D3D9 keeps render-target
    // contents across `Present`, so its last use stores, and the next
    // frame's first uncleared full-viewport pass on it loads.
    let mut s = fresh();
    s.set_color_render_target(portrait, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.set_viewport(0, 0, 64, 64, 0.0, 1.0);
    s.clear_color(0, 0, 0, 0);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[0].color_texture(), portrait);
    assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
    assert_eq!(s.passes()[1].color_texture(), backbuffer());
    assert_eq!(s.passes()[1].color_store(), StoreAction::Store);

    reset_test_frame(&mut s);
    s.set_color_render_target(portrait, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.set_viewport(0, 0, 64, 64, 0.0, 1.0);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_load_actions();
    let next = s
        .passes()
        .iter()
        .find(|p| p.color_texture() == portrait)
        .expect("the next frame's pass on the target");
    assert_eq!(next.color_load(), ColorLoad::Load);
}

#[test]
fn colour_target_sampled_later_keeps_store() {
    for (stage, bind) in sampler_binds(0x4000) {
        let rt = tex(0x4000);
        // A non-backbuffer color rt that is sampled by a later pass
        // must preserve its content.
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
        s.emit_command(bind);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_store_actions(false);
        assert_eq!(s.passes().len(), 2, "{stage:?}");
        assert_eq!(s.passes()[0].color_texture(), rt, "{stage:?}");
        assert_eq!(s.passes()[0].color_store(), StoreAction::Store, "{stage:?}");
    }
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
    s.set_depth_stencil_attachment(cascade_depth, BB_SIZE, false, false);
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
    s.set_depth_stencil_attachment(d2, BB_SIZE, false, false);
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
fn rule_h_recognizes_every_draw_command() {
    for (name, draw) in every_draw_command() {
        let mut s = fresh();
        s.note_draw_color_write_mask(0);
        s.emit_command(set_pso(PSO_WITH));
        s.emit_command(draw);
        s.end_current_pass("test");
        let mut alt = FxHashMap::default();
        alt.insert(PSO_WITH, pso(PSO_NO_COLOR));

        s.strip_color_from_no_color_draw_passes(&alt);

        assert_eq!(
            s.passes()[0].color_texture(),
            MetalHandle::NULL,
            "{name}: Rule H strips dead colour from a pass with draws",
        );
    }
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
    s.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
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
    // A depth surface the size of the target, as a caster pass binds, so Rule H
    // may strip the colour without widening the pass.
    s.set_depth_stencil_attachment(tex(0x2100), (256, 256), false, false);
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
fn rule_h_keeps_color_clear_quad_whose_store_survives_the_submission() {
    const PSO_CLEAR_QUAD_COLOR: u64 = 0xCAFE_BABE;
    // An offscreen target is cleared through a colour clear-quad (the
    // cross-pass shape: `Clear` on a closed pass of a target drawn earlier)
    // in a pass whose other draws are all zero-mask. Nothing later in this
    // submission reads the target, but D3D9 keeps its contents: a sampler, a
    // `StretchRect` or a readback may read it after a mid-frame flush or in a
    // later frame, and must see the clear. The colour store survives
    // finalisation, so Rule H leaves the pass alone.
    for frame_continues in [false, true] {
        let mut s = fresh();
        let target = tex(0x3000);
        s.set_color_render_target(target, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
        let start = s.open_color_clear_quad_block();
        s.emit_command(set_pso(PSO_CLEAR_QUAD_COLOR));
        s.emit_command(dummy_draw());
        s.close_color_clear_quad_block(start);
        for _ in 0..2 {
            s.note_draw_color_write_mask(0);
            s.emit_command(set_pso(PSO_WITH));
            s.emit_command(dummy_draw());
        }
        s.end_current_pass("test");
        let mut alt = FxHashMap::default();
        alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
        s.finalize_load_actions();
        s.finalize_store_actions(frame_continues);
        s.strip_dead_color_in_clear_only_passes();
        s.strip_color_from_no_color_draw_passes(&alt);
        s.cull_dead_clear_only_passes();

        let pass = &s.passes()[0];
        assert_eq!(
            pass.color_texture(),
            target,
            "continues={frame_continues}: colour kept"
        );
        assert_eq!(pass.color_store(), StoreAction::Store);
        assert_eq!(
            pass.color_clear_quad_ranges().len(),
            1,
            "continues={frame_continues}: clear-quad kept"
        );
        let pso_handles: Vec<u64> = pass
            .commands()
            .iter()
            .filter(|c| c.cmd == CommandType::SetRenderPipelineState as u32)
            .map(|c| c.param_b)
            .collect();
        assert_eq!(
            pso_handles,
            [PSO_CLEAR_QUAD_COLOR, PSO_WITH, PSO_WITH],
            "continues={frame_continues}: no pipeline rewritten"
        );
    }
}

#[test]
fn rule_h_strips_color_and_clear_quad_when_the_next_pass_clears_the_target() {
    const PSO_CLEAR_QUAD_COLOR: u64 = 0xCAFE_BABE;
    // Cascade caster pass shape: a mid-pass `Clear` on the cascade colour
    // atlas became a colour clear-quad, and the rest of the pass is
    // zero-mask caster draws. The next pass on the atlas opens with a full
    // `Clear`, so Rule C discards this pass's colour store: the clear-quad
    // is dead work. Rule H strips the colour attachment AND drains the
    // clear-quad's commands so the depth-only descriptor doesn't bind a
    // colour-output clear-quad pipeline.
    let mut s = fresh();
    let atlas = tex(0x3000);
    s.set_color_render_target(atlas, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    // A depth surface the size of the target, as a caster pass binds, so Rule H
    // may strip the colour without widening the pass.
    s.set_depth_stencil_attachment(tex(0x2100), (256, 256), false, false);
    // Color clear-quad block: none of its commands should tag
    // `color_writes_observed`.
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
    s.clear_color(0, 0, 0, 0);
    s.note_draw_color_write_mask(0xF);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    // Only the real caster needs a side-map entry; clear-quad
    // PSOs are removed wholesale and don't need to resolve.
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    assert_eq!(s.passes()[1].color_texture(), atlas);
    assert!(matches!(
        s.passes()[1].color_load(),
        ColorLoad::Clear { .. }
    ));
    assert_eq!(s.passes()[0].color_store(), StoreAction::DontCare);
    s.strip_dead_color_in_clear_only_passes();
    s.strip_color_from_no_color_draw_passes(&alt);
    s.cull_dead_clear_only_passes();

    assert_eq!(s.passes()[1].color_texture(), atlas, "clearing pass kept");
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

/// Run every submit-time pass rule in the order `apply_pass_rules` does.
fn apply_pass_rules_with(
    s: &mut PassState,
    alt: &FxHashMap<u64, MetalHandle<MTLRenderPipelineStateKind>>,
) {
    s.drop_overwritten_clear_only_passes();
    s.coalesce_clear_only_passes();
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    s.strip_dead_color_in_clear_only_passes();
    s.strip_color_from_no_color_draw_passes(alt);
    s.cull_dead_clear_only_passes();
    s.merge_adjacent_identical_passes();
}

/// The colour clear value the cascade placeholder tests clear to: 1.0 in every channel.
const PLACEHOLDER_CLEAR: u32 = 0x3F80_0000;

/// Record the cascade caster passes of `WoW` 3.3.5a, one pass per cascade depth texture.
///
/// Every pass binds the same colour placeholder, clears it and its own
/// depth texture in full, and draws casters with colour writes masked off.
/// Each cascade is a sampleable shadow map, so Rule B keeps its depth.
fn record_cascade_casters(
    s: &mut PassState,
    placeholder: MetalHandle<MTLTextureKind>,
    cascades: &[MetalHandle<MTLTextureKind>],
) {
    for &cascade in cascades {
        s.set_color_render_target(placeholder, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
        s.set_depth_stencil_attachment(cascade, (256, 256), true, false);
        s.set_viewport(0, 0, 256, 256, 0.0, 1.0);
        let v = PLACEHOLDER_CLEAR;
        s.clear_color(v, v, v, v);
        s.clear_depth(f32::to_bits(1.0));
        for _ in 0..2 {
            s.note_draw_color_write_mask(0);
            s.emit_command(set_pso(PSO_WITH));
            s.emit_command(dummy_draw());
        }
    }
    s.end_current_pass("test");
}

fn pipelines_of(pass: &Pass) -> Vec<u64> {
    pass.commands()
        .iter()
        .filter(|c| c.cmd == CommandType::SetRenderPipelineState as u32)
        .map(|c| c.param_b)
        .collect()
}

#[test]
fn rule_h_strips_a_cleared_placeholder_from_every_caster_pass_but_the_last() {
    // Each caster pass opens by clearing the shared placeholder, so Rule C
    // discards the colour store of all but the last, and those Clears write
    // nothing anyone observes. The last pass's Clear is stored: it is what
    // the placeholder holds after the frame, so that pass keeps its colour.
    let placeholder = tex(0x3000);
    let cascades = [tex(0x4000), tex(0x4001), tex(0x4002), tex(0x4003)];
    let mut s = fresh();
    record_cascade_casters(&mut s, placeholder, &cascades);
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    apply_pass_rules_with(&mut s, &alt);

    let passes = s.passes();
    assert_eq!(passes.len(), cascades.len(), "every caster pass survives");
    for (pass, cascade) in passes.iter().zip(cascades) {
        assert_eq!(pass.depth_texture(), cascade);
        assert_eq!(pass.depth_store(), StoreAction::Store, "shadow map kept");
    }
    let (last, stripped) = passes.split_last().expect("caster passes");
    for (i, pass) in stripped.iter().enumerate() {
        assert_eq!(
            pass.color_texture(),
            MetalHandle::NULL,
            "pass {i}: colour stripped"
        );
        assert_eq!(pass.color_load(), ColorLoad::DontCare, "pass {i}");
        assert!(
            pipelines_of(pass).iter().all(|&h| h == PSO_NO_COLOR),
            "pass {i}: every caster binds the no-colour pipeline"
        );
    }
    let v = PLACEHOLDER_CLEAR;
    assert_eq!(
        last.color_texture(),
        placeholder,
        "last caster pass keeps colour"
    );
    assert_eq!(
        last.color_load(),
        ColorLoad::Clear {
            r: v,
            g: v,
            b: v,
            a: v
        }
    );
    assert_eq!(last.color_store(), StoreAction::Store);
    assert!(pipelines_of(last).iter().all(|&h| h == PSO_WITH));
}

#[test]
fn rule_h_keeps_every_cleared_placeholder_a_sampler_reads() {
    // A sampled placeholder keeps every colour store (Rule C stands down), so
    // every caster pass's Clear is stored and every pass keeps its colour.
    let placeholder = tex(0x3000);
    let cascades = [tex(0x4000), tex(0x4001), tex(0x4002)];
    let mut s = fresh();
    s.note_texture_read(placeholder);
    record_cascade_casters(&mut s, placeholder, &cascades);
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    apply_pass_rules_with(&mut s, &alt);

    assert_eq!(s.passes().len(), cascades.len());
    for (i, pass) in s.passes().iter().enumerate() {
        assert_eq!(pass.color_texture(), placeholder, "pass {i}: colour kept");
        assert_eq!(pass.color_store(), StoreAction::Store, "pass {i}");
        assert!(
            pipelines_of(pass).iter().all(|&h| h == PSO_WITH),
            "pass {i}"
        );
    }
}

#[test]
fn rule_h_keeps_a_stored_clear_beside_zero_mask_draws() {
    // Nothing later in the submission clears the target, so its Clear is
    // stored and a later frame may read it: the pass keeps its colour.
    let target = tex(0x3000);
    let mut s = fresh();
    s.set_color_render_target(target, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.set_viewport(0, 0, 256, 256, 0.0, 1.0);
    s.clear_color(1, 2, 3, 4);
    s.note_draw_color_write_mask(0);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    apply_pass_rules_with(&mut s, &alt);

    let pass = &s.passes()[0];
    assert_eq!(pass.color_texture(), target, "colour kept");
    assert_eq!(
        pass.color_load(),
        ColorLoad::Clear {
            r: 1,
            g: 2,
            b: 3,
            a: 4
        }
    );
    assert_eq!(pass.color_store(), StoreAction::Store);
    assert_eq!(pipelines_of(pass), [PSO_WITH]);
}

#[test]
fn rule_h_keeps_a_pass_whose_extra_target_stores_its_clear() {
    // Render target 0's Clear is dead (the next pass on it clears it again),
    // but render target 1's Clear is stored. The no-colour pipeline drops
    // every colour attachment, so the pass keeps them all.
    let rt0 = tex(0x3000);
    let rt1 = tex(0x3001);
    let mut s = fresh();
    s.set_color_render_target(rt0, BB_SIZE.0, BB_SIZE.1, RT_FORMAT, RenderScale::IDENTITY);
    s.set_extra_color_render_target(1, Some(slot(rt1, BB_SIZE)));
    s.clear_color(0, 0, 0, 0);
    s.note_draw_color_write_mask(0);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.set_extra_color_render_target(1, None);
    s.clear_color(0, 0, 0, 0);
    s.note_draw_color_write_mask(0xF);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    apply_pass_rules_with(&mut s, &alt);

    let pass = &s.passes()[0];
    assert_eq!(
        pass.color_store(),
        StoreAction::DontCare,
        "Rule C fired on rt0"
    );
    assert_eq!(pass.extra_color()[0].store(), StoreAction::Store);
    assert_eq!(pass.color_texture(), rt0, "render target 0 kept");
    assert_eq!(pass.extra_color()[0].texture(), rt1, "render target 1 kept");
    assert_eq!(pipelines_of(pass), [PSO_WITH]);
}

#[test]
fn rule_h_strips_a_discarded_clear_and_its_clear_quads_together() {
    const PSO_CLEAR_QUAD_COLOR: u64 = 0xCAFE_BABE;
    // The pass opens with a full Clear and repaints part of the target with a
    // colour clear-quad before its zero-mask draws; the next pass clears the
    // target again. Both writes are dead, so the colour and the quad go.
    let atlas = tex(0x3000);
    let mut s = fresh();
    s.set_color_render_target(atlas, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    // A depth surface the size of the target, as a caster pass binds, so Rule H
    // may strip the colour without widening the pass.
    s.set_depth_stencil_attachment(tex(0x2100), (256, 256), false, false);
    s.set_viewport(0, 0, 256, 256, 0.0, 1.0);
    s.clear_color(1, 1, 1, 1);
    let start = s.open_color_clear_quad_block();
    s.emit_command(set_pso(PSO_CLEAR_QUAD_COLOR));
    s.emit_command(dummy_draw());
    s.close_color_clear_quad_block(start);
    s.note_draw_color_write_mask(0);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.clear_color(0, 0, 0, 0);
    s.note_draw_color_write_mask(0xF);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    apply_pass_rules_with(&mut s, &alt);

    let pass = &s.passes()[0];
    assert_eq!(pass.color_texture(), MetalHandle::NULL, "colour stripped");
    assert!(
        pass.color_clear_quad_ranges().is_empty(),
        "clear-quad drained"
    );
    assert_eq!(pipelines_of(pass), [PSO_NO_COLOR]);
    assert_eq!(s.passes()[1].color_texture(), atlas, "clearing pass kept");
}

// ── Draw-state replay: the submit-time rules must leave the encoder state
// ── every surviving draw sees exactly as it was recorded.

#[cfg(debug_assertions)]
const PSO_TILE_COLOR_CLEAR: u64 = 0xCAFE_BABE;
#[cfg(debug_assertions)]
const PSO_TILE_DEPTH_CLEAR: u64 = 0xCCCC_3333;
#[cfg(debug_assertions)]
const DSS_INERT: u64 = 0xD000;
#[cfg(debug_assertions)]
const DSS_DEPTH_CLEAR: u64 = 0xD001;
#[cfg(debug_assertions)]
const DSS_CASTER: u64 = 0xD002;
#[cfg(debug_assertions)]
const CASCADE_TILES: [(u32, u32, u32, u32); 2] = [(0, 0, 256, 256), (256, 0, 256, 256)];

/// Where a colour clear-quad binds the depth-stencil, scissor and cull state.
#[cfg(debug_assertions)]
enum ClearQuadStateOrder {
    /// Inside the block Rule H drains with the colour attachment.
    InsideBlock,
    /// Ahead of the block, so the state survives the drain.
    BeforeBlock,
}

/// Bind depth-stencil state, scissor and cull mode through the dedup cache, as the encoder does.
#[cfg(debug_assertions)]
fn bind_quad_state(
    s: &mut PassState,
    cache: &mut LastBoundCache,
    depth_stencil: u64,
    rect: (u32, u32, u32, u32),
    cull: CullMode,
) {
    if cache.depth_stencil_changed(depth_stencil) {
        s.emit_command(Command::set_depth_stencil_state(depth_stencil));
    }
    if cache.scissor_rect_changed(rect) {
        s.emit_command(Command::set_scissor_rect(rect.0, rect.1, rect.2, rect.3));
    }
    if cache.cull_mode_changed(cull) {
        s.emit_command(Command::set_cull_mode(cull));
    }
}

/// Bind `pipeline` through the dedup cache.
#[cfg(debug_assertions)]
fn bind_pipeline(s: &mut PassState, cache: &mut LastBoundCache, pipeline: u64) {
    if cache.pipeline_changed(pipeline) {
        s.emit_command(set_pso(pipeline));
    }
}

/// Record shadow-cascade tiles into one atlas pass the way `FrameEncoder` emits them.
///
/// Per tile: `SetViewport(tile)`, then `Clear(TARGET | ZBUFFER)` as a colour
/// clear-quad followed by a depth clear-quad, then a caster draw under
/// cull-back. Every state change goes through a real `LastBoundCache`, so the
/// depth quad re-binds neither the scissor nor the cull mode the colour quad
/// just bound.
#[cfg(debug_assertions)]
fn record_cascade_tiles(order: &ClearQuadStateOrder, caster_mask: u32) -> PassState {
    let mut s = fresh();
    s.set_color_render_target(tex(0x3000), 512, 256, RT_FORMAT, RenderScale::IDENTITY);
    // A depth surface the size of the target, as a caster pass binds, so Rule H
    // may strip the colour without widening the pass.
    s.set_depth_stencil_attachment(tex(0x2100), (512, 256), false, false);
    let mut cache = LastBoundCache::new();
    for tile in CASCADE_TILES {
        s.set_viewport(tile.0, tile.1, tile.2, tile.3, 0.0, 1.0);
        if matches!(order, ClearQuadStateOrder::BeforeBlock) {
            bind_quad_state(&mut s, &mut cache, DSS_INERT, tile, CullMode::None);
        }
        let start = s.open_color_clear_quad_block();
        bind_pipeline(&mut s, &mut cache, PSO_TILE_COLOR_CLEAR);
        if matches!(order, ClearQuadStateOrder::InsideBlock) {
            bind_quad_state(&mut s, &mut cache, DSS_INERT, tile, CullMode::None);
        }
        s.emit_command(Command::set_vertex_bytes_at(0x5000, 4, 0));
        cache.invalidate_vertex_buffer();
        s.emit_command(Command::set_fragment_bytes_at(0x5100, 16, 0));
        s.emit_command(dummy_draw());
        s.close_color_clear_quad_block(start);

        bind_pipeline(&mut s, &mut cache, PSO_TILE_DEPTH_CLEAR);
        bind_quad_state(&mut s, &mut cache, DSS_DEPTH_CLEAR, tile, CullMode::None);
        s.emit_command(Command::set_vertex_bytes_at(0x5200, 4, 0));
        cache.invalidate_vertex_buffer();
        s.emit_command(dummy_draw());

        s.note_draw_color_write_mask(caster_mask);
        bind_pipeline(&mut s, &mut cache, PSO_WITH);
        bind_quad_state(&mut s, &mut cache, DSS_CASTER, tile, CullMode::Back);
        if cache.vertex_buffer_changed(0, 0x7000, 0, 1) != VertexBufferBind::Same {
            s.emit_command(Command::set_vertex_buffer(0x7000, 0, 0));
        }
        s.emit_command(dummy_draw());
    }
    s.end_current_pass("test");
    // The next pass opens with a whole-atlas colour Clear (the legacy-break
    // form, so it folds into the load action rather than painting a quad over
    // the atlas the tiles already drew), so Rule C discards the caster pass's
    // colour store and Rule H may drain its colour clear-quads.
    s.clear_color_legacy_break(0, 0, 0, 0);
    s.note_draw_color_write_mask(0xF);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    s
}

/// The side map for the cascade pass: the caster's no-colour sibling, the depth quad itself.
#[cfg(debug_assertions)]
fn cascade_alt() -> FxHashMap<u64, MetalHandle<MTLRenderPipelineStateKind>> {
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    alt.insert(PSO_TILE_DEPTH_CLEAR, pso(PSO_TILE_DEPTH_CLEAR));
    alt
}

/// The scissor and raw cull mode a draw runs with; `None` while no cull mode is bound.
#[cfg(debug_assertions)]
#[derive(Debug, PartialEq, Eq)]
struct QuadDrawState {
    scissor: (u32, u32, u32, u32),
    cull: Option<u32>,
}

/// The state each depth clear-quad draw of `pass` runs with.
#[cfg(debug_assertions)]
fn depth_clear_draw_state(pass: &Pass) -> Vec<QuadDrawState> {
    let mut pipeline = 0;
    let mut scissor = (0, 0, 0, 0);
    let mut cull = None;
    let mut seen = Vec::new();
    for cmd in pass.commands() {
        if cmd.cmd == CommandType::SetRenderPipelineState as u32 {
            pipeline = cmd.param_b;
        } else if cmd.cmd == CommandType::SetScissorRect as u32 {
            scissor = unpack_scissor(cmd);
        } else if cmd.cmd == CommandType::SetCullMode as u32 {
            cull = Some(cmd.param_a);
        } else if cmd.is_draw() && pipeline == PSO_TILE_DEPTH_CLEAR {
            seen.push(QuadDrawState { scissor, cull });
        }
    }
    seen
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "pass rules changed the cull mode a surviving draw sees")]
fn draw_state_check_catches_state_a_drained_clear_quad_bound() {
    // The colour clear-quad binds cull mode and scissor inside its block;
    // Rule H drains the block, and the depth quad that deduplicated against
    // them runs with the previous tile's cull-back and scissor instead.
    let mut s = record_cascade_tiles(&ClearQuadStateOrder::InsideBlock, 0);
    let before = s.debug_record_draw_states();
    let alt = cascade_alt();
    s.strip_color_from_no_color_draw_passes(&alt);
    s.debug_assert_draw_states_preserved(&before, &alt);
}

#[cfg(debug_assertions)]
#[test]
fn rule_h_keeps_the_state_a_clear_quad_binds_ahead_of_its_block() {
    let mut s = record_cascade_tiles(&ClearQuadStateOrder::BeforeBlock, 0);
    let before = s.debug_record_draw_states();
    let alt = cascade_alt();
    s.strip_color_from_no_color_draw_passes(&alt);
    let pass = &s.passes()[0];
    assert_eq!(pass.color_texture(), MetalHandle::NULL, "colour stripped");
    assert!(pass.color_clear_quad_ranges().is_empty(), "blocks drained");
    s.debug_assert_draw_states_preserved(&before, &alt);
    // Each tile's depth clear-quad runs unculled under its own tile's scissor.
    let tile_state = |scissor| QuadDrawState {
        scissor,
        cull: Some(CullMode::None as u32),
    };
    assert_eq!(
        depth_clear_draw_state(&s.passes()[0]),
        vec![tile_state(CASCADE_TILES[0]), tile_state(CASCADE_TILES[1])],
    );
}

#[cfg(debug_assertions)]
#[test]
fn draw_state_check_is_quiet_when_no_rule_drops_a_block() {
    // A caster that writes colour keeps the attachment and the blocks, so
    // nothing the depth quads inherited goes away, whatever the order.
    let mut s = record_cascade_tiles(&ClearQuadStateOrder::InsideBlock, 0xF);
    let before = s.debug_record_draw_states();
    let alt = cascade_alt();
    s.strip_color_from_no_color_draw_passes(&alt);
    assert_eq!(s.passes()[0].color_clear_quad_ranges().len(), 2);
    s.debug_assert_draw_states_preserved(&before, &alt);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "pass rules changed the pipeline a surviving draw sees")]
fn draw_state_check_catches_a_pipeline_rewrite_outside_the_side_map() {
    let mut s = record_cascade_tiles(&ClearQuadStateOrder::BeforeBlock, 0);
    let before = s.debug_record_draw_states();
    s.strip_color_from_no_color_draw_passes(&cascade_alt());
    // Checked against a side map that never named the caster's sibling.
    let mut other = FxHashMap::default();
    other.insert(PSO_TILE_DEPTH_CLEAR, pso(PSO_TILE_DEPTH_CLEAR));
    s.debug_assert_draw_states_preserved(&before, &other);
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

/// A covering depth re-clear of a target drawn earlier in the frame folds.
///
/// The clear covers the whole attachment, so a full-attachment
/// `loadAction = Clear` is exactly D3D9's result and nothing drawn before it
/// needs preserving: no `Load` pass and no clear-quad.
#[test]
fn a_covering_depth_reclear_of_a_drawn_target_folds_into_the_load_action() {
    let mut s = fresh();
    let z = f32::to_bits(1.0);
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    assert_eq!(s.clear_depth(z), DepthClearOutcome::Folded);
    s.emit_command(dummy_draw());
    s.end_current_pass("test_color_rt_switch");

    let z2 = f32::to_bits(0.5);
    assert_eq!(s.clear_depth(z2), DepthClearOutcome::Folded);
    assert_eq!(s.passes().len(), 1, "the fold opens no pass of its own");
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[1].depth_load(), DepthLoad::Clear { value: z2 });
}

/// The stencil twin of the covering depth re-clear.
#[test]
fn a_covering_stencil_reclear_of_a_drawn_target_folds_into_the_load_action() {
    let ds = tex(0x3300);
    let mut s = fresh();
    s.set_depth_stencil_attachment(ds, BB_SIZE, false, true);
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");

    assert_eq!(s.clear_stencil(0x2A), StencilClearOutcome::Folded);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 2);
    assert_eq!(
        s.passes()[1].stencil_load(),
        StencilLoad::Clear { value: 0x2A }
    );
    assert_eq!(
        s.passes()[1].depth_load(),
        DepthLoad::Load,
        "a stencil-only clear leaves the depth plane loading"
    );
}

/// A pass that is open but holds no work takes a covering re-clear as its load action.
#[test]
fn a_covering_reclear_amends_an_open_pass_with_no_work() {
    let mut s = fresh();
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.ensure_pass_open();
    assert_eq!(s.passes()[1].color_load(), ColorLoad::Load);

    assert_eq!(s.clear_color(1, 2, 3, 4), ColorClearOutcome::Folded);
    assert_eq!(s.clear_depth(f32::to_bits(1.0)), DepthClearOutcome::Folded);
    assert_eq!(s.passes().len(), 2);
    assert_eq!(
        s.passes()[1].color_load(),
        ColorLoad::Clear {
            r: 1,
            g: 2,
            b: 3,
            a: 4
        }
    );
    assert!(matches!(
        s.passes()[1].depth_load(),
        DepthLoad::Clear { .. }
    ));
}

/// A combined colour and depth re-clear of drawn targets paints no quad for either plane.
///
/// The colour plane runs first. Were it to paint, its quad would give the
/// pass work and turn the depth plane into a quad as well.
#[test]
fn a_covering_colour_and_depth_reclear_paints_no_quad() {
    let mut s = fresh();
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");

    assert_eq!(s.clear_color(1, 2, 3, 4), ColorClearOutcome::Folded);
    assert_eq!(s.clear_depth(f32::to_bits(1.0)), DepthClearOutcome::Folded);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 2);
    assert!(matches!(
        s.passes()[1].color_load(),
        ColorLoad::Clear { .. }
    ));
    assert!(matches!(
        s.passes()[1].depth_load(),
        DepthLoad::Clear { .. }
    ));
    assert_eq!(
        s.passes()[1].commands().len(),
        2,
        "the viewport and the draw"
    );
}

/// Rule C discards the store of a pass whose target the next pass re-clears in full.
#[test]
fn a_covering_reclear_lets_rule_c_discard_the_previous_colour_store() {
    let rt = tex(0x3000);
    let mut s = fresh();
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.set_viewport(0, 0, 256, 256, 0.0, 1.0);
    s.emit_command(dummy_draw());
    s.set_depth_stencil_attachment(tex(0x5000), (256, 256), false, false);
    s.clear_color(1, 2, 3, 4);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);

    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[0].color_store(), StoreAction::DontCare);
    assert_eq!(s.passes()[1].color_store(), StoreAction::Store);
}

/// A sub-rect clear of a target drawn earlier is no whole-target clear.
///
/// The encoder asks the coverage predicate first and sends a strict
/// sub-region to the region path, which opens the pass with `Load` so the
/// earlier tile survives outside the quad.
#[test]
fn a_sub_rect_clear_of_a_drawn_target_takes_the_region_path() {
    let mut s = fresh();
    s.set_viewport(0, 0, BB_SIZE.0 / 2, BB_SIZE.1, 0.0, 1.0);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.set_viewport(BB_SIZE.0 / 2, 0, BB_SIZE.0 / 2, BB_SIZE.1, 0.0, 1.0);

    assert!(!s.viewport_covers_color_attachment());
    assert!(!s.viewport_covers_depth_attachment());
    s.begin_region_color_clear();
    s.begin_region_depth_stencil_clear();
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[1].color_load(), ColorLoad::Load);
    assert_eq!(s.passes()[1].depth_load(), DepthLoad::Load);
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
    s.set_depth_stencil_attachment(cascade_depth, BB_SIZE, /* is_sampleable */ true, false);
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
        rt_depth, BB_SIZE, /* is_sampleable */ false, /* has_stencil */ false,
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

/// A cascade rebound as the boundary reports it is a no-op.
///
/// Neither the pass break nor the loss of Rule B's exemption fires.
///
/// A save/restore cycle re-binds the same cascade handle mid-pass. The D3D9
/// boundary derives `is_sampleable` from the surface's owning texture, so
/// the second bind carries the same flag as the first, and the repeat-bind
/// early-out leaves the pass and the keep-Store exemption alone.
#[test]
fn cascade_rebind_with_the_same_sampleable_flag_is_a_no_op() {
    let cascade_depth = tex(0xCAFE_7000);
    let mut s = fresh();
    // Bind as sampleable, draw into it, then rebind the SAME handle the way
    // a save/restore of the cascade surface does.
    s.set_depth_stencil_attachment(cascade_depth, BB_SIZE, true, false);
    s.emit_command(dummy_draw());
    let passes_before = s.passes().len();
    s.set_depth_stencil_attachment(cascade_depth, BB_SIZE, true, false);
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
        "the sampleable flag stays set through the rebind",
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
        "Rule B keeps Store through the rebind",
    );
}

/// Drive a sample-then-reuse pair of frames, optionally retiring the texture between them.
///
/// Frame N renders into `rt` and samples it, which puts the handle in the
/// session-wide sampled set. Frame N+1 binds the same address as a colour
/// target nothing reads, then clears it in a later pass. Returns the first
/// pass's store action, which Rule C discards only for a handle it does not
/// consider sampled.
fn colour_reuse_after_sample(retire: bool, bind: Command) -> StoreAction {
    let rt = tex(0xCAFE_8000);
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
    s.emit_command(bind);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    if retire {
        s.unregister_texture(rt);
    }
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: BB_SIZE,
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
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
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.clear_color(0, 0, 0, 0);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    s.passes()
        .iter()
        .find(|p| p.color_texture() == rt)
        .expect("the reuse pass is present")
        .color_store()
}

/// A retired colour handle stops looking sampled, so its address can be reused.
///
/// `seen_sampled_textures` deliberately outlives the frame that filled it, so
/// every rule that consults it keeps `Load` and `Store` on the handle for as
/// long as the entry stands. That is right while the texture lives and wrong
/// the moment Metal is free to hand its address to an unrelated allocation.
#[test]
fn a_retired_colour_handle_drops_its_sampled_marking() {
    for ((stage, retired_bind), (_, live_bind)) in sampler_binds(0xCAFE_8000)
        .into_iter()
        .zip(sampler_binds(0xCAFE_8000))
    {
        assert_eq!(
            colour_reuse_after_sample(true, retired_bind),
            StoreAction::DontCare,
            "{stage:?}: nothing reads the reused address before its next clear, so Rule C \
             drops the store",
        );
        assert_eq!(
            colour_reuse_after_sample(false, live_bind),
            StoreAction::Store,
            "{stage:?}: a live texture sampled last frame keeps its store, which is what the \
             prune must not weaken",
        );
    }
}

/// Retiring a texture re-arms every frame-scoped record keyed on its handle.
///
/// The colour seen-set is keyed on `(handle, subresource)` and the sampled
/// set on the handle alone, so a destroy and a same-frame reallocation onto
/// the same address must leave both answering for the new texture.
#[test]
fn unregister_texture_re_arms_the_frame_scoped_sets() {
    let rt = tex(0xCAFE_8100);
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
    assert!(
        s.texture_sampled_this_frame(rt),
        "the bind marks the handle sampled this frame",
    );
    s.unregister_texture(rt);
    assert!(
        !s.texture_sampled_this_frame(rt),
        "an upload into the reallocated address must not trigger a rename",
    );
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    let reuse = s.passes().len() - 1;
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.clear_color(0, 0, 0, 0);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    assert_eq!(
        s.passes()[reuse].color_store(),
        StoreAction::DontCare,
        "the reused address no longer counts as sampled, so Rule C drops the store \
         its next clear overwrites",
    );
}

/// A replaced back-buffer view retires the registration the old one held.
///
/// `Reset` destroys the back buffer together with its sRGB view and creates
/// both again, and Metal hands the freed address back for the replacement
/// readily enough that the texture can come back at the address it had while
/// the view behind it is a different object. The retired view's entry would
/// otherwise resolve every texture allocated at its address to this back
/// buffer, which is a colour target a pass is free to be held ahead of.
#[test]
fn a_replaced_backbuffer_view_retires_the_old_registration() {
    let mut s = fresh();
    let replacement = tex(0x1002);
    assert_eq!(
        s.twin_of(backbuffer()),
        backbuffer_srgb(),
        "the frame registers the pair it was handed",
    );
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_srgb: replacement,
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: BB_SIZE,
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    assert_eq!(
        s.twin_of(backbuffer()),
        replacement,
        "the fresh view takes the slot",
    );
    assert_eq!(
        s.texture_view_to_base.get(&backbuffer_srgb()),
        Some(&backbuffer())
    );
    s.unregister_texture(backbuffer_srgb());
    assert!(!s.texture_view_to_base.contains_key(&backbuffer_srgb()));
}

/// A retired depth handle stops being sampleable, so its address can be reused.
///
/// Metal is free to hand the address of a destroyed `MTLTexture` back for the
/// next allocation. The encoder reports the retirement, and an unrelated
/// depth surface that lands on that address binds non-sampleable and gets
/// Rule B's `DontCare` back rather than inheriting the cascade's `Store`.
#[test]
fn a_retired_depth_handle_drops_its_sampleable_marking() {
    let handle = tex(0xCAFE_7100);
    let mut s = fresh();
    s.set_depth_stencil_attachment(handle, BB_SIZE, true, false);
    s.emit_command(dummy_draw());
    assert!(
        s.is_depth_handle_sampleable(handle),
        "the cascade bind marks the handle",
    );
    s.unregister_texture(handle);
    assert!(
        !s.is_depth_handle_sampleable(handle),
        "retiring the texture drops the marking",
    );
    // The address comes back as a standalone depth surface: non-sampleable,
    // never sampled, so Rule B applies.
    s.set_depth_stencil_attachment(handle, BB_SIZE, false, false);
    s.emit_command(dummy_draw());
    s.finalize_store_actions(false);
    let reused_pass = s.passes().last().expect("the reuse pass is present");
    assert_eq!(
        reused_pass.depth_store(),
        StoreAction::DontCare,
        "the reused address does not inherit the cascade keep-Store exemption",
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
        msaa_texture: MetalHandle::NULL,
        msaa_srgb_texture: MetalHandle::NULL,
        sample_count: 1,
        subresource: 0,
        size,
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
    // A game target keeps its contents across frames, so even its first use
    // under a covering viewport loads, while the back buffer beside it
    // takes Rule A's discard.
    assert_eq!(pass.extra_color()[0].load(), ColorLoad::Load);
    assert_eq!(pass.color_load(), ColorLoad::DontCare);
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
fn rebinding_an_extra_target_with_a_new_format_breaks_the_pass() {
    // Same rule as render target 0: an extra's format is frozen into the
    // pass at open, so a same-handle rebind that changes it closes the pass.
    let mut s = fresh();
    s.set_extra_color_render_target(1, Some(slot(tex(0x3000), BB_SIZE)));
    s.emit_command(dummy_draw());
    let mut recoloured = slot(tex(0x3000), BB_SIZE);
    recoloured.format = PixelFormat::Rgba16Float;
    s.set_extra_color_render_target(1, Some(recoloured));
    assert!(s.current_pass_closed(), "a new format ends the pass");
    s.emit_command(dummy_draw());

    assert_eq!(s.passes().len(), 2);
    assert_eq!(
        s.passes()[0].extra_color()[0].format(),
        PixelFormat::R8Unorm
    );
    assert_eq!(
        s.passes()[1].extra_color()[0].format(),
        PixelFormat::Rgba16Float
    );
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
fn clear_after_a_target_was_drawn_folds_into_every_targets_load_action() {
    let mut s = fresh();
    s.set_extra_color_render_target(1, Some(slot(tex(0x3000), BB_SIZE)));
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    s.emit_command(dummy_draw());
    // A depth change ends the pass; the extra target has content now.
    s.set_depth_stencil_attachment(tex(0x5000), BB_SIZE, false, false);
    assert_eq!(s.clear_color(1, 2, 3, 4), ColorClearOutcome::Folded);
    s.emit_command(dummy_draw());
    let pass = &s.passes()[1];
    let clear = ColorLoad::Clear {
        r: 1,
        g: 2,
        b: 3,
        a: 4,
    };
    assert_eq!(pass.color_load(), clear);
    assert_eq!(pass.extra_color()[0].load(), clear);
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
        &TargetExtent::new(target.scale, target.logical_size, target.size),
        target.format,
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

/// A clear pending when the extras are taken lands on every target it was issued against.
///
/// `Clear` with no pass open stashes the clear, then the per-target bracket
/// takes the binding set because render target 2 is sized differently. The
/// clear-only pass that flushes out must still carry render target 1.
#[test]
fn take_flushes_a_pending_clear_onto_the_bound_extras() {
    let mut s = fresh();
    s.set_extra_color_render_target(1, Some(slot(tex(0x3000), BB_SIZE)));
    s.set_extra_color_render_target(2, Some(slot(tex(0x3001), (64, 64))));
    assert_eq!(s.extra_present_mask(), 0b001);
    assert_eq!(s.clear_color(1, 2, 3, 4), ColorClearOutcome::Folded);
    let saved = s.take_color_attachments();
    assert!(saved.slot(1).is_some());
    let pass = s.passes().last().expect("clear-only pass");
    let clear = ColorLoad::Clear {
        r: 1,
        g: 2,
        b: 3,
        a: 4,
    };
    assert_eq!(pass.color_load(), clear);
    assert_eq!(pass.extra_color()[0].texture(), tex(0x3000));
    assert_eq!(pass.extra_color()[0].load(), clear);
    assert!(!pass.extra_color()[1].is_bound());
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
fn rule_c_applies_per_attachment() {
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
    // rt_b is never used again this frame: it keeps its store for a later frame.
    assert_eq!(first.extra_color()[1].store(), StoreAction::Store);
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
    let original = command_allocations(s.passes());
    s.coalesce_clear_only_passes();
    assert_eq!(s.passes().len(), 2, "clear-only pass folded");
    assert_eq!(command_allocations(s.passes()), original[1..]);
    assert_eq!(s.command_vec_pool.len(), 1);
    assert_eq!(s.command_vec_pool[0].as_ptr(), original[0].0);
    assert_eq!(s.command_vec_pool[0].capacity(), original[0].1);
    assert!(s.command_vec_pool[0].is_empty());
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
fn rule_e_refuses_a_pass_that_draws_into_the_target_as_an_extra() {
    // Clear(rt), then an MRT pass that binds rt as render target 1 and
    // draws into it, then a pass that rebinds rt as render target 0 with
    // Load. The MRT pass carries a different attachment set, so it is no
    // merge target, but it writes rt: the clear must stay ahead of it
    // instead of folding into the third pass and wiping those writes.
    let rt = tex(0x3000);
    let other = tex(0x4000);
    let mut s = fresh();
    s.set_color_render_target(rt, BB_SIZE.0, BB_SIZE.1, BB_FORMAT, RenderScale::IDENTITY);
    s.clear_color(1, 2, 3, 4);
    s.set_color_render_target(
        other,
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_extra_color_render_target(1, Some(slot(rt, BB_SIZE)));
    s.emit_command(dummy_draw());
    s.set_extra_color_render_target(1, None);
    s.set_color_render_target(rt, BB_SIZE.0, BB_SIZE.1, BB_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 3);
    assert_eq!(s.passes()[1].extra_color()[0].texture(), rt);
    assert_eq!(s.passes()[1].extra_color()[0].load(), ColorLoad::Load);
    assert_eq!(s.passes()[2].color_texture(), rt);
    assert_eq!(s.passes()[2].color_load(), ColorLoad::Load);
    s.coalesce_clear_only_passes();
    assert_eq!(
        s.passes().len(),
        3,
        "the clear stays ahead of the pass that draws into rt as an extra"
    );
    assert_eq!(s.passes()[2].color_load(), ColorLoad::Load);
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
    // rt_a is read back later, so its store survives; rt_b is cleared again
    // by the next pass, so Rule C kills its store.
    s.note_color_read_back(rt_a);
    s.set_extra_color_render_target(1, None);
    s.set_extra_color_render_target(2, None);
    s.set_color_render_target(rt_b, BB_SIZE.0, BB_SIZE.1, BB_FORMAT, RenderScale::IDENTITY);
    s.clear_color(5, 6, 7, 8);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    s.strip_dead_color_in_clear_only_passes();
    assert_eq!(s.passes().len(), 2);
    let pass = &s.passes()[0];
    assert!(
        pass.extra_color()[0].is_bound(),
        "read-back target keeps its store"
    );
    assert!(!pass.extra_color()[1].is_bound(), "dead extra stripped");
    s.cull_dead_clear_only_passes();
    assert_eq!(s.passes().len(), 2, "a live store keeps the pass");
}

#[test]
fn clear_colour_survives_rule_g_stripping_target_zero() {
    // Clear rt_a (target 0) and rt_b (target 1) red, unbind rt_b so a
    // clear-only pass materialises, then clear rt_a blue and draw while
    // sampling rt_b. Rule C kills rt_a's store in the clear-only pass and
    // Rule G strips it; rt_b keeps its store and its Clear. The pass still
    // has to carry red for rt_b, not the zeros of a stripped target 0.
    let rt_a = tex(0x3000);
    let rt_b = tex(0x3001);
    let red = (1.0f32.to_bits(), 0, 0, 1.0f32.to_bits());
    let blue = (0, 0, 1.0f32.to_bits(), 1.0f32.to_bits());
    let mut s = fresh();
    s.set_color_render_target(rt_a, BB_SIZE.0, BB_SIZE.1, RT_FORMAT, RenderScale::IDENTITY);
    s.set_extra_color_render_target(1, Some(slot(rt_b, BB_SIZE)));
    s.clear_color(red.0, red.1, red.2, red.3);
    s.set_extra_color_render_target(1, None);
    s.clear_color(blue.0, blue.1, blue.2, blue.3);
    s.emit_command(Command::set_fragment_texture(rt_b.raw(), 0));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    s.strip_dead_color_in_clear_only_passes();
    assert_eq!(s.passes().len(), 2);
    let pass = &s.passes()[0];
    assert_eq!(pass.color_texture(), MetalHandle::NULL, "target 0 stripped");
    assert_eq!(pass.color_load(), ColorLoad::DontCare);
    let extra = &pass.extra_color()[0];
    assert_eq!(extra.texture(), rt_b);
    assert_eq!(extra.store(), StoreAction::Store, "sampled later");
    assert_eq!(
        extra.load(),
        ColorLoad::Clear {
            r: red.0,
            g: red.1,
            b: red.2,
            a: red.3
        }
    );
    assert_eq!(pass.color_clear_rgba(), Some(red));
    assert_eq!(s.passes()[1].color_clear_rgba(), Some(blue));
}

#[test]
fn mid_frame_flush_keeps_every_colour_store() {
    // Two clear-only passes on two targets; the first is read back, which
    // flushes the frame. The second target is read back afterwards, so
    // its last-use store must survive the flush and Rule F must keep its
    // pass.
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
    // A real frame end keeps it too: a later frame may read the target.
    s.finalize_store_actions(false);
    assert_eq!(s.passes()[1].color_store(), StoreAction::Store);
}

// ── Blits in the read/write model ─────────────────────────────

fn copy_blit(src: MetalHandle<MTLTextureKind>, dst: MetalHandle<MTLTextureKind>) -> BlitCommand {
    BlitCommand::copy_texture_to_texture_full_mip(src.raw(), dst.raw(), 0, 64, 64)
}

#[test]
fn a_stretch_rect_read_keeps_the_last_use_store() {
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
    // A copy into the back buffer is queued while rt_y is bound, so it
    // lands in rt_y's pass. The back buffer's own first pass, which Rule A
    // would otherwise open with a discard, must still Load the copy.
    let rt_src = tex(0x3000);
    let rt_y = tex(0x5000);
    let mut s = fresh();
    s.set_color_render_target(rt_y, 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.set_viewport(0, 0, 64, 64, 0.0, 1.0);
    s.push_pending_leading_blit(copy_blit(rt_src, backbuffer()));
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[0].color_texture(), rt_y);
    assert_eq!(s.passes()[0].leading_blits().len(), 1);
    assert_eq!(s.passes()[1].color_texture(), backbuffer());
    assert_eq!(s.passes()[1].leading_blits().len(), 0);
    assert_eq!(s.passes()[1].color_load(), ColorLoad::Load);
}

#[test]
fn blit_written_set_resets_with_the_frame() {
    let rt_src = tex(0x3000);
    let mut s = fresh();
    s.push_pending_leading_blit(copy_blit(rt_src, backbuffer()));
    s.take_pending_leading_blits();
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: BB_SIZE,
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    s.emit_command(dummy_draw());
    assert_eq!(s.passes()[0].color_texture(), backbuffer());
    assert_eq!(s.passes()[0].color_load(), ColorLoad::DontCare);
}

#[test]
fn blit_written_set_survives_a_mid_frame_flush() {
    // A copy into the back buffer and one into the depth surface run as a
    // trailing blit pass, then a mid-frame flush (a readback). The D3D9 frame
    // continues, so the continuation's first pass on those attachments must
    // Load the copies, not open with Rule A's first-use `DontCare`.
    let rt_src = tex(0x3000);
    let depth_src = tex(0x4000);
    let mut s = fresh();
    s.push_pending_leading_blit(copy_blit(rt_src, backbuffer()));
    s.push_pending_leading_blit(copy_blit(depth_src, depth()));
    s.take_pending_leading_blits();
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: BB_SIZE,
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: true,
    });
    s.emit_command(dummy_draw());
    assert_eq!(s.passes()[0].color_texture(), backbuffer());
    assert_eq!(
        s.passes()[0].color_load(),
        ColorLoad::Load,
        "continuation loads the back buffer a blit wrote before the flush",
    );
    assert_eq!(
        s.passes()[0].depth_load(),
        DepthLoad::Load,
        "continuation loads the depth surface a blit wrote before the flush",
    );
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

#[test]
fn rule_e_aborts_when_the_target_passes_depth_transfer_reads_the_depth() {
    // Clear(ZBUFFER) with no draw materialises as a clear-only pass, then a
    // RESZ-style depth transfer reads that depth into another texture as a
    // leading blit of the next pass on the same depth. The blit runs before
    // that pass's load action, so folding the clear into it would hand the
    // transfer the pre-clear depth.
    let resolved = tex(0x9000);
    let mut s = fresh();
    s.clear_depth(f32::to_bits(1.0));
    s.flush_pending_clears();
    let mut transfer = copy_blit(depth(), resolved);
    transfer.cmd = BlitCommandType::TransferDepth as u32;
    s.push_pending_leading_blit(transfer);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[1].leading_blits().len(), 1);
    assert_eq!(s.passes()[1].depth_load(), DepthLoad::Load);
    s.coalesce_clear_only_passes();
    assert_eq!(s.passes().len(), 2, "the clear stays ahead of the transfer");
    assert!(matches!(
        s.passes()[0].depth_load(),
        DepthLoad::Clear { .. }
    ));
    assert_eq!(s.passes()[1].depth_load(), DepthLoad::Load);
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
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: BB_SIZE,
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
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: BB_SIZE,
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    s.emit_command(dummy_draw());
    assert_eq!(s.passes()[0].color_load(), ColorLoad::DontCare);
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::DontCare);
}

/// A different mip level of the same depth texture is a different attachment.
#[test]
fn depth_level_change_breaks_the_pass_and_a_repeat_bind_does_not() {
    let d = depth();
    let mut s = fresh();
    s.set_depth_stencil_attachment_level(d, 0, BB_SIZE, false, false);
    s.emit_command(dummy_draw());
    s.set_depth_stencil_attachment_level(d, 0, BB_SIZE, false, false);
    s.emit_command(dummy_draw());
    s.set_depth_stencil_attachment_level(d, 1, (BB_SIZE.0 / 2, BB_SIZE.1 / 2), false, false);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(
        s.passes().len(),
        2,
        "a repeat bind of the same level stays in the pass; a new level ends it"
    );
    assert_eq!(s.passes()[0].depth_level(), 0);
    assert_eq!(s.passes()[1].depth_level(), 1);
}

/// The whole-target `Clear` bound: a covering viewport folds, a sub-rect does not.
///
/// Both predicates in one test because the two attachments answer
/// independently and the interesting cases are the same four for each.
#[test]
fn viewport_coverage_is_answered_per_attachment() {
    let mut s = fresh();

    // The default viewport (the game never called `SetViewport`) falls back to
    // the attachment's own extent, so it covers both.
    assert!(s.viewport_covers_color_attachment());
    assert!(s.viewport_covers_depth_attachment());

    // Exactly the attachment: covers.
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    assert!(s.viewport_covers_color_attachment());
    assert!(s.viewport_covers_depth_attachment());

    // Larger than the attachment still covers: D3D9 clips the viewport to the
    // render target, so an oversized one is not a sub-region. The comparison
    // has to be greater-or-equal, not equality.
    s.set_viewport(0, 0, 8192, 8192, 0.0, 1.0);
    assert!(s.viewport_covers_color_attachment());
    assert!(s.viewport_covers_depth_attachment());

    // Full extent but moved off the origin: a sub-region, because the far
    // edges fall outside.
    s.set_viewport(16, 16, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    assert!(!s.viewport_covers_color_attachment());
    assert!(!s.viewport_covers_depth_attachment());

    // At the origin but narrower on one axis only.
    s.set_viewport(0, 0, BB_SIZE.0 - 1, BB_SIZE.1, 0.0, 1.0);
    assert!(!s.viewport_covers_color_attachment());
    assert!(!s.viewport_covers_depth_attachment());
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1 - 1, 0.0, 1.0);
    assert!(!s.viewport_covers_color_attachment());
    assert!(!s.viewport_covers_depth_attachment());
}

/// Nothing attached means nothing to bound, so the clear folds.
#[test]
fn viewport_coverage_folds_when_the_attachment_is_absent() {
    let mut s = fresh();
    s.set_viewport(100, 100, 64, 64, 0.0, 1.0);
    assert!(!s.viewport_covers_color_attachment(), "both are bound");
    assert!(!s.viewport_covers_depth_attachment(), "both are bound");

    s.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
    assert!(
        s.viewport_covers_depth_attachment(),
        "an unbound depth attachment has no extent to bound the clear to"
    );
    assert!(
        !s.viewport_covers_color_attachment(),
        "and the colour side is unaffected by the depth unbind"
    );
}

/// The two attachments are measured separately, not through render target 0.
#[test]
fn a_depth_attachment_sized_unlike_the_colour_one_is_measured_on_its_own() {
    let mut s = fresh();
    // A cascade tile: the depth attachment is smaller than the back buffer and
    // the viewport covers all of it.
    s.set_depth_stencil_attachment(tex(0x9000), (256, 256), false, false);
    s.set_viewport(0, 0, 256, 256, 0.0, 1.0);
    assert!(
        s.viewport_covers_depth_attachment(),
        "the viewport covers the whole depth attachment"
    );
    assert!(
        !s.viewport_covers_color_attachment(),
        "the same viewport is a sub-region of the larger colour attachment"
    );

    // And the other way round: a depth surface larger than render target 0,
    // which D3D9 permits.
    s.set_depth_stencil_attachment(tex(0x9100), (1024, 1024), false, false);
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    assert!(s.viewport_covers_color_attachment());
    assert!(
        !s.viewport_covers_depth_attachment(),
        "a viewport the size of render target 0 is a sub-region of a larger depth surface"
    );
}

/// Coverage is asked in the bound texture's space, not the reported one.
#[test]
fn viewport_coverage_converts_through_the_render_scale() {
    let mut s = fresh_scaled();
    // The game's own numbers describe the whole reported back buffer; the
    // rasterized attachments are half that, and the converted viewport has to
    // be compared against the rasterized extent for the answer to hold.
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    assert!(s.viewport_covers_color_attachment());
    assert!(s.viewport_covers_depth_attachment());

    s.set_viewport(0, 0, BB_SIZE.0 / 2, BB_SIZE.1 / 2, 0.0, 1.0);
    assert!(!s.viewport_covers_color_attachment());
    assert!(!s.viewport_covers_depth_attachment());
}

/// A clipped `Clear` region is measured like the viewport, per attachment.
///
/// A scissor or rect that spans the target turns a region clear into a
/// whole-target one, which may fold into the load action.
#[test]
fn region_coverage_is_answered_per_attachment() {
    let mut s = fresh();
    let whole = (0, 0, BB_SIZE.0, BB_SIZE.1);

    // A scissor equal to the target, and a rect larger than it.
    assert!(s.region_covers_color_attachment(whole));
    assert!(s.region_covers_depth_attachment(whole));
    assert!(s.region_covers_color_attachment((0, 0, 8192, 8192)));
    assert!(s.region_covers_depth_attachment((0, 0, 8192, 8192)));

    // A scissor smaller than the target on one axis, or off the origin.
    for region in [
        (0, 0, BB_SIZE.0 - 1, BB_SIZE.1),
        (0, 0, BB_SIZE.0, BB_SIZE.1 - 1),
        (1, 0, BB_SIZE.0, BB_SIZE.1),
        (0, 1, BB_SIZE.0, BB_SIZE.1),
    ] {
        assert!(!s.region_covers_color_attachment(region), "{region:?}");
        assert!(!s.region_covers_depth_attachment(region), "{region:?}");
    }

    // A depth surface larger than render target 0 is judged against its own
    // extent: the region that covers the colour target leaves depth out.
    s.set_depth_stencil_attachment(tex(0x9100), (1024, 1024), false, false);
    assert!(s.region_covers_color_attachment(whole));
    assert!(!s.region_covers_depth_attachment(whole));
    assert!(s.region_covers_depth_attachment((0, 0, 1024, 1024)));

    // Nothing bound means nothing to bound.
    s.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
    assert!(s.region_covers_depth_attachment((5, 5, 1, 1)));
}

/// Region coverage compares in the bound texture's space.
///
/// The encoder converts the game's rects before clipping them, so a rect
/// over the whole reported back buffer covers the smaller rasterized one and
/// a rect over the rasterized size in reported numbers does not.
#[test]
fn region_coverage_is_measured_in_texture_space() {
    let s = fresh_scaled();
    let texture_region = |r: (i32, i32, i32, i32)| {
        let (x1, y1, x2, y2) = s.target_scale().rect_edges_i32(r);
        (
            x1.cast_unsigned(),
            y1.cast_unsigned(),
            (x2 - x1).cast_unsigned(),
            (y2 - y1).cast_unsigned(),
        )
    };
    let full = (0, 0, BB_SIZE.0.cast_signed(), BB_SIZE.1.cast_signed());
    assert!(s.region_covers_color_attachment(texture_region(full)));
    assert!(s.region_covers_depth_attachment(texture_region(full)));

    let half = (0, 0, full.2 / 2, full.3 / 2);
    assert!(!s.region_covers_color_attachment(texture_region(half)));
    assert!(!s.region_covers_depth_attachment(texture_region(half)));
}

#[test]
fn srgb_twin_bind_marks_the_base_texture_sampled() {
    for ((stage, bind), (_, stale_bind)) in
        sampler_binds(0x7E11).into_iter().zip(sampler_binds(0x7E11))
    {
        let mut s = fresh();
        let base = tex(0x7E10);
        let twin = tex(0x7E11);
        s.register_srgb_twin(twin, base);
        // A draw sampling through the sRGB twin reads the base's storage:
        // rename-at-overlap and the store-action rules must see the base as
        // sampled even though the command stream only carries the twin.
        s.emit_command(bind);
        assert!(s.texture_sampled_this_frame(base), "{stage:?}");
        assert!(s.texture_sampled_this_frame(twin), "{stage:?}");
        s.reset_frame(&FrameReset {
            backbuffer: backbuffer(),
            backbuffer_srgb: backbuffer_srgb(),
            backbuffer_msaa: MetalHandle::NULL,
            backbuffer_msaa_srgb: MetalHandle::NULL,
            backbuffer_sample_count: 1,
            backbuffer_size: BB_SIZE,
            backbuffer_format: BB_FORMAT,
            backbuffer_contents: BackbufferContents::Undefined,
            depth_texture: depth(),
            depth_size: BB_SIZE,
            depth_has_stencil: false,
            render_scale: RenderScale::IDENTITY,
            continues_frame: false,
        });
        assert!(!s.texture_sampled_this_frame(base), "{stage:?}");
        assert!(!s.texture_sampled_this_frame(twin), "{stage:?}");
        assert!(
            s.seen_sampled_textures.contains(&base),
            "{stage:?}: reset_frame must preserve the session-wide base read"
        );
        assert!(
            s.seen_sampled_textures.contains(&twin),
            "{stage:?}: reset_frame must preserve the session-wide view read"
        );
        // Detaching an attachment preserves identity for queued sampling commands.
        s.unregister_srgb_twin(twin);
        s.emit_command(stale_bind);
        assert!(s.texture_sampled_this_frame(base), "{stage:?}");
        assert!(
            s.seen_sampled_textures.contains(&base),
            "{stage:?}: unregistering the view mapping does not retire either texture"
        );
        s.unregister_texture(twin);
        s.unregister_texture(base);
        assert!(!s.seen_sampled_textures.contains(&base), "{stage:?}");
        assert!(!s.seen_sampled_textures.contains(&twin), "{stage:?}");
    }
}

/// `D3DRS_SRGBWRITEENABLE` attaches the render target's sRGB twin view.
///
/// The pass must carry the base handle for identity (the load/store rules
/// and the seen sets reason about it) and the twin only as the attachment,
/// with the pipeline format keyed on the sRGB variant so the render
/// pipeline the draw path builds declares what the pass binds.
#[test]
fn srgb_write_attaches_the_twin_view_and_keys_the_srgb_format() {
    let mut s = fresh();
    let base = tex(0x7F10);
    let twin = tex(0x7F11);
    s.register_srgb_twin(twin, base);
    s.set_color_render_target(base, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.set_srgb_write_enabled(true);
    assert!(s.pass_srgb_write());
    assert_eq!(s.current_color_format(), PixelFormat::Bgra8UnormSrgb);
    s.ensure_pass_open();
    let pass = &s.passes()[0];
    assert_eq!(pass.color_texture(), base, "identity stays on the base");
    assert_eq!(pass.color_attachment_texture(), twin);
    assert_eq!(pass.color_format(), PixelFormat::Bgra8UnormSrgb);
}

/// Turning the state off mid-frame ends the pass and returns to the base view.
///
/// One render encoder has one set of attachment views, so draws on either
/// side of the toggle cannot share a pass.
#[test]
fn srgb_write_toggle_breaks_the_pass() {
    let mut s = fresh();
    let base = tex(0x7F20);
    let twin = tex(0x7F21);
    s.register_srgb_twin(twin, base);
    s.set_color_render_target(base, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.set_srgb_write_enabled(true);
    s.ensure_pass_open();
    s.emit_command(dummy_draw());
    s.set_srgb_write_enabled(false);
    s.ensure_pass_open();
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.passes()[0].color_attachment_texture(), twin);
    assert_eq!(s.passes()[1].color_attachment_texture(), base);
    assert_eq!(s.passes()[1].color_format(), RT_FORMAT);
}

/// A colour target with no sRGB view keeps the linear attachment.
///
/// The encode then has to happen in the pixel shader, which is what the
/// `VariantFlags::SRGB_WRITE` emitter path is for.
#[test]
fn srgb_write_without_a_twin_keeps_the_linear_attachment() {
    let mut s = fresh();
    let base = tex(0x7F30);
    s.set_color_render_target(base, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.set_srgb_write_enabled(true);
    assert!(!s.pass_srgb_write());
    assert_eq!(s.current_color_format(), RT_FORMAT);
    s.ensure_pass_open();
    assert_eq!(s.passes()[0].color_attachment_texture(), base);
}

/// One extra target without a twin keeps the whole attachment set linear.
///
/// A render pass binds one set of views: a target that cannot encode would
/// otherwise be written linear while its neighbours encode.
#[test]
fn an_extra_target_without_a_twin_keeps_the_whole_set_linear() {
    let mut s = fresh();
    let base = tex(0x7F40);
    let twin = tex(0x7F41);
    let extra = tex(0x7F42);
    s.register_srgb_twin(twin, base);
    s.set_color_render_target(base, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.set_extra_color_render_target(
        1,
        Some(ExtraColorSlot {
            texture: extra,
            msaa_texture: MetalHandle::NULL,
            msaa_srgb_texture: MetalHandle::NULL,
            sample_count: 1,
            subresource: 0,
            size: (256, 256),
            logical_size: (256, 256),
            format: RT_FORMAT,
            scale: RenderScale::IDENTITY,
            has_alpha: true,
        }),
    );
    s.set_srgb_write_enabled(true);
    assert!(!s.pass_srgb_write());

    // Give the extra a twin of its own and the whole set can encode.
    let extra_twin = tex(0x7F43);
    s.register_srgb_twin(extra_twin, extra);
    assert!(s.pass_srgb_write());
    s.ensure_pass_open();
    let pass = &s.passes()[0];
    assert_eq!(pass.color_attachment_texture(), twin);
    assert_eq!(pass.extra_color()[0].attachment_texture(), extra_twin);
    assert_eq!(pass.extra_color()[0].texture(), extra);
    assert_eq!(
        s.extra_color_attachments().formats[0],
        PixelFormat::Bgra8UnormSrgb
    );
}

/// Rule E never folds a linear clear into a pass that writes through the twin.
///
/// The clear value is stored raw through the linear view and sRGB-encoded
/// through the twin, so moving the load action across the two would change
/// the colour the target ends up holding.
#[test]
fn a_clear_only_pass_does_not_coalesce_across_an_srgb_view_change() {
    let mut s = fresh();
    let base = tex(0x7F50);
    let twin = tex(0x7F51);
    s.register_srgb_twin(twin, base);
    s.set_color_render_target(base, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
    // Clear with linear writes, then draw with sRGB writes: the clear-only
    // pass and the draw pass carry the same texture but different views.
    s.clear_color(1, 2, 3, 4);
    s.set_srgb_write_enabled(true);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();
    assert_eq!(s.passes().len(), 2, "the clear must stay in its own pass");
    assert!(matches!(
        s.passes()[0].color_load(),
        ColorLoad::Clear { .. }
    ));
    assert!(s.passes()[0].color_srgb_texture.is_null());
    assert_eq!(s.passes()[1].color_attachment_texture(), twin);
}

/// The back buffer's sRGB twin is registered from the frame reset.
///
/// A `D3DRS_SRGBWRITEENABLE` draw straight onto the swap chain therefore
/// attaches the twin, exactly as one onto a render-target texture does. The
/// pair is re-supplied every frame because `Reset` and an auto-resize
/// replace both halves together.
#[test]
fn the_backbuffer_attaches_its_srgb_twin() {
    let mut s = fresh();
    s.set_srgb_write_enabled(true);
    assert!(s.pass_srgb_write());
    assert_eq!(s.current_color_format(), PixelFormat::Bgra8UnormSrgb);
    s.ensure_pass_open();
    let pass = &s.passes()[0];
    assert_eq!(pass.color_texture(), backbuffer());
    assert_eq!(pass.color_attachment_texture(), backbuffer_srgb());
}

/// A replaced back buffer drops the retired twin's registration.
///
/// `Reset` destroys the old texture and its view together, so a later
/// binding must not be able to resolve the dead one.
#[test]
fn replacing_the_backbuffer_forgets_the_retired_twin() {
    let mut s = fresh();
    let fresh_bb = tex(0x1100);
    let fresh_twin = tex(0x1101);
    s.reset_frame(&FrameReset {
        backbuffer: fresh_bb,
        backbuffer_srgb: fresh_twin,
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: BB_SIZE,
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    s.set_srgb_write_enabled(true);
    s.ensure_pass_open();
    assert_eq!(s.passes()[0].color_attachment_texture(), fresh_twin);
    // The retired pair is gone: rebinding the old texture finds no twin.
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    assert!(!s.pass_srgb_write());
}

/// A scoped internal pass attaches the base view even while the game's state is on.
///
/// `StretchRect` copies pixels verbatim, so no render state reaches it and
/// its quad pipeline declares the destination's own format. If the scoped
/// pass took the sRGB view from a leftover `D3DRS_SRGBWRITEENABLE`, the
/// pipeline and the attachment would disagree, which Metal treats as
/// undefined behaviour with the validation layer off.
#[test]
fn a_scoped_pass_with_srgb_write_off_attaches_the_base_view() {
    let mut s = fresh();
    let dst = tex(0x7F60);
    let dst_twin = tex(0x7F61);
    s.register_srgb_twin(dst_twin, dst);
    // The game leaves sRGB writes on over its own target.
    s.set_srgb_write_enabled(true);
    assert!(s.pass_srgb_write());

    // The scoped pass takes the device's binding, clears the state, and binds
    // the copy destination.
    let saved = s.take_color_attachments();
    s.set_srgb_write_enabled(false);
    s.set_color_render_target(dst, 128, 128, RT_FORMAT, RenderScale::IDENTITY);
    s.ensure_pass_open();
    assert!(!s.pass_srgb_write());
    assert_eq!(s.current_color_format(), RT_FORMAT);
    let pass = s.passes().last().expect("scoped pass");
    assert_eq!(pass.color_attachment_texture(), dst);
    assert_eq!(pass.color_format(), RT_FORMAT);

    // Putting the device's binding back leaves the next draw free to re-apply
    // the game's state.
    s.end_current_pass("test");
    s.restore_color_attachments(saved);
    s.set_srgb_write_enabled(true);
    assert!(s.pass_srgb_write());
    s.ensure_pass_open();
    assert_eq!(
        s.passes()
            .last()
            .expect("device pass")
            .color_attachment_texture(),
        backbuffer_srgb()
    );
}

/// Retiring the bound depth texture unbinds it and forgets every set naming it.
///
/// The standalone surface that owns the texture finalizes while the device
/// still has it bound, and the Metal texture is destroyed once the submit
/// seq gating it retires. A handle left in the session-wide sampled or
/// sampleable-depth sets would then answer for whatever Metal hands back at
/// the same address next.
#[test]
fn retiring_the_bound_depth_texture_unbinds_and_forgets_it() {
    let mut s = fresh();
    let shadow = tex(0x9100);
    s.set_depth_stencil_attachment(shadow, (256, 256), true, true);
    s.emit_command(Command::set_fragment_texture(shadow.raw(), 0));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.current_depth_texture(), shadow);
    assert!(s.is_depth_handle_sampleable(shadow));
    assert!(s.texture_sampled_this_frame(shadow));

    s.retire_depth_texture(shadow);

    assert!(s.current_depth_texture().is_null(), "attachment unbound");
    assert_eq!(s.current_depth_size(), (0, 0));
    assert!(!s.current_depth_has_stencil());
    assert!(!s.current_depth_is_sampleable());
    assert!(!s.is_depth_handle_sampleable(shadow));
    assert!(!s.texture_sampled_this_frame(shadow));
}

/// Retiring a texture that is not the bound one leaves the attachment alone.
///
/// A surface released while a different depth target is bound is the common
/// case, and the retire must not break the pass the game is building.
#[test]
fn retiring_an_unbound_depth_texture_keeps_the_binding() {
    let mut s = fresh();
    let other = tex(0x9200);
    s.emit_command(dummy_draw());
    let passes_before = s.passes().len();

    s.retire_depth_texture(other);

    assert_eq!(s.current_depth_texture(), depth());
    assert_eq!(s.passes().len(), passes_before);
    assert!(!s.current_pass_closed(), "the open pass survives");
}

// ── Multisample resolve ──

/// A frame whose back buffer is 4x multisampled.
fn fresh_multisampled() -> PassState {
    let mut s = PassState::new();
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: msaa_backbuffer(),
        backbuffer_msaa_srgb: msaa_backbuffer_srgb(),
        backbuffer_sample_count: 4,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: BB_SIZE,
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    s
}

fn msaa_backbuffer() -> MetalHandle<MTLTextureKind> {
    tex(0x1001)
}

fn msaa_backbuffer_srgb() -> MetalHandle<MTLTextureKind> {
    tex(0x1002)
}

#[test]
fn a_multisampled_pass_attaches_the_companion_and_resolves_into_the_twin() {
    let mut s = fresh_multisampled();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);

    let pass = &s.passes()[0];
    assert_eq!(
        pass.color_attachment_texture(),
        msaa_backbuffer(),
        "the pass renders into the multisampled companion"
    );
    assert_eq!(
        pass.color_texture(),
        backbuffer(),
        "the identity every rule keys on stays the single-sample twin"
    );
    assert_eq!(
        pass.color_resolve_texture(),
        backbuffer(),
        "and the twin is what the pass resolves into"
    );
}

#[test]
fn pass_binds_depth_answers_for_the_attachment_the_pass_takes() {
    // Every pipeline built for a pass reads this predicate to decide whether
    // to declare a depth and a stencil format, and Metal rejects a draw whose
    // pipeline declares one the pass has no attachment for. So it has to
    // answer for the attachment the pass takes, not for the binding the app
    // made: a single-sampled depth surface under a 4x target is dropped, and a
    // clear of its planes has nothing to paint.
    let mut s = fresh_multisampled();
    assert!(
        s.pass_binds_depth(),
        "the frame's own depth companion matches the target"
    );
    s.set_depth_stencil_attachment(tex(0x2001), BB_SIZE, false, true);
    assert!(
        !s.pass_binds_depth(),
        "a single-sampled depth surface under a 4x target is dropped"
    );
    assert_eq!(
        s.clear_depth(0),
        DepthClearOutcome::NoOp,
        "a depth clear has no attachment to paint"
    );
    assert_eq!(
        s.clear_stencil(0),
        StencilClearOutcome::NoOp,
        "and neither has a stencil clear"
    );
    s.set_depth_sample_count(4);
    assert!(
        s.pass_binds_depth(),
        "a depth surface at the target's own count binds again"
    );
    assert!(
        !PassState::new().pass_binds_depth(),
        "and no depth surface at all binds nothing"
    );
}

#[test]
fn only_the_last_pass_of_a_submission_takes_the_resolve() {
    // Two passes on the multisampled back buffer with an offscreen target in
    // between: the multisample content lives on across the middle pass and is
    // resolved once, at the end.
    let rt = tex(0x3000);
    let mut s = fresh_multisampled();
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
    s.set_color_msaa(msaa_backbuffer(), msaa_backbuffer_srgb(), 4);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);

    assert_eq!(s.passes().len(), 3);
    assert!(
        s.passes()[0].color_resolve_texture().is_null(),
        "the first pass keeps its multisample content for the third"
    );
    assert!(
        s.passes()[1].color_resolve_texture().is_null(),
        "the single-sampled target in between resolves nothing"
    );
    assert_eq!(
        s.passes()[2].color_resolve_texture(),
        backbuffer(),
        "the last use takes the resolve"
    );
}

#[test]
fn a_read_between_passes_pulls_the_resolve_forward() {
    // A `StretchRect` out of the multisampled target lands between the two
    // passes, so the twin has to be current before the second one runs.
    let rt = tex(0x3000);
    let mut s = fresh_multisampled();
    s.emit_command(dummy_draw());
    s.note_msaa_read(backbuffer());
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_color_msaa(msaa_backbuffer(), msaa_backbuffer_srgb(), 4);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);

    assert_eq!(
        s.passes()[0].color_resolve_texture(),
        backbuffer(),
        "the pass the read followed resolves"
    );
    assert_eq!(
        s.passes()[2].color_resolve_texture(),
        backbuffer(),
        "and the last use still resolves"
    );
}

#[test]
fn a_read_after_a_pending_clear_resolves_the_clear_only_pass() {
    // Clear(rt) with no pass open, StretchRect(rt -> dst), then a draw into rt
    // in the same submission. The `StretchRect` path materialises the clear
    // before it notes the read, so the clear-only pass takes the resolve and
    // the copy reads the cleared contents rather than whatever the twin held.
    let rt = tex(0x3400);
    let rt_msaa = tex(0x3401);
    let dst = tex(0x3402);
    let mut s = fresh();
    s.set_color_render_target(rt, BB_SIZE.0, BB_SIZE.1, BB_FORMAT, RenderScale::IDENTITY);
    s.set_color_msaa(rt_msaa, MetalHandle::NULL, 4);
    s.clear_color(0, 0, 255, 255);
    assert!(
        s.pending_color_clear().is_some(),
        "a first clear of an untouched target waits for a pass"
    );
    s.flush_pending_clears();
    s.note_msaa_read(rt);
    s.push_pending_leading_blit(copy_blit(rt, dst));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();
    s.finalize_store_actions(false);

    assert_eq!(s.passes().len(), 2, "the clear stays a pass of its own");
    assert!(
        matches!(s.passes()[0].color_load(), ColorLoad::Clear { .. }),
        "the first pass is the clear"
    );
    assert_eq!(
        s.passes()[0].color_resolve_texture(),
        rt,
        "the clear-only pass resolves before the copy reads rt"
    );
    assert_eq!(
        s.passes()[1].leading_blits().len(),
        1,
        "the copy runs ahead of the draw"
    );
    assert_eq!(
        s.passes()[1].color_resolve_texture(),
        rt,
        "and the last use still resolves"
    );
}

#[test]
fn a_depth_attachment_that_disagrees_on_samples_is_dropped() {
    // The depth surface is single-sampled while render target 0 is 4x: Metal
    // rejects such a pass outright, so the attachment goes rather than the
    // draw.
    let single = tex(0x2001);
    let mut s = fresh_multisampled();
    s.set_depth_stencil_attachment(single, BB_SIZE, false, false);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert!(
        s.passes()[0].depth_texture().is_null(),
        "the mismatched depth attachment is dropped"
    );

    // Declared at the matching count it binds normally.
    let mut s = fresh_multisampled();
    s.set_depth_stencil_attachment(single, BB_SIZE, false, false);
    s.set_depth_sample_count(4);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes()[0].depth_texture(), single);
}

/// A 4x frame with a depth and a stencil clear stashed, then a pass on a 1x target.
///
/// The pass drops the 4x depth surface, so the clears must stay pending
/// rather than be consumed by a pass that has no attachment to apply them to.
fn msaa_depth_clear_then_single_sampled_draw() -> PassState {
    let mut s = fresh_multisampled();
    assert_eq!(s.clear_depth(0x3f80_0000), DepthClearOutcome::Folded);
    assert_eq!(s.clear_stencil(7), StencilClearOutcome::Folded);
    s.set_color_render_target(tex(0x3000), 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    assert!(s.passes()[0].depth_texture().is_null());
    assert_eq!(s.pending_depth_clear(), Some(0x3f80_0000));
    assert_eq!(s.pending_stencil_clear, Some(7));
    s
}

#[test]
fn a_pass_that_drops_depth_leaves_the_depth_clear_pending() {
    let mut s = msaa_depth_clear_then_single_sampled_draw();
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_color_msaa(msaa_backbuffer(), msaa_backbuffer_srgb(), 4);
    s.emit_command(dummy_draw());
    let pass = &s.passes()[1];
    assert_eq!(pass.depth_texture(), depth());
    assert_eq!(
        pass.depth_load(),
        DepthLoad::Clear { value: 0x3f80_0000 },
        "the next pass that binds the depth surface applies the clear"
    );
    assert_eq!(pass.stencil_load(), StencilLoad::Clear { value: 7 });
}

#[test]
fn a_depth_clear_left_pending_flushes_onto_its_own_surface() {
    // Rebinding depth, submitting and retiring the surface all go through the
    // flush while render target 0 still disagrees on samples: the clear lands
    // in a depth-only pass on the surface it was issued for, never on the
    // next one bound.
    let expect_depth_only_clear = |s: &PassState| {
        let pass = s.passes().last().expect("a clear pass");
        assert!(pass.color_texture().is_null());
        assert_eq!(pass.depth_texture(), depth());
        assert_eq!(pass.depth_load(), DepthLoad::Clear { value: 0x3f80_0000 });
        assert_eq!(pass.stencil_load(), StencilLoad::Clear { value: 7 });
        assert!(s.pending_depth_clear().is_none());
        assert!(s.pending_stencil_clear.is_none());
        assert!(s.current_pass_closed());
    };

    let other = tex(0x4000);
    let mut s = msaa_depth_clear_then_single_sampled_draw();
    s.set_depth_stencil_attachment(other, (256, 256), false, true);
    expect_depth_only_clear(&s);
    s.emit_command(dummy_draw());
    let pass = s.passes().last().unwrap();
    assert_eq!(pass.depth_texture(), other);
    assert!(
        !matches!(pass.depth_load(), DepthLoad::Clear { .. }),
        "the replacement surface does not inherit the clear"
    );

    let mut s = msaa_depth_clear_then_single_sampled_draw();
    s.flush_pending_clears();
    expect_depth_only_clear(&s);

    let mut s = msaa_depth_clear_then_single_sampled_draw();
    s.retire_depth_texture(depth());
    expect_depth_only_clear(&s);
}

#[test]
fn a_resolving_clear_only_pass_survives_the_cull() {
    // The pass has no draw and its multisample content is dead, but the
    // resolve still writes the twin every later reader looks at.
    let mut s = fresh_multisampled();
    s.clear_color(1, 1, 1, 1);
    s.ensure_pass_open();
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    s.strip_dead_color_in_clear_only_passes();
    s.cull_dead_clear_only_passes();

    assert_eq!(s.passes().len(), 1, "the resolving pass is not dead work");
    assert_eq!(s.passes()[0].color_resolve_texture(), backbuffer());
    assert_eq!(
        s.passes()[0].color_attachment_texture(),
        msaa_backbuffer(),
        "and it keeps its colour attachment"
    );
}

/// `D3DRS_SRGBWRITEENABLE` on a multisampled target attaches both sRGB views.
///
/// Metal takes the resolve destination's pixel format from the attachment's,
/// so a pass that renders through the companion's twin has to resolve into
/// the single-sample twin rather than into the base texture.
#[test]
fn a_multisampled_srgb_pass_attaches_and_resolves_through_the_twins() {
    let mut s = fresh_multisampled();
    s.set_srgb_write_enabled(true);
    assert!(s.pass_srgb_write());
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);

    let pass = &s.passes()[0];
    assert_eq!(
        pass.color_attachment_texture(),
        msaa_backbuffer_srgb(),
        "the pass renders through the companion's sRGB view"
    );
    assert_eq!(
        pass.color_resolve_texture(),
        backbuffer_srgb(),
        "and resolves into the single-sample view of the same format"
    );
    assert_eq!(
        pass.color_texture(),
        backbuffer(),
        "identity still keys on the base texture"
    );
}

/// A companion without an sRGB twin keeps the whole pass on the shader path.
///
/// Attaching the linear companion and resolving into the sRGB twin is a
/// format mismatch Metal rejects, so the encode falls back to the pixel
/// shader's OETF variant exactly as a target with no twin at all does.
#[test]
fn a_multisampled_target_without_a_companion_twin_keeps_the_linear_attachment() {
    let mut s = fresh_multisampled();
    s.set_color_msaa(msaa_backbuffer(), MetalHandle::NULL, 4);
    s.set_srgb_write_enabled(true);
    assert!(!s.pass_srgb_write());
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);

    let pass = &s.passes()[0];
    assert_eq!(pass.color_attachment_texture(), msaa_backbuffer());
    assert_eq!(pass.color_resolve_texture(), backbuffer());
}

// ── Colour strips drop every view of attachment 0 ──

/// Rule H on a pass that encodes through the sRGB twin attaches no colour at all.
///
/// The pass's pipelines are rewritten to the no-colour variant, so an
/// attachment left behind through the twin view fails Metal's
/// pipeline-versus-render-pass format validation.
#[test]
fn rule_h_strip_drops_the_srgb_twin_view() {
    let mut s = fresh();
    s.set_srgb_write_enabled(true);
    assert!(s.pass_srgb_write());
    s.note_draw_color_write_mask(0);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes()[0].color_attachment_texture(), backbuffer_srgb());
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));

    s.strip_color_from_no_color_draw_passes(&alt);

    assert_eq!(s.passes()[0].color_texture(), MetalHandle::NULL);
    assert_eq!(
        s.passes()[0].color_attachment_texture(),
        MetalHandle::NULL,
        "the stripped pass must not attach the sRGB twin"
    );
}

/// Rule H on a multisampled pass that does not take the resolve attaches no colour at all.
///
/// Leaving the companion attached with `DontCare` store would also discard
/// the samples an earlier pass stored for the later pass that loads them.
#[test]
fn rule_h_strip_drops_the_multisampled_companion() {
    let rt = tex(0x3000);
    let mut s = fresh_multisampled();
    s.note_draw_color_write_mask(0);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.note_draw_color_write_mask(0xF);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_color_msaa(msaa_backbuffer(), msaa_backbuffer_srgb(), 4);
    s.note_draw_color_write_mask(0xF);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 3);
    assert!(
        s.passes()[0].color_resolve_texture().is_null(),
        "the last pass on the back buffer takes the resolve"
    );
    assert_eq!(s.passes()[0].color_attachment_texture(), msaa_backbuffer());
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));

    s.strip_color_from_no_color_draw_passes(&alt);

    assert_eq!(s.passes()[0].color_texture(), MetalHandle::NULL);
    assert_eq!(
        s.passes()[0].color_attachment_texture(),
        MetalHandle::NULL,
        "the stripped pass must not attach the multisampled companion"
    );
    assert_eq!(
        s.passes()[2].color_attachment_texture(),
        msaa_backbuffer(),
        "the colour-writing pass keeps its attachment"
    );
}

/// Rule G on a clear-only pass that encodes through the sRGB twin attaches no colour at all.
#[test]
fn rule_g_strip_drops_the_srgb_twin_view() {
    let cascade_color = tex(0x3000);
    let cascade_twin = tex(0x3001);
    let cascade_d0 = tex(0x9000);
    let cascade_d1 = tex(0x9100);
    let mut s = fresh();
    s.register_srgb_twin(cascade_twin, cascade_color);
    s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT, RenderScale::IDENTITY);
    s.set_srgb_write_enabled(true);
    assert!(s.pass_srgb_write());
    s.set_depth_stencil_attachment(cascade_d0, BB_SIZE, false, false);
    s.clear_color(1, 2, 3, 4);
    s.clear_depth(f32::to_bits(1.0));
    s.set_depth_stencil_attachment(cascade_d1, BB_SIZE, false, false);
    s.clear_color(1, 2, 3, 4);
    s.clear_depth(f32::to_bits(1.0));
    s.emit_command(dummy_draw());
    s.set_srgb_write_enabled(false);
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_depth_stencil_attachment(depth(), BB_SIZE, false, false);
    s.emit_command(Command::set_fragment_texture(cascade_d0.raw(), 4));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    let before = s
        .passes()
        .iter()
        .find(|p| p.depth_texture() == cascade_d0)
        .expect("cascade_d0 pass");
    assert_eq!(before.color_attachment_texture(), cascade_twin);
    assert_eq!(before.color_store(), StoreAction::DontCare);

    s.strip_dead_color_in_clear_only_passes();

    let stripped = s
        .passes()
        .iter()
        .find(|p| p.depth_texture() == cascade_d0)
        .expect("cascade_d0 pass");
    assert_eq!(stripped.color_texture(), MetalHandle::NULL);
    assert_eq!(
        stripped.color_attachment_texture(),
        MetalHandle::NULL,
        "the stripped pass must not attach the sRGB twin"
    );
}

/// Rule G on a multisampled clear-only pass that does not take the resolve attaches no colour.
#[test]
fn rule_g_strip_drops_the_multisampled_companion() {
    let second_depth = tex(0x9000);
    let mut s = fresh_multisampled();
    s.clear_color(1, 1, 1, 1);
    s.clear_depth(f32::to_bits(1.0));
    s.ensure_pass_open();
    s.set_depth_stencil_attachment(second_depth, BB_SIZE, false, false);
    s.set_depth_sample_count(4);
    s.clear_color(1, 1, 1, 1);
    s.note_draw_color_write_mask(0xF);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    assert_eq!(s.passes().len(), 2);
    assert!(s.passes()[0].color_resolve_texture().is_null());
    assert_eq!(s.passes()[0].color_store(), StoreAction::DontCare);
    assert_eq!(s.passes()[0].color_attachment_texture(), msaa_backbuffer());

    s.strip_dead_color_in_clear_only_passes();

    assert_eq!(s.passes()[0].color_texture(), MetalHandle::NULL);
    assert_eq!(
        s.passes()[0].color_attachment_texture(),
        MetalHandle::NULL,
        "the stripped pass must not attach the multisampled companion"
    );
}

// ── Depth transfers (RESZ and the depth StretchRect resolve) ──

/// A depth transfer out of `source` into `destination`, as the encoder queues it.
fn depth_transfer(
    source: MetalHandle<MTLTextureKind>,
    destination: MetalHandle<MTLTextureKind>,
) -> BlitCommand {
    let mut blit = BlitCommand::copy_texture_to_texture_full_mip(
        source.raw(),
        destination.raw(),
        0,
        BB_SIZE.0,
        BB_SIZE.1,
    );
    blit.cmd = BlitCommandType::TransferDepth as u32;
    blit
}

#[test]
fn a_depth_transfer_runs_between_the_pass_that_wrote_its_source_and_the_next() {
    // The transfer is a leading blit of the pass that opens after it, so it
    // reads what the draws before it left in the multisampled depth and
    // anything after it sees what it wrote.
    let destination = tex(0x4000);
    let mut s = fresh_multisampled();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.push_pending_leading_blit(depth_transfer(depth(), destination));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");

    assert_eq!(s.passes().len(), 2);
    assert!(s.passes()[0].leading_blits().is_empty());
    let blits = s.passes()[1].leading_blits();
    assert_eq!(blits.len(), 1, "the transfer leads the pass after the draw");
    assert_eq!(blits[0].cmd, BlitCommandType::TransferDepth as u32);
    assert_eq!(blits[0].src_handle, depth().raw());
    assert_eq!(blits[0].dst_handle, destination.raw());
}

#[test]
fn a_depth_transfer_keeps_the_store_of_the_pass_that_wrote_its_source() {
    // Rule B drops the depth store on a depth texture's last pass of the
    // frame. The transfer reads that depth from memory after the pass, so the
    // store has to survive.
    let destination = tex(0x4001);
    let mut s = fresh_multisampled();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.push_pending_leading_blit(depth_transfer(depth(), destination));
    s.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_load_actions();
    s.finalize_store_actions(false);

    assert_eq!(s.passes()[0].depth_texture(), depth());
    assert_eq!(
        s.passes()[0].depth_store(),
        StoreAction::Store,
        "the transfer reads the samples the draw left"
    );
}

#[test]
fn a_depth_transfer_destination_opens_a_later_pass_on_its_contents() {
    // Rule A would discard the destination's first use this frame; the
    // transfer wrote it, so the bind that follows has to load.
    let destination = tex(0x4002);
    let mut s = fresh_multisampled();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.push_pending_leading_blit(depth_transfer(depth(), destination));
    s.set_depth_stencil_attachment(destination, BB_SIZE, false, false);
    s.set_depth_sample_count(4);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_load_actions();

    assert!(s.texture_written_by_blit_this_frame(destination));
    assert_eq!(
        s.passes()[1].depth_load(),
        DepthLoad::Load,
        "the transferred contents are loaded, not discarded"
    );
}

#[test]
fn a_clear_only_pass_does_not_fold_past_a_depth_transfer_out_of_its_target() {
    // Clear(ds) → transfer out of ds → draw against ds. Rule E may not move
    // the clear into the draw pass's load action: the transfer runs ahead of
    // that pass's render encoder, so it would read the depth from before the
    // clear.
    let other_rt = tex(0x3000);
    let source = tex(0x4006);
    let destination = tex(0x4007);
    let mut s = fresh();
    s.set_color_render_target(other_rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.set_depth_stencil_attachment(source, (256, 256), false, false);
    s.clear_depth(f32::to_bits(1.0));
    s.push_leading_blit_after_clears(depth_transfer(source, destination), "test");
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();

    assert_eq!(
        s.passes().len(),
        2,
        "the clear-only pass stays where it was"
    );
    assert!(
        matches!(s.passes()[0].depth_load(), DepthLoad::Clear { .. }),
        "the clear lands before the transfer"
    );
    assert!(
        matches!(s.passes()[1].depth_load(), DepthLoad::Load),
        "the pass the transfer leads loads the cleared depth"
    );
}

/// Run the load/store rules in the order the encoder applies them at submit.
///
/// Rule I and Rule H, which needs a pipeline side map, are left out.
fn apply_submit_rules(s: &mut PassState) {
    s.coalesce_clear_only_passes();
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    s.strip_dead_color_in_clear_only_passes();
    s.cull_dead_clear_only_passes();
    s.merge_adjacent_identical_passes();
}

#[test]
fn a_pending_depth_clear_lands_before_a_depth_transfer_out_of_its_surface() {
    // Clear(ZBUFFER) on the bound multisampled depth with no pass open, then
    // RESZ with no draw in between: the transfer reads the cleared depth, so
    // the clear is recorded as a pass ahead of the one the transfer leads.
    let cleared = f32::to_bits(0.5);
    let destination = tex(0x400a);
    let mut s = fresh_multisampled();
    assert_eq!(s.clear_depth(cleared), DepthClearOutcome::Folded);
    assert_eq!(s.pending_depth_clear(), Some(cleared));
    s.push_leading_blit_after_clears(depth_transfer(depth(), destination), "test");
    assert!(s.pending_depth_clear().is_none(), "nothing is left pending");
    s.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    apply_submit_rules(&mut s);

    assert_eq!(s.passes().len(), 2);
    let clear = &s.passes()[0];
    assert_eq!(clear.depth_texture(), depth());
    assert_eq!(clear.depth_load(), DepthLoad::Clear { value: cleared });
    assert_eq!(
        clear.depth_store(),
        StoreAction::Store,
        "the cleared depth reaches memory for the transfer to read"
    );
    assert!(clear.leading_blits().is_empty());
    let blits = s.passes()[1].leading_blits();
    assert_eq!(blits.len(), 1, "the transfer runs after the clear");
    assert_eq!(blits[0].cmd, BlitCommandType::TransferDepth as u32);
    assert_eq!(blits[0].src_handle, depth().raw());
}

#[test]
fn a_depth_clear_the_target_cannot_carry_lands_before_a_depth_transfer() {
    // Render target 0 disagrees with the depth surface on samples, so the
    // pending clear lands as a depth-only pass on that surface, still ahead
    // of the transfer out of it.
    let destination = tex(0x400b);
    let mut s = msaa_depth_clear_then_single_sampled_draw();
    s.push_leading_blit_after_clears(depth_transfer(depth(), destination), "test");
    assert!(s.pending_depth_clear().is_none(), "nothing is left pending");
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    apply_submit_rules(&mut s);

    let transfer_pass = s
        .passes()
        .iter()
        .position(|p| !p.leading_blits().is_empty())
        .expect("the pass the transfer leads");
    let clear_pass = s
        .passes()
        .iter()
        .position(|p| p.depth_texture() == depth())
        .expect("the depth-only clear pass");
    assert!(clear_pass < transfer_pass, "the clear runs first");
    let clear = &s.passes()[clear_pass];
    assert!(clear.color_texture().is_null(), "a depth-only pass");
    assert_eq!(clear.depth_load(), DepthLoad::Clear { value: 0x3f80_0000 });
    assert_eq!(clear.depth_store(), StoreAction::Store);
}

#[test]
fn a_clear_only_pass_does_not_fold_past_a_depth_transfer_into_its_target() {
    // Clear(ds) → transfer into ds → depth-test against ds. The transfer lands
    // between them, and a clear moved past it would wipe what it wrote.
    let other_rt = tex(0x3000);
    let source = tex(0x4008);
    let resolved = tex(0x4009);
    let mut s = fresh();
    s.set_depth_stencil_attachment(resolved, BB_SIZE, false, false);
    s.clear_depth(f32::to_bits(1.0));
    s.set_color_render_target(other_rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.set_depth_stencil_attachment(source, BB_SIZE, false, false);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.push_pending_leading_blit(depth_transfer(source, resolved));
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_depth_stencil_attachment(resolved, BB_SIZE, false, false);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();

    assert_eq!(
        s.passes().len(),
        3,
        "the clear-only pass stays where it was"
    );
    assert!(
        matches!(s.passes()[0].depth_load(), DepthLoad::Clear { .. }),
        "the clear lands before the transfer"
    );
    assert!(
        matches!(s.passes()[2].depth_load(), DepthLoad::Load),
        "the draw after the transfer loads what it wrote"
    );
}

/// A multisampled extra-target binding sized like the back buffer.
fn msaa_slot(
    texture: MetalHandle<MTLTextureKind>,
    msaa_texture: MetalHandle<MTLTextureKind>,
    size: (u32, u32),
) -> ExtraColorSlot {
    ExtraColorSlot {
        texture,
        msaa_texture,
        msaa_srgb_texture: MetalHandle::NULL,
        sample_count: 4,
        subresource: 0,
        size,
        logical_size: size,
        format: BB_FORMAT,
        scale: RenderScale::IDENTITY,
        has_alpha: false,
    }
}

#[test]
fn a_clear_only_pass_does_not_fold_past_a_colour_resolve_into_its_target() {
    // Clear(rt) -> a pass that resolves its multisampled companion into rt ->
    // draw into rt. Rule E may not move the clear into the last pass's load
    // action: the resolve lands between them, and a clear moved past it would
    // wipe what it wrote.
    let rt = tex(0x3200);
    let rt_msaa = tex(0x3201);
    let scene = tex(0x3202);
    let scene_msaa = tex(0x3203);
    let mut s = fresh();
    s.set_color_render_target(rt, BB_SIZE.0, BB_SIZE.1, BB_FORMAT, RenderScale::IDENTITY);
    s.set_color_msaa(rt_msaa, MetalHandle::NULL, 4);
    s.clear_color(1, 2, 3, 4);
    // Switching the target materialises the clear as a pass of its own.
    s.set_color_render_target(
        scene,
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_color_msaa(scene_msaa, MetalHandle::NULL, 4);
    s.set_extra_color_render_target(1, Some(msaa_slot(rt, rt_msaa, BB_SIZE)));
    s.emit_command(dummy_draw());
    // The read a `StretchRect` out of rt performs takes the resolve on the
    // pass that last rendered into its companion.
    s.note_msaa_read(rt);
    s.set_extra_color_render_target(1, None);
    s.set_color_render_target(rt, BB_SIZE.0, BB_SIZE.1, BB_FORMAT, RenderScale::IDENTITY);
    s.set_color_msaa(rt_msaa, MetalHandle::NULL, 4);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(
        s.passes()[0].color_attachment_texture(),
        rt_msaa,
        "the clear paints the companion the resolve reads"
    );
    assert_eq!(
        s.passes()[1].extra_color()[0].resolve_texture(),
        rt,
        "the middle pass resolves into the cleared target"
    );
    s.coalesce_clear_only_passes();

    assert_eq!(
        s.passes().len(),
        3,
        "the clear-only pass stays where it was"
    );
    assert!(
        matches!(s.passes()[0].color_load(), ColorLoad::Clear { .. }),
        "the clear lands before the resolve"
    );
    assert!(
        matches!(s.passes()[2].color_load(), ColorLoad::Load),
        "the draw after the resolve loads what it wrote"
    );
}

#[test]
fn a_target_switch_flushes_pending_clears_onto_the_outgoing_multisampled_attachments() {
    // Clear(TARGET | ZBUFFER) with no pass open, then SetRenderTarget: the
    // clear-only pass paints the target the clear was issued against, so it
    // carries the back buffer's companion and its 4x depth surface.
    let offscreen = tex(0x3300);
    let mut s = fresh_multisampled();
    s.clear_color(1, 2, 3, 4);
    s.clear_depth(f32::to_bits(1.0));
    s.set_color_render_target(
        offscreen,
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        RenderScale::IDENTITY,
    );

    assert_eq!(s.passes().len(), 1, "the clears materialise as a pass");
    let pass = &s.passes()[0];
    assert_eq!(pass.color_texture(), backbuffer());
    assert_eq!(
        pass.color_attachment_texture(),
        msaa_backbuffer(),
        "the colour clear paints the companion the frame resolves"
    );
    assert!(matches!(pass.color_load(), ColorLoad::Clear { .. }));
    assert_eq!(
        pass.depth_texture(),
        depth(),
        "the depth surface matches the companion's sample count"
    );
    assert!(matches!(pass.depth_load(), DepthLoad::Clear { .. }));
}

#[test]
fn a_depth_switch_flushes_the_pending_depth_clear_onto_the_outgoing_multisampled_surface() {
    let other = tex(0x3301);
    let mut s = fresh_multisampled();
    s.clear_depth(f32::to_bits(1.0));
    s.set_depth_stencil_attachment(other, BB_SIZE, false, false);
    s.set_depth_sample_count(4);

    assert_eq!(s.passes().len(), 1, "the clear materialises as a pass");
    let pass = &s.passes()[0];
    assert_eq!(
        pass.depth_texture(),
        depth(),
        "the clear lands on the surface it was issued against"
    );
    assert!(matches!(pass.depth_load(), DepthLoad::Clear { .. }));
    assert_eq!(s.current_depth_texture(), other);
    assert_eq!(s.current_depth_sample_count(), 4);
}

#[test]
fn rebinding_render_target_0_keeps_multisampled_extras_in_the_pass() {
    let extra = tex(0x3302);
    let extra_msaa = tex(0x3303);
    let scene = tex(0x3304);
    let scene_msaa = tex(0x3305);
    let mut s = fresh_multisampled();
    s.set_extra_color_render_target(1, Some(msaa_slot(extra, extra_msaa, BB_SIZE)));
    assert_eq!(s.extra_present_mask(), 0b001);
    s.emit_command(dummy_draw());

    // Games re-assert the bound target between scenes.
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_color_msaa(msaa_backbuffer(), msaa_backbuffer_srgb(), 4);
    assert_eq!(
        s.extra_present_mask(),
        0b001,
        "a redundant rebind keeps the 4x extra"
    );
    assert_eq!(s.passes().len(), 1);
    assert!(!s.current_pass_closed(), "and keeps the pass open");

    s.set_color_render_target(
        scene,
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_color_msaa(scene_msaa, MetalHandle::NULL, 4);
    assert_eq!(
        s.extra_present_mask(),
        0b001,
        "a 4x target 0 readmits the 4x extra"
    );
    s.emit_command(dummy_draw());
    let pass = s.passes().last().expect("the scene pass");
    assert_eq!(pass.color_attachment_texture(), scene_msaa);
    assert_eq!(pass.extra_color()[0].texture(), extra);
}

fn uneven_command_frame(s: &mut PassState) {
    for pass in 0..37 {
        for _ in 0..(65 << (pass % 6)) {
            s.emit_command(dummy_draw());
        }
        s.end_current_pass("test");
    }
}

fn command_allocations(passes: &[Pass]) -> Vec<(*const Command, usize)> {
    passes
        .iter()
        .map(|p| (p.commands.as_ptr(), p.commands.capacity()))
        .collect()
}

#[test]
fn command_pool_reuses_37_uneven_passes_on_reset() {
    let mut s = fresh();
    uneven_command_frame(&mut s);
    let allocations = command_allocations(s.passes());
    let capacity = PassState::cmd_vec_capacity_bytes(s.passes());
    let warmup_copies = s.take_cmd_vec_realloc_bytes();
    assert!(warmup_copies > 0);
    for _ in 0..8 {
        reset_test_frame(&mut s);
        assert_eq!(s.command_vec_pool.len(), 37);
        uneven_command_frame(&mut s);
        assert_eq!(command_allocations(s.passes()), allocations);
        assert_eq!(s.take_cmd_vec_realloc_bytes(), 0);
        assert_eq!(PassState::cmd_vec_capacity_bytes(s.passes()), capacity);
    }
    println!(
        "37 passes: retained={capacity} bytes, warmup growth copy estimate={warmup_copies} bytes, steady growth copy estimate=0, steady command-vector allocations=0"
    );
}

#[test]
fn command_pool_reuses_detached_passes_and_accounts_only_the_payload() {
    let mut s = fresh();
    uneven_command_frame(&mut s);
    let mut payload = s.take_finished_passes();
    let allocations = command_allocations(&payload);
    let capacity = PassState::cmd_vec_capacity_bytes(&payload);
    assert!(capacity > 0);
    assert_eq!(PassState::cmd_vec_capacity_bytes(s.passes()), 0);
    reset_test_frame(&mut s);
    assert!(s.command_vec_pool.is_empty());
    for _ in 0..8 {
        let pass_capacity = payload.capacity();
        s.recycle_passes(&mut payload);
        assert_eq!(payload.capacity(), pass_capacity);
        assert!(payload.is_empty());
        uneven_command_frame(&mut s);
        assert_eq!(command_allocations(s.passes()), allocations);
        assert_eq!(s.take_cmd_vec_realloc_bytes(), 0);
        payload = s.take_finished_passes();
        assert_eq!(PassState::cmd_vec_capacity_bytes(&payload), capacity);
        reset_test_frame(&mut s);
    }
}

#[test]
fn command_pool_keeps_two_outstanding_payloads_disjoint() {
    let mut s = fresh();
    uneven_command_frame(&mut s);
    let mut first = s.take_finished_passes();
    let first_allocations = command_allocations(&first);
    reset_test_frame(&mut s);
    uneven_command_frame(&mut s);
    let mut second = s.take_finished_passes();
    let second_allocations = command_allocations(&second);
    reset_test_frame(&mut s);
    uneven_command_frame(&mut s);
    let third_allocations = command_allocations(s.passes());
    assert_eq!(command_allocations(&first), first_allocations);
    assert_eq!(command_allocations(&second), second_allocations);
    for (left, right) in [
        (&first_allocations, &second_allocations),
        (&first_allocations, &third_allocations),
        (&second_allocations, &third_allocations),
    ] {
        assert!(
            left.iter()
                .all(|(ptr, _)| right.iter().all(|(other, _)| ptr != other))
        );
    }
    let mut third = s.take_finished_passes();
    reset_test_frame(&mut s);
    for _ in 0..8 {
        s.recycle_passes(&mut first);
        uneven_command_frame(&mut s);
        assert_eq!(command_allocations(s.passes()), first_allocations);
        assert_eq!(command_allocations(&second), second_allocations);
        assert_eq!(command_allocations(&third), third_allocations);
        assert_eq!(s.take_cmd_vec_realloc_bytes(), 0);
        first = s.take_finished_passes();
        reset_test_frame(&mut s);
        s.recycle_passes(&mut second);
        uneven_command_frame(&mut s);
        assert_eq!(command_allocations(s.passes()), second_allocations);
        assert_eq!(command_allocations(&first), first_allocations);
        assert_eq!(command_allocations(&third), third_allocations);
        assert_eq!(s.take_cmd_vec_realloc_bytes(), 0);
        second = s.take_finished_passes();
        reset_test_frame(&mut s);
        s.recycle_passes(&mut third);
        uneven_command_frame(&mut s);
        assert_eq!(command_allocations(s.passes()), third_allocations);
        assert_eq!(command_allocations(&first), first_allocations);
        assert_eq!(command_allocations(&second), second_allocations);
        assert_eq!(s.take_cmd_vec_realloc_bytes(), 0);
        third = s.take_finished_passes();
        reset_test_frame(&mut s);
    }
    let capacity = PassState::cmd_vec_capacity_bytes(&first)
        + PassState::cmd_vec_capacity_bytes(&second)
        + PassState::cmd_vec_capacity_bytes(&third);
    s.recycle_passes(&mut first);
    s.recycle_passes(&mut second);
    s.recycle_passes(&mut third);
    assert_eq!(s.command_vec_pool.len(), 111);
    assert_eq!(
        s.command_vec_pool
            .iter()
            .map(|v| v.capacity() as u64 * size_of::<Command>() as u64)
            .sum::<u64>(),
        capacity
    );
    println!(
        "Two outstanding payloads plus a live 37-pass list: retained={capacity} bytes, steady growth copy estimate=0, steady command-vector allocations=0"
    );
}

#[test]
fn command_pool_retires_dead_clears_without_reordering_survivors() {
    let mut s = fresh();
    for _ in 0..6 {
        s.ensure_pass_open();
        s.end_current_pass("test");
    }
    s.passes[0].commands.push(dummy_draw());
    s.passes[1].color_store = StoreAction::DontCare;
    s.passes[1].depth_store = StoreAction::DontCare;
    s.passes[2].leading_blits.push(dummy_blit());
    s.passes[3].color_store = StoreAction::DontCare;
    s.passes[3].depth_store = StoreAction::DontCare;
    s.passes[4].color_load = ColorLoad::Clear {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    s.passes[4].color_store = StoreAction::Store;
    s.passes[5].color_resolve_texture = tex(0x4000);
    let original = command_allocations(s.passes());
    s.cull_dead_clear_only_passes();
    assert_eq!(
        command_allocations(s.passes()),
        [original[0], original[2], original[4], original[5]]
    );
    assert_eq!(s.command_vec_pool.len(), 2);
    let pooled: Vec<_> = s
        .command_vec_pool
        .iter()
        .map(|v| (v.as_ptr(), v.capacity()))
        .collect();
    assert!(pooled.contains(&original[1]));
    assert!(pooled.contains(&original[3]));
    assert!(s.command_vec_pool.iter().all(Vec::is_empty));
}

#[test]
fn command_pool_does_not_park_unallocated_blit_pass_vectors() {
    let mut s = fresh();
    s.ensure_pass_open();
    s.end_current_pass("test");
    let allocation = command_allocations(s.passes());
    s.ensure_pass_open();
    s.end_current_pass("test");
    s.passes[1].commands = Vec::new();
    s.passes[1].leading_blits.push(dummy_blit());
    let mut payload = s.take_finished_passes();
    s.recycle_passes(&mut payload);
    assert_eq!(s.command_vec_pool.len(), 1);
    reset_test_frame(&mut s);
    s.ensure_pass_open();
    assert_eq!(command_allocations(s.passes()), allocation);
}

fn mixed_command_frame(s: &mut PassState) {
    for count in [65, 513, 129] {
        for _ in 0..count {
            s.emit_command(dummy_draw());
        }
        s.end_current_pass("test");
    }
    s.pending_depth_clear = Some(f32::to_bits(1.0));
    s.push_depth_clear_pass();
    for target in [tex(0x5000), tex(0x6000)] {
        s.push_upload_pass(
            &UploadPassTarget {
                texture: target,
                subresource: (0, 0),
                size: BB_SIZE,
                format: BB_FORMAT,
                rect: (0, 0, BB_SIZE.0, BB_SIZE.1),
            },
            &[dummy_draw()],
            Vec::new(),
        );
    }
    assert_eq!(s.passes.len(), 6);
    assert_eq!(s.passes[0].color_texture, tex(0x5000));
    assert_eq!(s.passes[1].color_texture, tex(0x6000));
    assert!(s.passes[5].commands.is_empty());
    assert!(matches!(s.passes[5].depth_load, DepthLoad::Clear { .. }));
}

#[test]
fn upload_passes_keep_every_volume_depth_plane_and_level() {
    let mut s = fresh();
    let planes = [0, 7, 8, 255, 256, 2047];
    for plane in planes {
        s.push_upload_pass(
            &UploadPassTarget {
                texture: tex(0x5000),
                subresource: (plane, 1),
                size: (2, 2),
                format: BB_FORMAT,
                rect: (0, 0, 2, 2),
            },
            &[dummy_draw()],
            Vec::new(),
        );
    }
    assert_eq!(s.upload_pass_count(), planes.len());
    for (pass, plane) in s.passes().iter().zip(planes) {
        assert_eq!((pass.color_slice(), pass.color_level()), (plane, 1));
    }
}

#[test]
fn command_pool_converges_with_head_uploads_and_depth_clear_passes() {
    let mut s = fresh();
    // Uploads move to the front after borrowing vectors in recording order,
    // so their capacity alignment can take more than one frame to converge.
    for _ in 0..12 {
        mixed_command_frame(&mut s);
        reset_test_frame(&mut s);
    }
    mixed_command_frame(&mut s);
    let mut allocations = command_allocations(s.passes());
    allocations.sort_unstable();
    for _ in 0..12 {
        reset_test_frame(&mut s);
        assert_eq!(s.command_vec_pool.len(), 6);
        mixed_command_frame(&mut s);
        let mut next = command_allocations(s.passes());
        next.sort_unstable();
        assert_eq!(next, allocations);
        assert_eq!(s.take_cmd_vec_realloc_bytes(), 0);
    }
}

#[test]
fn snapshot_bytes_identity_and_equal_distinct_tokens() {
    let a = vec![1, 2, 3, 4];
    let b = a.clone();
    let mut cache = super::SnapshotBytesCache::new();
    assert!(cache.changed(a.as_slice()));
    assert!(!cache.changed(a.as_slice()));
    assert!(!cache.changed(b.as_slice()));
    assert!(core::ptr::eq(cache.snapshot.unwrap(), b.as_slice()));
    assert!(!cache.changed(b.as_slice()));
}

#[test]
fn snapshot_bytes_changes_length_and_empty_preserves_binding() {
    let bytes = [1, 2, 3, 4];
    let changed = [1, 2, 3, 5];
    let mut cache = super::SnapshotBytesCache::new();
    assert!(!cache.changed(&bytes[..0]));
    assert!(cache.changed(bytes.as_slice()));
    assert!(cache.changed(changed.as_slice()));
    assert!(cache.changed(&changed[..3]));
    assert!(!cache.changed(&bytes[..0]));
    assert!(!cache.changed(&changed[..3]));
    assert!(cache.changed(changed.as_slice()));
}

#[test]
fn snapshot_bytes_float_equality_is_bitwise() {
    let nan = f32::from_bits(0x7fc0_0001).to_ne_bytes();
    let same_nan = nan;
    let other_nan = f32::from_bits(0x7fc0_0002).to_ne_bytes();
    let positive_zero = 0.0_f32.to_ne_bytes();
    let negative_zero = (-0.0_f32).to_ne_bytes();
    let mut cache = super::SnapshotBytesCache::new();
    assert!(cache.changed(nan.as_slice()));
    assert!(!cache.changed(same_nan.as_slice()));
    assert!(cache.changed(other_nan.as_slice()));
    assert!(cache.changed(positive_zero.as_slice()));
    assert!(cache.changed(negative_zero.as_slice()));
}

#[test]
fn snapshot_bytes_reset_rebinds_reused_address() {
    let mut bytes = std::rc::Rc::<[u8]>::from([1, 2, 3, 4]);
    let address = bytes.as_ptr();
    let mut cache = super::SnapshotBytesCache::new();
    assert!(cache.changed(std::rc::Rc::clone(&bytes)));
    cache.reset();
    assert!(cache.snapshot.is_none());
    std::rc::Rc::get_mut(&mut bytes).expect("reset released the snapshot")[0] = 5;
    assert_eq!(bytes.as_ptr(), address);
    assert!(cache.changed(std::rc::Rc::clone(&bytes)));
    assert!(!cache.changed(bytes));
}

/// Mixed uploads keep their blits in API order without closing the application's pass.
#[test]
fn upload_prefix_preserves_order_through_pass_optimization() {
    let mut s = fresh();
    s.emit_command(dummy_draw());
    let application_commands = s.passes()[0].commands().as_ptr();
    let old = tex(0x8000);
    let fresh_texture = tex(0x9000);
    for (target, blits) in [
        (
            old,
            vec![BlitCommand::notify_buffer_did_modify_range(0x7000, 0, 16)],
        ),
        (fresh_texture, vec![copy_blit(old, fresh_texture)]),
    ] {
        s.push_upload_pass(
            &UploadPassTarget {
                texture: target,
                subresource: (0, 1),
                size: (2, 2),
                format: BB_FORMAT,
                rect: (0, 0, 2, 2),
            },
            &[dummy_draw()],
            blits,
        );
        assert!(!s.current_pass_closed());
        assert_eq!(s.current_color_texture(), backbuffer());
        assert_eq!(s.current_depth_texture(), depth());
        assert_eq!(s.effective_viewport(), (0, 0, BB_SIZE.0, BB_SIZE.1));
        s.emit_command(dummy_draw());
    }
    assert_eq!(s.upload_pass_count(), 2);
    assert!(!s.texture_written_by_blit_this_frame(old));
    assert!(!s.texture_written_by_blit_this_frame(fresh_texture));
    assert_eq!(s.passes()[2].commands().as_ptr(), application_commands);
    assert_eq!(s.passes()[2].commands().len(), 4);
    let prefix_commands: Vec<_> = s.passes()[..2]
        .iter()
        .map(|pass| pass.commands().as_ptr())
        .collect();
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    s.strip_dead_color_in_clear_only_passes();
    s.strip_color_from_no_color_draw_passes(&FxHashMap::default());
    s.cull_dead_clear_only_passes();
    assert_eq!(s.passes().len(), 3);
    for (index, texture) in [old, fresh_texture].into_iter().enumerate() {
        let pass = &s.passes()[index];
        assert_eq!(pass.color_texture(), texture);
        assert_eq!(pass.color_level(), 1);
        assert_eq!(pass.commands().as_ptr(), prefix_commands[index]);
        assert_eq!(pass.color_store(), StoreAction::Store);
        assert_eq!(pass.leading_blits().len(), 1);
    }
    assert_eq!(
        s.passes()[0].leading_blits()[0].cmd,
        BlitCommandType::NotifyBufferDidModifyRange as u32,
    );
    assert_eq!(s.passes()[1].leading_blits()[0].src_handle, old.raw());
    assert_eq!(
        s.passes()[1].leading_blits()[0].dst_handle,
        fresh_texture.raw()
    );
    reset_test_frame(&mut s);
    assert_eq!(s.upload_pass_count(), 0);
}

#[test]
fn triangle_fill_dedup_starts_solid_and_resets_at_each_encoder() {
    let mut cache = LastBoundCache::new();
    assert!(!cache.triangle_fill_mode_changed(TriangleFillMode::Fill));
    assert!(cache.triangle_fill_mode_changed(TriangleFillMode::Lines));
    assert!(!cache.triangle_fill_mode_changed(TriangleFillMode::Lines));
    assert!(cache.triangle_fill_mode_changed(TriangleFillMode::Fill));
    assert!(!cache.triangle_fill_mode_changed(TriangleFillMode::Fill));
    assert!(cache.triangle_fill_mode_changed(TriangleFillMode::Lines));
    cache.reset();
    assert!(!cache.triangle_fill_mode_changed(TriangleFillMode::Fill));
    assert!(cache.triangle_fill_mode_changed(TriangleFillMode::Lines));
}

#[test]
fn rule_h_keeps_fill_changes_outside_removed_color_clear() {
    const CLEAR_PIPELINE: u64 = 0xCAFE_BABE;
    let mut s = fresh();
    s.set_color_render_target(tex(0x3000), 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    // A depth surface the size of the target, as a caster pass binds, so Rule H
    // may strip the colour without widening the pass.
    s.set_depth_stencil_attachment(tex(0x2100), (256, 256), false, false);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(Command::set_triangle_fill_mode(TriangleFillMode::Lines));
    s.emit_command(dummy_draw());
    s.emit_command(Command::set_triangle_fill_mode(TriangleFillMode::Fill));
    let start = s.open_color_clear_quad_block();
    s.emit_command(set_pso(CLEAR_PIPELINE));
    s.emit_command(dummy_draw());
    s.close_color_clear_quad_block(start);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.emit_command(Command::set_triangle_fill_mode(TriangleFillMode::Lines));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    // The next pass clears the target in full, so the colour clear is dead
    // and Rule H may remove it.
    s.clear_color(0, 0, 0, 0);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    s.strip_color_from_no_color_draw_passes(&alt);
    let pass = &s.passes()[0];
    assert!(pass.color_texture().is_null());
    assert!(pass.color_clear_quad_ranges().is_empty());
    let mut fill = TriangleFillMode::Fill as u32;
    let mut draw_modes = Vec::new();
    for cmd in pass.commands() {
        if cmd.cmd == CommandType::SetTriangleFillMode as u32 {
            fill = cmd.param_a;
        }
        if cmd.is_draw() {
            draw_modes.push(fill);
        }
    }
    assert_eq!(
        draw_modes,
        [
            TriangleFillMode::Lines as u32,
            TriangleFillMode::Fill as u32,
            TriangleFillMode::Lines as u32
        ]
    );
}

#[test]
fn sampling_alias_never_becomes_an_srgb_attachment_and_retires_before_reuse() {
    for (stage, bind) in sampler_binds(0x7E12) {
        let mut s = fresh();
        let base = tex(0x7E10);
        let attachment = tex(0x7E11);
        let sample = tex(0x7E12);
        s.register_srgb_twin(attachment, base);
        s.register_texture_view(sample, base);
        assert_eq!(s.twin_of(base), attachment);
        s.emit_command(bind);
        assert!(s.texture_sampled_this_frame(base), "{stage:?}");
        s.unregister_srgb_twin(attachment);
        assert_eq!(s.twin_of(base), MetalHandle::NULL);
        assert_eq!(s.texture_view_to_base.get(&sample), Some(&base));
        s.unregister_texture(sample);
        s.unregister_texture(attachment);
        s.unregister_texture(base);
        assert!(!s.texture_view_to_base.contains_key(&sample));
        assert!(!s.texture_view_to_base.contains_key(&attachment));
        for (_, bind) in sampler_binds(sample.raw()) {
            s.emit_command(bind);
        }
        assert!(
            !s.texture_sampled_this_frame(base),
            "{stage:?}: address reuse"
        );
    }
}

#[test]
fn released_sampling_alias_preserves_stores_and_clear_coalescing() {
    for (stage, _) in sampler_binds(0x4002) {
        for clear_only in [false, true] {
            let rt = tex(0x4000);
            let srgb = tex(0x4001);
            let sample = tex(0x4002);
            let mut s = fresh();
            s.register_srgb_twin(srgb, rt);
            s.register_texture_view(sample, rt);
            s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
            s.clear_color(1, 2, 3, 4);
            if !clear_only {
                s.emit_command(dummy_draw());
            }
            s.set_color_render_target(tex(0x5000), 256, 256, RT_FORMAT, RenderScale::IDENTITY);
            let bind = sampler_binds(sample.raw())
                .into_iter()
                .find(|(candidate, _)| *candidate == stage)
                .expect("stage")
                .1;
            s.emit_command(bind);
            s.emit_command(dummy_draw());
            s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
            if !clear_only {
                s.clear_color(5, 6, 7, 8);
            }
            s.emit_command(dummy_draw());
            s.end_current_pass("test");
            // Release/rename detaches attachment selection before pass finalization.
            s.unregister_srgb_twin(srgb);
            s.coalesce_clear_only_passes();
            s.finalize_load_actions();
            s.finalize_store_actions(false);
            assert_eq!(s.passes().len(), 3, "{stage:?}, clear-only={clear_only}");
            assert_eq!(
                s.passes()[0].color_store(),
                StoreAction::Store,
                "{stage:?}, clear-only={clear_only}"
            );
            s.unregister_texture(sample);
            s.unregister_texture(srgb);
            s.unregister_texture(rt);
            assert!(!s.texture_view_to_base.contains_key(&sample));
        }
    }
}

#[test]
fn ordered_upload_keeps_application_passes_and_blits_in_sequence() {
    let mut s = fresh();
    let destination = tex(0x9100);
    s.emit_command(dummy_draw());
    s.end_current_pass("before conversion");
    s.push_pending_leading_blit(copy_blit(backbuffer(), destination));
    let leading = s.take_pending_leading_blits();
    s.push_upload_pass_with_order::<true>(
        &UploadPassTarget {
            texture: destination,
            subresource: (0, 0),
            size: (6, 2),
            format: BB_FORMAT,
            rect: (2, 1, 2, 1),
        },
        &[dummy_draw()],
        leading,
    );
    s.emit_command(Command::set_fragment_texture(destination.raw(), 0));
    s.emit_command(dummy_draw());
    s.end_current_pass("after conversion");
    s.coalesce_clear_only_passes();
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    s.cull_dead_clear_only_passes();
    assert_eq!(
        s.upload_pass_count(),
        0,
        "the conversion is not a frame prefix"
    );
    assert_eq!(s.passes().len(), 3);
    assert_eq!(s.passes()[0].color_texture(), backbuffer());
    assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
    let conversion = &s.passes()[1];
    assert_eq!(conversion.color_texture(), destination);
    assert!(s.texture_written_by_blit_this_frame(destination));
    assert_eq!(conversion.color_load(), ColorLoad::Load);
    assert_eq!(conversion.color_store(), StoreAction::Store);
    assert_eq!(conversion.leading_blits().len(), 1);
    assert_eq!(conversion.leading_blits()[0].src_handle, backbuffer().raw());
    assert_eq!(conversion.leading_blits()[0].dst_handle, destination.raw());
    assert_eq!(s.passes()[2].color_texture(), backbuffer());
}

// ── Rule I: clears overwritten before a read ──

fn resz_intz() -> MetalHandle<MTLTextureKind> {
    tex(0x7000)
}

fn resz_ds() -> MetalHandle<MTLTextureKind> {
    tex(0x7200)
}

/// The RESZ frame up to its multisampled draw: a depth clear on INTZ, then a draw on a 2x target.
///
/// The clear lands in a depth-only pass of its own, because the multisampled
/// render target cannot share a pass with the single-sample INTZ surface.
/// `before_draw` runs inside the draw pass ahead of its draw; `stencil` makes
/// the INTZ surface a stencil format and clears its stencil plane too.
fn resz_clear_then_msaa_draw(stencil: bool, before_draw: impl FnOnce(&mut PassState)) -> PassState {
    let mut s = fresh();
    s.set_depth_stencil_attachment(resz_intz(), BB_SIZE, true, stencil);
    assert_eq!(s.clear_depth(f32::to_bits(1.0)), DepthClearOutcome::Folded);
    if stencil {
        assert_eq!(s.clear_stencil(0), StencilClearOutcome::Folded);
    }
    s.set_color_render_target(
        tex(0x7100),
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_color_msaa(tex(0x7101), MetalHandle::NULL, 2);
    s.set_depth_stencil_attachment(resz_ds(), BB_SIZE, false, true);
    s.set_depth_sample_count(2);
    s.ensure_pass_open();
    before_draw(&mut s);
    s.emit_command(dummy_draw());
    let clear = &s.passes()[0];
    assert!(
        clear.color_texture().is_null(),
        "the clear is a depth-only pass"
    );
    assert_eq!(clear.depth_texture(), resz_intz());
    assert_eq!(
        clear.depth_load(),
        DepthLoad::Clear {
            value: f32::to_bits(1.0)
        }
    );
    s
}

/// Queue the RESZ transfer of the bound 2x depth surface into level `level` of INTZ.
///
/// The shape `FrameEncoder::queue_depth_transfer` records: a `TransferDepth`
/// queued after the clears, leading the next pass that opens. The unix side
/// writes the whole destination level whatever the region fields say.
fn transfer_into_intz(s: &mut PassState, level: u32) {
    let mut blit = depth_transfer(resz_ds(), resz_intz());
    blit.dst_mip_level = level;
    s.push_leading_blit_after_clears(blit, "test");
}

/// Whether pass `index` of `s` leads with the transfer into INTZ.
fn leads_with_intz_transfer(s: &PassState, index: usize) -> bool {
    s.passes()[index].leading_blits().iter().any(|b| {
        b.cmd == BlitCommandType::TransferDepth as u32 && b.dst_handle == resz_intz().raw()
    })
}

/// Draw onto the back buffer with INTZ bound to sampler 0, and close the pass.
fn sample_intz_on_the_backbuffer(s: &mut PassState) {
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
    s.emit_command(Command::set_fragment_texture(resz_intz().raw(), 0));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
}

/// Whether some pass of `s` still clears the depth plane of `texture`.
fn clears_depth_of(s: &PassState, texture: MetalHandle<MTLTextureKind>) -> bool {
    s.passes()
        .iter()
        .any(|p| p.depth_texture() == texture && matches!(p.depth_load(), DepthLoad::Clear { .. }))
}

#[test]
fn rule_i_drops_a_depth_clear_the_resz_transfer_overwrites() {
    // The clear on INTZ, the multisampled draw, the transfer into INTZ's
    // level 0, then a pass sampling INTZ: nothing reads the cleared depth
    // before the transfer replaces every texel of it.
    let mut s = resz_clear_then_msaa_draw(false, |_| {});
    transfer_into_intz(&mut s, 0);
    sample_intz_on_the_backbuffer(&mut s);
    assert_eq!(s.passes().len(), 3);
    assert!(leads_with_intz_transfer(&s, 2));

    s.drop_overwritten_clear_only_passes();

    assert_eq!(s.passes().len(), 2, "the clear-only pass is gone");
    assert!(!clears_depth_of(&s, resz_intz()));
    assert_eq!(
        s.passes()[0].depth_texture(),
        resz_ds(),
        "the draw pass stays"
    );
    assert_eq!(s.passes()[1].color_texture(), backbuffer());
    assert!(
        leads_with_intz_transfer(&s, 1),
        "the transfer still leads the sampling pass"
    );

    // The rest of the pipeline keeps the transfer as well.
    s.coalesce_clear_only_passes();
    s.finalize_load_actions();
    s.finalize_store_actions(false);
    s.strip_dead_color_in_clear_only_passes();
    s.cull_dead_clear_only_passes();
    assert_eq!(s.passes().len(), 2);
    assert!(leads_with_intz_transfer(&s, 1));
}

#[test]
fn rule_i_takes_a_depth_transfer_as_the_whole_level_whatever_its_region() {
    // The unix encoder reads neither the origin nor the region of a
    // TransferDepth: it writes the destination level at its full extent. A
    // region that disagrees with the level does not make it partial.
    let mut s = resz_clear_then_msaa_draw(false, |_| {});
    let mut blit = depth_transfer(resz_ds(), resz_intz());
    blit.region_w = 1;
    blit.region_h = 1;
    s.push_leading_blit_after_clears(blit, "test");
    sample_intz_on_the_backbuffer(&mut s);
    s.drop_overwritten_clear_only_passes();
    assert!(!clears_depth_of(&s, resz_intz()));
    assert!(leads_with_intz_transfer(&s, 1));
}

#[test]
fn rule_i_keeps_a_clear_read_before_the_overwrite() {
    // A sampler bind of INTZ in the draw pass reads the cleared depth.
    let mut s = resz_clear_then_msaa_draw(false, |s| {
        s.emit_command(Command::set_fragment_texture(resz_intz().raw(), 0));
    });
    transfer_into_intz(&mut s, 0);
    sample_intz_on_the_backbuffer(&mut s);
    s.drop_overwritten_clear_only_passes();
    assert_eq!(s.passes().len(), 3, "a sampler read keeps the clear");
    assert!(clears_depth_of(&s, resz_intz()));

    // So does a copy out of INTZ queued ahead of the transfer.
    let mut s = resz_clear_then_msaa_draw(false, |_| {});
    s.end_current_pass("test");
    s.push_pending_leading_blit(BlitCommand::copy_texture_to_texture_full_mip(
        resz_intz().raw(),
        0x7300,
        0,
        BB_SIZE.0,
        BB_SIZE.1,
    ));
    transfer_into_intz(&mut s, 0);
    sample_intz_on_the_backbuffer(&mut s);
    s.drop_overwritten_clear_only_passes();
    assert!(
        clears_depth_of(&s, resz_intz()),
        "a blit read keeps the clear"
    );

    // And a transfer out of INTZ ahead of the one into it.
    let mut s = resz_clear_then_msaa_draw(false, |_| {});
    s.push_leading_blit_after_clears(depth_transfer(resz_intz(), tex(0x7300)), "test");
    transfer_into_intz(&mut s, 0);
    sample_intz_on_the_backbuffer(&mut s);
    s.drop_overwritten_clear_only_passes();
    assert!(
        clears_depth_of(&s, resz_intz()),
        "a transfer reading INTZ keeps the clear"
    );

    // And a later pass that attaches INTZ, whatever it loads.
    let mut s = resz_clear_then_msaa_draw(false, |_| {});
    s.set_color_render_target(
        tex(0x7400),
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_depth_stencil_attachment(resz_intz(), BB_SIZE, true, false);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        tex(0x7100),
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_color_msaa(tex(0x7101), MetalHandle::NULL, 2);
    s.set_depth_stencil_attachment(resz_ds(), BB_SIZE, false, true);
    s.set_depth_sample_count(2);
    s.emit_command(dummy_draw());
    transfer_into_intz(&mut s, 0);
    sample_intz_on_the_backbuffer(&mut s);
    s.drop_overwritten_clear_only_passes();
    assert!(
        clears_depth_of(&s, resz_intz()),
        "an attachment keeps the clear"
    );
}

#[test]
fn rule_i_keeps_a_clear_only_partly_overwritten() {
    // A copy smaller than the level leaves cleared texels behind.
    let mut s = resz_clear_then_msaa_draw(false, |_| {});
    s.push_leading_blit_after_clears(
        BlitCommand::copy_texture_to_texture_full_mip(
            resz_ds().raw(),
            resz_intz().raw(),
            0,
            BB_SIZE.0 / 2,
            BB_SIZE.1 / 2,
        ),
        "test",
    );
    sample_intz_on_the_backbuffer(&mut s);
    s.drop_overwritten_clear_only_passes();
    assert!(
        clears_depth_of(&s, resz_intz()),
        "a smaller copy keeps the clear"
    );

    // A transfer into another level leaves the cleared level untouched.
    let mut s = resz_clear_then_msaa_draw(false, |_| {});
    transfer_into_intz(&mut s, 1);
    sample_intz_on_the_backbuffer(&mut s);
    s.drop_overwritten_clear_only_passes();
    assert!(
        clears_depth_of(&s, resz_intz()),
        "another level keeps the clear"
    );

    // The same for a copy onto a cleared colour target.
    for (label, blit) in [
        (
            "a smaller region",
            BlitCommand::copy_texture_to_texture_full_mip(0x7500, 0x7600, 0, 128, 128),
        ),
        (
            "another level",
            BlitCommand::copy_texture_to_texture_full_mip(0x7500, 0x7600, 1, 256, 256),
        ),
    ] {
        let mut s = colour_clear_then_draw_elsewhere(tex(0x7600));
        s.push_pending_leading_blit(blit);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.drop_overwritten_clear_only_passes();
        assert!(clears_colour_of(&s, tex(0x7600)), "{label} keeps the clear");
    }
}

#[test]
fn rule_i_keeps_a_clear_nothing_overwrites_in_the_submission() {
    // The store is the result: D3D9 keeps it across `Present`, and a flush
    // continues the frame.
    let mut s = resz_clear_then_msaa_draw(false, |_| {});
    s.end_current_pass("test");
    s.drop_overwritten_clear_only_passes();
    assert_eq!(s.passes().len(), 2);
    assert!(clears_depth_of(&s, resz_intz()));

    let mut s = colour_clear_then_draw_elsewhere(tex(0x7600));
    s.drop_overwritten_clear_only_passes();
    assert!(clears_colour_of(&s, tex(0x7600)));
}

#[test]
fn rule_i_keeps_a_stencil_clear_under_a_depth_transfer() {
    // The transfer carries stencil only when both ends are
    // Depth32FloatStencil8, which the blit does not record, so the cleared
    // stencil could be lost with the pass.
    let mut s = resz_clear_then_msaa_draw(true, |_| {});
    assert_eq!(
        s.passes()[0].stencil_load(),
        StencilLoad::Clear { value: 0 }
    );
    transfer_into_intz(&mut s, 0);
    sample_intz_on_the_backbuffer(&mut s);
    s.drop_overwritten_clear_only_passes();
    assert_eq!(s.passes().len(), 3, "the stencil clear stays");
    assert!(clears_depth_of(&s, resz_intz()));
}

/// A colour clear on `rt` materialised as a clear-only pass, then a draw on the back buffer.
///
/// The back buffer pass is closed, so a blit pushed next leads the pass after it.
fn colour_clear_then_draw_elsewhere(rt: MetalHandle<MTLTextureKind>) -> PassState {
    let mut s = fresh();
    s.set_color_render_target(rt, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    assert_eq!(s.clear_color(1, 2, 3, 4), ColorClearOutcome::Folded);
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let clear = &s.passes()[0];
    assert_eq!(clear.color_texture(), rt);
    assert!(matches!(clear.color_load(), ColorLoad::Clear { .. }));
    assert!(!clear.commands().iter().any(Command::is_draw));
    s
}

/// Whether some pass of `s` still clears render target 0 `texture`.
fn clears_colour_of(s: &PassState, texture: MetalHandle<MTLTextureKind>) -> bool {
    s.passes()
        .iter()
        .any(|p| p.color_texture() == texture && matches!(p.color_load(), ColorLoad::Clear { .. }))
}

#[test]
fn rule_i_drops_a_colour_clear_a_full_copy_overwrites() {
    // A copy onto the whole level lands before anything reads the target,
    // so the clear under it is never seen.
    let rt = tex(0x7600);
    let mut s = colour_clear_then_draw_elsewhere(rt);
    s.push_pending_leading_blit(BlitCommand::copy_texture_to_texture_full_mip(
        0x7500,
        rt.raw(),
        0,
        256,
        256,
    ));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
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
    let before = s.passes().len();

    s.drop_overwritten_clear_only_passes();

    assert_eq!(s.passes().len(), before - 1, "the clear-only pass is gone");
    assert!(!clears_colour_of(&s, rt));
    assert_eq!(
        s.passes()[1].leading_blits().len(),
        1,
        "the copy keeps its place ahead of the pass it leads"
    );
}

// ── Dead draws: nothing written, nothing recorded ──

/// The state of the dummy draw ahead of a RESZ resolve, minus the render state under test.
///
/// Colour masked off on render target 0, depth test off with the depth write
/// still enabled, the default stencil state, and a depth-stencil attachment
/// carrying both planes: nothing this draw runs can write.
struct DeadDrawState {
    rs: PipelineRsBits,
    extra: ExtraColorAttachments,
    ps_color_out_mask: u8,
    depth_stencil: DepthStencilSnapshot,
    attach: PipelineAttachFlags,
    counting_query: bool,
}

impl DeadDrawState {
    fn new() -> Self {
        Self {
            rs: PipelineRsBits::default(),
            extra: ExtraColorAttachments::NONE,
            ps_color_out_mask: 1,
            depth_stencil: DepthStencilSnapshot {
                depth_write: 1,
                ..DepthStencilSnapshot::inert()
            },
            attach: PipelineAttachFlags::HAS_DEPTH | PipelineAttachFlags::HAS_STENCIL,
            counting_query: false,
        }
    }

    fn writes(&self) -> DrawWrites<'_> {
        DrawWrites {
            rs: &self.rs,
            extra: &self.extra,
            ps_color_out_mask: self.ps_color_out_mask,
            depth_stencil: &self.depth_stencil,
            attach: self.attach,
            counting_query: self.counting_query,
        }
    }

    fn is_dead(&self) -> bool {
        draw_writes_nothing(&self.writes())
    }
}

/// Assert that applying `change` to the dead baseline makes the draw live.
fn assert_makes_live(label: &str, change: impl FnOnce(&mut DeadDrawState)) {
    let mut state = DeadDrawState::new();
    assert!(state.is_dead(), "the baseline writes nothing");
    change(&mut state);
    assert!(!state.is_dead(), "{label} makes the draw observable");
}

/// Assert that applying `change` to the dead baseline leaves the draw dead.
fn assert_stays_dead(label: &str, change: impl FnOnce(&mut DeadDrawState)) {
    let mut state = DeadDrawState::new();
    change(&mut state);
    assert!(state.is_dead(), "{label} still writes nothing");
}

#[test]
fn dead_draw_colour_write_on_any_bound_target_is_live() {
    assert_makes_live("COLORWRITEENABLE on render target 0", |s| {
        s.rs.color_write_mask = 0x1;
    });
    assert_makes_live("COLORWRITEENABLE1 on a bound, written target 1", |s| {
        s.extra.present_mask = 0b001;
        s.ps_color_out_mask = 0b011;
        s.rs.color_write_mask_ext[0] = 0xF;
    });
    assert_makes_live("COLORWRITEENABLE3 on a bound, written target 3", |s| {
        s.extra.present_mask = 0b100;
        s.ps_color_out_mask = 0b1001;
        s.rs.color_write_mask_ext[2] = 0x8;
    });
    // A mask on a target the pass does not attach, or one the shader never
    // writes, writes nothing, the same reading the pipeline key takes.
    assert_stays_dead("COLORWRITEENABLE1 with no target 1", |s| {
        s.ps_color_out_mask = 0b011;
        s.rs.color_write_mask_ext[0] = 0xF;
    });
    assert_stays_dead("COLORWRITEENABLE1 on a target the shader skips", |s| {
        s.extra.present_mask = 0b001;
        s.rs.color_write_mask_ext[0] = 0xF;
    });
}

#[test]
fn dead_draw_depth_write_needs_the_test_the_write_and_the_plane() {
    assert_makes_live("ZENABLE with ZWRITEENABLE", |s| {
        s.depth_stencil.depth_enable = 1;
    });
    assert_stays_dead("ZENABLE without ZWRITEENABLE", |s| {
        s.depth_stencil.depth_enable = 1;
        s.depth_stencil.depth_write = 0;
    });
    assert_stays_dead("ZENABLE and ZWRITEENABLE with no depth attachment", |s| {
        s.depth_stencil.depth_enable = 1;
        s.attach = PipelineAttachFlags::empty();
    });
}

#[test]
fn dead_draw_stencil_write_needs_an_op_the_mask_and_the_plane() {
    let replace = u8::try_from(mtld3d_types::D3DSTENCILOP_REPLACE).expect("stencil op fits u8");
    let incr = u8::try_from(mtld3d_types::D3DSTENCILOP_INCR).expect("stencil op fits u8");
    let enabled = |s: &mut DeadDrawState| {
        s.depth_stencil.stencil_enable = 1;
        s.depth_stencil.write_mask = u32::MAX;
    };
    // Every operation of either face writes on its own.
    assert_makes_live("a front-face fail op", |s| {
        enabled(s);
        s.depth_stencil.front.fail_op = replace;
    });
    assert_makes_live("a front-face depth-fail op", |s| {
        enabled(s);
        s.depth_stencil.front.depth_fail_op = incr;
    });
    assert_makes_live("a front-face pass op", |s| {
        enabled(s);
        s.depth_stencil.front.pass_op = replace;
    });
    assert_makes_live("a back-face op under two-sided stencil", |s| {
        enabled(s);
        s.depth_stencil.back.pass_op = incr;
    });
    // `KEEP` everywhere, a write mask the 8-bit plane cannot see, stencil
    // off, or no stencil plane: nothing changes.
    assert_stays_dead("KEEP on every op", enabled);
    assert_stays_dead("a write mask above the stencil bits", |s| {
        enabled(s);
        s.depth_stencil.write_mask = !STENCIL_MASK_BITS;
        s.depth_stencil.front.pass_op = replace;
    });
    assert_stays_dead("STENCILENABLE off", |s| {
        enabled(s);
        s.depth_stencil.stencil_enable = 0;
        s.depth_stencil.front.pass_op = replace;
    });
    assert_stays_dead("no stencil plane", |s| {
        enabled(s);
        s.depth_stencil.front.pass_op = replace;
        s.attach = PipelineAttachFlags::HAS_DEPTH;
    });
}

#[test]
fn dead_draw_counting_query_observes_it() {
    assert_makes_live("an open occlusion query", |s| s.counting_query = true);
}

#[test]
fn dead_draw_coverage_states_alone_write_nothing() {
    // Alpha-to-coverage only narrows which samples a write reaches; with no
    // write left there is nothing for it to narrow.
    assert_stays_dead("alpha-to-coverage", |s| {
        s.rs.flags |= crate::pipeline_state::PipelineRsFlags::ALPHA_TO_COVERAGE;
    });
    assert_stays_dead("blending", |s| {
        s.rs.flags |= crate::pipeline_state::PipelineRsFlags::BLEND_ENABLE;
    });
}

#[test]
fn dead_draw_is_not_skipped_while_it_would_land_a_pending_clear() {
    let dead = DeadDrawState::new();
    // A depth clear with no pass open waits for the pass the draw would open.
    let mut s = fresh();
    assert_eq!(s.clear_depth(f32::to_bits(1.0)), DepthClearOutcome::Folded);
    assert!(!s.skip_dead_draw(&dead.writes()));
    s.ensure_pass_open();
    assert!(
        s.skip_dead_draw(&dead.writes()),
        "once the pass is open it may go"
    );

    let mut s = fresh();
    assert_eq!(s.clear_color(1, 2, 3, 4), ColorClearOutcome::Folded);
    assert!(!s.skip_dead_draw(&dead.writes()));

    // Nothing pending and no pass open: the draw opens nothing either.
    assert!(fresh().skip_dead_draw(&dead.writes()));

    // And a draw that writes stays whatever is pending.
    let mut live = DeadDrawState::new();
    live.depth_stencil.depth_enable = 1;
    let mut s = fresh();
    s.ensure_pass_open();
    assert!(!s.skip_dead_draw(&live.writes()));
}

/// What the encoder records for a draw that samples INTZ: nothing when it is dead.
fn draw_sampling_intz(s: &mut PassState, state: &DeadDrawState) {
    if s.skip_dead_draw(&state.writes()) {
        return;
    }
    s.emit_command(Command::set_fragment_texture(resz_intz().raw(), 0));
    s.emit_command(dummy_draw());
}

#[test]
fn dead_draw_before_resz_leaves_rule_i_free_to_drop_the_clear() {
    // The RESZ frame: the depth clear on INTZ, the multisampled draw pass
    // with its depth writes, then the dummy draw with INTZ on sampler 0,
    // COLORWRITEENABLE 0 and ZENABLE off, then the transfer into INTZ and a
    // pass sampling it.
    let dead = DeadDrawState::new();
    let mut s = resz_clear_then_msaa_draw(false, |_| {});
    let draw_pass = s.passes().len() - 1;
    let commands_before = s.passes()[draw_pass].commands().len();
    draw_sampling_intz(&mut s, &dead);
    let pass = &s.passes()[draw_pass];
    assert_eq!(
        pass.commands().len(),
        commands_before,
        "the dead draw records nothing"
    );
    assert_eq!(pass.commands().iter().filter(|c| c.is_draw()).count(), 1);
    assert!(!pass_samples_texture(
        pass,
        resz_intz(),
        &s.texture_view_to_base,
        &s.frame_sampled_textures
    ));
    transfer_into_intz(&mut s, 0);
    sample_intz_on_the_backbuffer(&mut s);
    s.drop_overwritten_clear_only_passes();
    assert!(
        !clears_depth_of(&s, resz_intz()),
        "nothing reads the cleared INTZ before the transfer replaces it"
    );

    // The same draw with the depth test on writes depth, so it is recorded,
    // its sampler bind reads the cleared INTZ, and the clear stays.
    let mut live = DeadDrawState::new();
    live.depth_stencil.depth_enable = 1;
    let mut s = resz_clear_then_msaa_draw(false, |_| {});
    draw_sampling_intz(&mut s, &live);
    transfer_into_intz(&mut s, 0);
    sample_intz_on_the_backbuffer(&mut s);
    s.drop_overwritten_clear_only_passes();
    assert!(
        clears_depth_of(&s, resz_intz()),
        "a live draw's read keeps it"
    );
}

#[test]
fn dead_draw_alone_before_resz_leaves_rule_i_free_to_drop_the_clear() {
    // The resz_test shape: the INTZ clear lands as a depth-only pass when the
    // multisampled surfaces are bound, the pass on them gets only the dummy
    // draw, which is dead, then the transfer into INTZ and a pass sampling it.
    let dead = DeadDrawState::new();
    let mut s = fresh();
    s.set_depth_stencil_attachment(resz_intz(), BB_SIZE, true, false);
    assert_eq!(s.clear_depth(f32::to_bits(1.0)), DepthClearOutcome::Folded);
    s.set_color_render_target(
        tex(0x7100),
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_color_msaa(tex(0x7101), MetalHandle::NULL, 2);
    s.set_depth_stencil_attachment(resz_ds(), BB_SIZE, false, true);
    s.set_depth_sample_count(2);
    assert_eq!(s.passes().len(), 1, "the clear landed before the rebind");
    assert!(s.passes()[0].color_texture().is_null());
    assert_eq!(s.passes()[0].depth_texture(), resz_intz());
    draw_sampling_intz(&mut s, &dead);
    assert_eq!(s.passes().len(), 1, "the dead draw opens no pass");
    transfer_into_intz(&mut s, 0);
    sample_intz_on_the_backbuffer(&mut s);
    assert_eq!(s.passes().len(), 2);
    assert!(leads_with_intz_transfer(&s, 1));

    s.drop_overwritten_clear_only_passes();

    assert_eq!(s.passes().len(), 1, "the clear-only pass is gone");
    assert!(!clears_depth_of(&s, resz_intz()));
    assert!(leads_with_intz_transfer(&s, 0));
    assert_eq!(s.passes()[0].color_texture(), backbuffer());
}

// ── Depth and stencil planes nothing observes ──

/// A frame whose default depth surface carries a stencil plane (D24S8).
fn fresh_with_stencil() -> PassState {
    let mut s = PassState::new();
    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: BB_SIZE,
        depth_has_stencil: true,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    s
}

const BOTH_PLANES: PipelineAttachFlags =
    PipelineAttachFlags::HAS_DEPTH.union(PipelineAttachFlags::HAS_STENCIL);

/// A draw that tests and writes depth, stencil off.
fn depth_draw(s: &mut PassState) {
    s.note_draw_depth_stencil(&DepthStencilSnapshot::depth_overwrite(), BOTH_PLANES);
    s.emit_command(dummy_draw());
}

/// A draw that writes stencil (the stencil clear-quad's state, drawn by the game).
fn stencil_writing_draw(s: &mut PassState) {
    s.note_draw_depth_stencil(&DepthStencilSnapshot::stencil_overwrite(), BOTH_PLANES);
    s.emit_command(dummy_draw());
}

/// A draw with depth and stencil both off, like an interface draw.
fn depthless_draw(s: &mut PassState) {
    s.note_draw_depth_stencil(&DepthStencilSnapshot::inert(), BOTH_PLANES);
    s.emit_command(dummy_draw());
}

/// Bounce render target 0 to `other` and back, ending the pass on the back buffer.
fn break_pass_via(s: &mut PassState, other: MetalHandle<MTLTextureKind>) {
    s.set_color_render_target(other, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
}

/// Draw one pass into `other` with no depth attachment, then rebind the back buffer and `depth()`.
///
/// Leaves a pass between two passes on the depth texture that does not attach
/// it, so the second one is the first's next use. `leading` is queued ahead
/// of that pass.
fn depthless_pass(
    s: &mut PassState,
    other: MetalHandle<MTLTextureKind>,
    has_stencil: bool,
    leading: Option<BlitCommand>,
) {
    s.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
    s.set_color_render_target(other, 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    if let Some(blit) = leading {
        s.push_pending_leading_blit(blit);
    }
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_depth_stencil_attachment(depth(), BB_SIZE, false, has_stencil);
}

/// Open a pass on the bound attachments whose planes load with a whole-attachment `Clear`.
///
/// Issues the covering `Clear` of each plane given through the real clear
/// path, which folds into the next pass's load action even on a texture the
/// submission already drew into. `None` leaves that plane's load as the pass
/// opens it.
fn open_clearing_pass(s: &mut PassState, depth: Option<u32>, stencil: Option<u32>) {
    s.end_current_pass("test");
    if let Some(value) = depth {
        assert_eq!(s.clear_depth(value), DepthClearOutcome::Folded);
    }
    if let Some(value) = stencil {
        assert_eq!(s.clear_stencil(value), StencilClearOutcome::Folded);
    }
    s.ensure_pass_open();
    let pass = s.passes().last().expect("a pass is open");
    if let Some(value) = depth {
        assert_eq!(pass.depth_load(), DepthLoad::Clear { value });
    }
    if let Some(value) = stencil {
        assert_eq!(pass.stencil_load(), StencilLoad::Clear { value });
    }
}

// Rule C's depth arm.

#[test]
fn depth_arm_discards_a_depth_store_the_next_pass_clears() {
    for frame_continues in [false, true] {
        let mut s = fresh();
        depth_draw(&mut s);
        depthless_pass(&mut s, tex(0x3000), false, None);
        open_clearing_pass(&mut s, Some(0), None);
        depth_draw(&mut s);
        s.end_current_pass("test");
        s.finalize_store_actions(frame_continues);
        assert_eq!(s.passes().len(), 3);
        assert_eq!(
            s.passes()[0].depth_store(),
            StoreAction::DontCare,
            "the clear in pass 2 overwrites what pass 0 stored (frame_continues={frame_continues})"
        );
    }
}

#[test]
fn depth_arm_keeps_the_store_when_the_next_pass_loads() {
    let mut s = fresh();
    depth_draw(&mut s);
    break_pass_via(&mut s, tex(0x3000));
    depth_draw(&mut s);
    s.end_current_pass("test");
    s.finalize_store_actions(true);
    assert!(
        s.passes()
            .iter()
            .all(|p| p.depth_store() == StoreAction::Store)
    );
}

#[test]
fn depth_arm_keeps_the_store_of_a_sampled_or_sampleable_texture() {
    let mut sampled = fresh();
    depth_draw(&mut sampled);
    sampled.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
    sampled.set_color_render_target(tex(0x3000), 256, 256, RT_FORMAT, RenderScale::IDENTITY);
    sampled.emit_command(Command::set_fragment_texture(depth().raw(), 0));
    sampled.emit_command(dummy_draw());
    sampled.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        sampled.render_scale,
    );
    sampled.set_depth_stencil_attachment(depth(), BB_SIZE, false, false);
    open_clearing_pass(&mut sampled, Some(0), None);
    depth_draw(&mut sampled);
    sampled.end_current_pass("test");
    sampled.finalize_store_actions(true);
    assert_eq!(
        sampled.passes()[0].depth_store(),
        StoreAction::Store,
        "the sampler in pass 1 reads what pass 0 stored"
    );

    let shadow = tex(0x6000);
    let mut sampleable = fresh();
    sampleable.set_depth_stencil_attachment(shadow, BB_SIZE, true, false);
    depth_draw(&mut sampleable);
    sampleable.end_current_pass("test");
    open_clearing_pass(&mut sampleable, Some(0), None);
    depth_draw(&mut sampleable);
    sampleable.end_current_pass("test");
    sampleable.finalize_store_actions(true);
    assert_eq!(
        sampleable.passes()[0].depth_store(),
        StoreAction::Store,
        "a sampleable shadow map keeps every store"
    );
}

#[test]
fn depth_arm_keeps_the_store_when_a_blit_between_touches_the_texture() {
    // A depth transfer into the texture, carried by the pass in between or
    // by the clearing pass itself, runs after pass 0's store and before the
    // clear. Neither is a read `seen_sampled_textures` records, so the scan
    // is what keeps the store.
    let other = tex(0x6100);
    for on_clearing_pass in [false, true] {
        let mut s = fresh();
        depth_draw(&mut s);
        let between = (!on_clearing_pass).then(|| depth_transfer(other, depth()));
        depthless_pass(&mut s, tex(0x3000), false, between);
        if on_clearing_pass {
            s.push_pending_leading_blit(depth_transfer(other, depth()));
        }
        open_clearing_pass(&mut s, Some(0), None);
        depth_draw(&mut s);
        s.end_current_pass("test");
        assert!(!s.seen_sampled_textures.contains(&depth()));
        s.finalize_store_actions(true);
        assert_eq!(
            s.passes()[0].depth_store(),
            StoreAction::Store,
            "a blit touching the texture runs before the clear (on_clearing_pass={on_clearing_pass})"
        );
    }
}

#[test]
fn depth_arm_keeps_the_store_when_a_pass_between_reads_the_texture() {
    // A read the session set has not recorded (inserted straight into the
    // pass list) still keeps the store: the scan checks the passes between.
    let mut s = fresh();
    depth_draw(&mut s);
    depthless_pass(&mut s, tex(0x3000), false, None);
    open_clearing_pass(&mut s, Some(0), None);
    depth_draw(&mut s);
    s.end_current_pass("test");
    s.passes[1]
        .leading_blits
        .push(copy_blit(depth(), tex(0x6200)));
    s.finalize_store_actions(true);
    assert_eq!(s.passes()[0].depth_store(), StoreAction::Store);
}

#[test]
fn depth_arm_matches_the_mip_level() {
    let half = (BB_SIZE.0 / 2, BB_SIZE.1 / 2);
    let mut s = fresh();
    s.set_depth_stencil_attachment_level(depth(), 0, BB_SIZE, false, false);
    depth_draw(&mut s);
    s.set_depth_stencil_attachment_level(depth(), 1, half, false, false);
    open_clearing_pass(&mut s, Some(0), None);
    depth_draw(&mut s);
    s.end_current_pass("test");
    s.finalize_store_actions(true);
    assert_eq!(
        s.passes()[0].depth_store(),
        StoreAction::Store,
        "a clear of level 1 does not overwrite level 0"
    );
}

#[test]
fn depth_arm_decides_the_depth_and_stencil_planes_apart() {
    // depth only, stencil only, both: each plane's store goes only when the
    // next pass clears that plane.
    let cases = [
        (Some(0), None, StoreAction::DontCare, StoreAction::Store),
        (None, Some(0), StoreAction::Store, StoreAction::DontCare),
        (
            Some(0),
            Some(0),
            StoreAction::DontCare,
            StoreAction::DontCare,
        ),
    ];
    for (depth_clear, stencil_clear, depth_store, stencil_store) in cases {
        let mut s = fresh_with_stencil();
        stencil_writing_draw(&mut s);
        depth_draw(&mut s);
        depthless_pass(&mut s, tex(0x3000), true, None);
        open_clearing_pass(&mut s, depth_clear, stencil_clear);
        depth_draw(&mut s);
        s.end_current_pass("test");
        s.finalize_store_actions(true);
        let pass = &s.passes()[0];
        assert_eq!(
            (pass.depth_store(), pass.stencil_store()),
            (depth_store, stencil_store),
            "next pass clears depth={depth_clear:?} stencil={stencil_clear:?}"
        );
    }
}

// A stencil plane nothing has written.

#[test]
fn an_unwritten_stencil_plane_loads_and_stores_dontcare() {
    let mut s = fresh_with_stencil();
    depth_draw(&mut s);
    break_pass_via(&mut s, tex(0x3000));
    depth_draw(&mut s);
    s.end_current_pass("test");
    s.finalize_store_actions(true);
    for pass in s.passes().iter().filter(|p| p.depth_texture() == depth()) {
        assert_eq!(pass.stencil_load(), StencilLoad::DontCare);
        assert_eq!(pass.stencil_store(), StoreAction::DontCare);
        assert_eq!(pass.depth_store(), StoreAction::Store, "depth is untouched");
    }
    assert_eq!(s.passes()[2].depth_load(), DepthLoad::Load);
}

#[test]
fn draws_that_do_not_write_stencil_leave_it_unwritten() {
    let keep_ops_test = DepthStencilSnapshot {
        stencil_enable: 1,
        write_mask: STENCIL_MASK_BITS,
        ..DepthStencilSnapshot::inert()
    };
    let masked_writes = DepthStencilSnapshot {
        write_mask: 0,
        ..DepthStencilSnapshot::stencil_overwrite()
    };
    for (name, state) in [
        ("inert", DepthStencilSnapshot::inert()),
        ("depth only", DepthStencilSnapshot::depth_overwrite()),
        ("stencil test with KEEP on every outcome", keep_ops_test),
        ("stencil write mask zero", masked_writes),
    ] {
        let mut s = fresh_with_stencil();
        s.note_draw_depth_stencil(&state, BOTH_PLANES);
        s.emit_command(dummy_draw());
        break_pass_via(&mut s, tex(0x3000));
        depth_draw(&mut s);
        s.end_current_pass("test");
        s.finalize_store_actions(true);
        assert_eq!(
            s.passes()[0].stencil_store(),
            StoreAction::DontCare,
            "{name}"
        );
    }
}

#[test]
fn every_stencil_write_keeps_the_stencil_plane() {
    // A stencil-writing draw in the last pass keeps the first pass's load
    // and store too: a write anywhere in the submission counts.
    let written_by = [
        (
            "a stencil-writing draw",
            stencil_writing_draw as fn(&mut PassState),
        ),
        ("a stencil clear-quad", |s: &mut PassState| {
            s.note_depth_stencil_clear_quad(true);
            s.emit_command(dummy_draw());
        }),
        ("a folded stencil clear", |s: &mut PassState| {
            s.end_current_pass("test");
            open_clearing_pass(s, None, Some(7));
            s.emit_command(dummy_draw());
        }),
        ("a stencil upload blit", |s: &mut PassState| {
            let mut upload = copy_blit(tex(0x6300), depth());
            upload.cmd = BlitCommandType::CopyBufferToStencil as u32;
            s.push_leading_blit_after_clears(upload, "test");
            s.emit_command(dummy_draw());
        }),
    ];
    for (name, write) in written_by {
        let mut s = fresh_with_stencil();
        depth_draw(&mut s);
        break_pass_via(&mut s, tex(0x3000));
        write(&mut s);
        s.end_current_pass("test");
        s.finalize_store_actions(true);
        assert_eq!(
            s.passes()[0].stencil_load(),
            StencilLoad::DontCare,
            "{name}: Rule A"
        );
        assert_eq!(s.passes()[0].stencil_store(), StoreAction::Store, "{name}");
        assert_ne!(
            s.passes().last().unwrap().stencil_load(),
            StencilLoad::DontCare,
            "{name}"
        );
    }
}

#[test]
fn a_depth_transfer_or_frame_head_copy_marks_its_destination_written() {
    let mut s = fresh_with_stencil();
    s.push_leading_blit_after_clears(depth_transfer(tex(0x6400), depth()), "test");
    depth_draw(&mut s);
    s.end_current_pass("test");
    s.finalize_store_actions(true);
    assert_eq!(s.passes()[0].stencil_store(), StoreAction::Store);

    let mut s = fresh_with_stencil();
    s.note_stencil_blit(&copy_blit(tex(0x6400), depth()));
    depth_draw(&mut s);
    s.end_current_pass("test");
    s.finalize_store_actions(true);
    assert_eq!(s.passes()[0].stencil_store(), StoreAction::Store);
}

#[test]
fn a_stencil_write_survives_into_the_next_submission() {
    // Written before a mid-frame flush, tested after it: the continuation
    // must load what the first submission stored.
    let mut s = fresh_with_stencil();
    stencil_writing_draw(&mut s);
    s.end_current_pass("test");
    s.finalize_store_actions(true);
    assert_eq!(s.passes()[0].stencil_store(), StoreAction::Store);

    s.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: depth(),
        depth_size: BB_SIZE,
        depth_has_stencil: true,
        render_scale: RenderScale::IDENTITY,
        continues_frame: true,
    });
    let keep_ops_test = DepthStencilSnapshot {
        stencil_enable: 1,
        ..DepthStencilSnapshot::inert()
    };
    s.note_draw_depth_stencil(&keep_ops_test, BOTH_PLANES);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(true);
    assert_eq!(s.passes()[0].stencil_load(), StencilLoad::Load);
    assert_eq!(s.passes()[0].stencil_store(), StoreAction::Store);

    s.unregister_texture(depth());
    assert!(
        !s.stencil_written_textures.contains(&depth()),
        "a destroyed texture's address may come back as a fresh surface"
    );
}

#[test]
fn a_texture_without_stencil_keeps_its_stencil_fields() {
    let mut s = fresh();
    depth_draw(&mut s);
    break_pass_via(&mut s, tex(0x3000));
    depth_draw(&mut s);
    s.end_current_pass("test");
    s.finalize_store_actions(true);
    assert_eq!(s.passes()[0].stencil_store(), StoreAction::Store);
}

#[test]
fn a_clear_only_pass_that_keeps_only_its_stencil_survives_the_cull() {
    // Pass 0 clears stencil alone; pass 1 clears colour and depth and tests
    // the stencil pass 0 left. Pass 0's colour and depth stores go (pass 1
    // clears both), its stencil store stays, so Rule F must keep it.
    let mut s = fresh_with_stencil();
    open_clearing_pass(&mut s, None, Some(3));
    s.end_current_pass("test");
    open_clearing_pass(&mut s, Some(0), None);
    s.passes.last_mut().unwrap().color_load = ColorLoad::Clear {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    let keep_ops_test = DepthStencilSnapshot {
        stencil_enable: 1,
        ..DepthStencilSnapshot::inert()
    };
    s.note_draw_depth_stencil(&keep_ops_test, BOTH_PLANES);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(true);
    assert_eq!(s.passes()[0].color_store(), StoreAction::DontCare);
    assert_eq!(s.passes()[0].depth_store(), StoreAction::DontCare);
    assert_eq!(s.passes()[0].stencil_store(), StoreAction::Store);
    s.strip_dead_color_in_clear_only_passes();
    s.cull_dead_clear_only_passes();
    assert_eq!(s.passes().len(), 2, "the stencil clear is kept");
    assert_eq!(
        s.passes()[0].stencil_load(),
        StencilLoad::Clear { value: 3 }
    );
}

// A pass that never uses depth.

#[test]
fn a_last_pass_that_never_uses_depth_discards_its_loads() {
    let mut s = fresh_with_stencil();
    stencil_writing_draw(&mut s);
    depth_draw(&mut s);
    break_pass_via(&mut s, tex(0x3000));
    depthless_draw(&mut s);
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    let last = s.passes().last().unwrap();
    assert_eq!(last.depth_store(), StoreAction::DontCare, "Rule B");
    assert_eq!(last.depth_load(), DepthLoad::DontCare);
    assert_eq!(last.stencil_load(), StencilLoad::DontCare);
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::DontCare, "Rule A");
    assert_eq!(s.passes()[0].depth_store(), StoreAction::Store);
}

#[test]
fn any_depth_or_stencil_use_keeps_the_loads() {
    let uses = [
        ("a depth-testing draw", depth_draw as fn(&mut PassState)),
        ("a stencil-testing draw", |s: &mut PassState| {
            s.note_draw_depth_stencil(
                &DepthStencilSnapshot {
                    stencil_enable: 1,
                    ..DepthStencilSnapshot::inert()
                },
                BOTH_PLANES,
            );
            s.emit_command(dummy_draw());
        }),
        ("a depth clear-quad", |s: &mut PassState| {
            s.note_depth_stencil_clear_quad(false);
            s.emit_command(dummy_draw());
        }),
        ("a stencil clear-quad", |s: &mut PassState| {
            s.note_depth_stencil_clear_quad(true);
            s.emit_command(dummy_draw());
        }),
    ];
    for (name, use_depth) in uses {
        let mut s = fresh_with_stencil();
        stencil_writing_draw(&mut s);
        break_pass_via(&mut s, tex(0x3000));
        depthless_draw(&mut s);
        use_depth(&mut s);
        s.end_current_pass("test");
        s.finalize_store_actions(false);
        let last = s.passes().last().unwrap();
        assert_eq!(last.depth_load(), DepthLoad::Load, "{name}");
        assert_eq!(last.stencil_load(), StencilLoad::Load, "{name}");
    }
}

#[test]
fn an_unused_pass_keeps_its_loads_when_its_stores_are_kept() {
    // A mid-frame flush turns Rule B off, so the stores stay and so do the loads.
    let mut s = fresh_with_stencil();
    stencil_writing_draw(&mut s);
    break_pass_via(&mut s, tex(0x3000));
    depthless_draw(&mut s);
    s.end_current_pass("test");
    s.finalize_store_actions(true);
    let last = s.passes().last().unwrap();
    assert_eq!(last.depth_load(), DepthLoad::Load);
    assert_eq!(last.stencil_load(), StencilLoad::Load);
}

#[test]
fn an_unused_pass_decides_each_plane_on_its_own_store() {
    // Pass 1 never uses depth and is followed by a pass that clears depth
    // alone: its depth load goes, its stencil load stays for the store that
    // keeps what pass 0 wrote.
    let mut s = fresh_with_stencil();
    stencil_writing_draw(&mut s);
    s.end_current_pass("test");
    depthless_draw(&mut s);
    s.end_current_pass("test");
    open_clearing_pass(&mut s, Some(0), None);
    depth_draw(&mut s);
    s.end_current_pass("test");
    s.finalize_store_actions(true);
    let unused = &s.passes()[1];
    assert_eq!(unused.depth_texture(), depth());
    assert_eq!(unused.depth_store(), StoreAction::DontCare);
    assert_eq!(unused.depth_load(), DepthLoad::DontCare);
    assert_eq!(unused.stencil_store(), StoreAction::Store);
    assert_eq!(unused.stencil_load(), StencilLoad::Load);
}

#[test]
fn an_unused_pass_keeps_a_clear_load() {
    let mut s = fresh_with_stencil();
    open_clearing_pass(&mut s, Some(0), Some(0));
    depthless_draw(&mut s);
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::Clear { value: 0 });
    assert_eq!(
        s.passes()[0].stencil_load(),
        StencilLoad::Clear { value: 0 }
    );
}

// The presented multisampled back buffer.

#[test]
fn a_presented_multisampled_back_buffer_resolves_without_storing_its_samples() {
    let mut s = fresh_multisampled();
    s.emit_command(dummy_draw());
    break_pass_via(&mut s, tex(0x3000));
    s.set_color_msaa(msaa_backbuffer(), msaa_backbuffer_srgb(), 4);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.finalize_store_actions(false);
    let last = s.passes().last().unwrap();
    assert_eq!(last.color_resolve_texture(), backbuffer());
    assert_eq!(last.color_store(), StoreAction::DontCare);
    assert_eq!(
        s.passes()[0].color_store(),
        StoreAction::Store,
        "the earlier pass keeps its samples for the last one"
    );
}

#[test]
fn the_multisampled_back_buffer_keeps_its_samples_unless_presented_under_discard() {
    let mut flush = fresh_multisampled();
    flush.emit_command(dummy_draw());
    flush.end_current_pass("test");
    flush.finalize_store_actions(true);
    assert_eq!(
        flush.passes()[0].color_store(),
        StoreAction::Store,
        "a mid-frame flush: a later pass may load the samples"
    );

    let mut preserved = PassState::new();
    preserved.reset_frame(&FrameReset {
        backbuffer: backbuffer(),
        backbuffer_srgb: backbuffer_srgb(),
        backbuffer_msaa: msaa_backbuffer(),
        backbuffer_msaa_srgb: msaa_backbuffer_srgb(),
        backbuffer_sample_count: 4,
        backbuffer_size: BB_SIZE,
        backbuffer_format: BB_FORMAT,
        backbuffer_contents: BackbufferContents::Preserved,
        depth_texture: depth(),
        depth_size: BB_SIZE,
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    preserved.emit_command(dummy_draw());
    preserved.end_current_pass("test");
    preserved.finalize_store_actions(false);
    assert_eq!(
        preserved.passes()[0].color_store(),
        StoreAction::Store,
        "a copy or flip swap effect keeps the back buffer across Present"
    );

    let rt = tex(0x3400);
    let mut offscreen = fresh();
    offscreen.set_color_render_target(rt, BB_SIZE.0, BB_SIZE.1, BB_FORMAT, RenderScale::IDENTITY);
    offscreen.set_color_msaa(tex(0x3401), MetalHandle::NULL, 4);
    offscreen.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
    offscreen.emit_command(dummy_draw());
    offscreen.end_current_pass("test");
    offscreen.finalize_store_actions(false);
    assert_eq!(offscreen.passes()[0].color_resolve_texture(), rt);
    assert_eq!(
        offscreen.passes()[0].color_store(),
        StoreAction::Store,
        "a multisampled render target keeps its samples across Present"
    );
}

// ── Rules G and F: attachments and passes a clear-only pass leaves unchanged ──

#[test]
fn rule_g_strips_a_loading_colour_target_from_a_depth_clear_pass() {
    // Clear(ZBUFFER) with nothing drawn before the depth surface changes
    // lands as a clear-only pass on the bound colour target and depth. The
    // colour target only loads there, so its load and store change nothing.
    let target = tex(0x3000);
    let other_depth = tex(0x9100);
    let mut s = fresh();
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.clear_depth(f32::to_bits(1.0)), DepthClearOutcome::Folded);
    s.set_depth_stencil_attachment(other_depth, BB_SIZE, false, false);
    // A later pass samples the cleared depth, so its store survives Rule B.
    s.emit_command(Command::set_fragment_texture(depth().raw(), 0));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 3);
    assert_eq!(s.passes()[1].color_texture(), target);
    assert_eq!(s.passes()[1].color_load(), ColorLoad::Load);

    apply_submit_rules(&mut s);

    assert_eq!(s.passes().len(), 3);
    let clear = &s.passes()[1];
    assert!(matches!(clear.depth_load(), DepthLoad::Clear { .. }));
    assert_eq!(clear.depth_store(), StoreAction::Store);
    assert_eq!(
        clear.color_attachment_texture(),
        MetalHandle::NULL,
        "the colour target the clear leaves unchanged is not attached"
    );
    assert_eq!(s.passes()[0].color_texture(), target, "draw passes keep it");
    assert_eq!(s.passes()[2].color_texture(), target);
}

/// `count` passes on the back buffer and the depth surface, each closed with nothing in it.
fn clear_only_passes(count: usize) -> PassState {
    let mut s = fresh();
    for _ in 0..count {
        s.ensure_pass_open();
        s.end_current_pass("test");
    }
    s
}

#[test]
fn rule_g_strips_unwritten_colour_targets_whatever_their_store() {
    let extra = tex(0x3300);
    let mut s = clear_only_passes(3);
    // A `DontCare` load (the back buffer's first use) stored, a `Load`
    // stored, and a render target 1 that loads beside a stored clear of
    // render target 0.
    assert_eq!(s.passes[0].color_load, ColorLoad::DontCare);
    assert_eq!(s.passes[1].color_load, ColorLoad::Load);
    s.passes[2].color_load = ColorLoad::Clear {
        r: 1,
        g: 2,
        b: 3,
        a: 4,
    };
    s.passes[2].extra_color[0] = PassColorAttachment {
        texture: extra,
        size: BB_SIZE,
        load: ColorLoad::Load,
        store: StoreAction::Store,
        ..PassColorAttachment::NONE
    };
    for pass in &s.passes {
        assert_eq!(pass.color_store, StoreAction::Store);
    }

    s.strip_dead_color_in_clear_only_passes();

    assert!(
        s.passes[0].color_texture.is_null(),
        "DontCare load stripped"
    );
    assert!(s.passes[1].color_texture.is_null(), "Load stripped");
    assert_eq!(s.passes[2].color_texture, backbuffer(), "stored clear kept");
    assert!(
        !s.passes[2].extra_color[0].is_bound(),
        "loading extra stripped"
    );
    for pass in &s.passes {
        assert_eq!(pass.depth_texture, depth(), "depth stays attached");
    }
}

#[test]
fn rule_g_keeps_the_colour_targets_a_clear_only_pass_still_needs() {
    let mut s = clear_only_passes(3);
    // A resolve writes the single-sample twin.
    s.passes[0].color_resolve_texture = backbuffer();
    // Without a depth attachment render target 0 is the pass's only one.
    s.passes[1].depth_texture = MetalHandle::NULL;
    // A leading blit is work of its own; the pass is not clear-only.
    s.passes[2].leading_blits.push(dummy_blit());

    s.strip_dead_color_in_clear_only_passes();

    for (index, pass) in s.passes.iter().enumerate() {
        assert_eq!(
            pass.color_texture,
            backbuffer(),
            "pass {index} keeps colour"
        );
    }
}

#[test]
fn rule_f_culls_a_pass_holding_only_a_viewport_and_a_visibility_mode() {
    // `Issue(BEGIN)` right after a draw's pass closed opens a pass for its
    // Counting mode, and a render-target change closes it again with nothing
    // drawn. The slot it armed counts nothing whether or not the pass runs.
    let other = tex(0x3000);
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.emit_command(Command::set_visibility_result_mode(
        VisibilityResultMode::Counting,
        8,
    ));
    s.set_color_render_target(
        other,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 3);
    assert!(s.passes()[1].has_counting_visibility());
    assert_eq!(s.passes()[1].commands().len(), 2, "viewport and mode");

    apply_submit_rules(&mut s);

    assert_eq!(s.passes().len(), 2, "the empty pass is culled");
    assert!(s.passes().iter().all(|p| !p.has_counting_visibility()));
    assert_eq!(s.passes()[0].color_texture(), backbuffer());
    assert_eq!(s.passes()[1].color_texture(), other);
}

#[test]
fn rule_f_culls_a_pass_whose_draw_was_dropped_after_it_opened() {
    // A draw that bound its state and then failed leaves the pass it opened
    // with state commands only; loading and storing the attachments around
    // them changes nothing.
    let other = tex(0x3000);
    let mut s = fresh();
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        other,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(set_pso(0xAB00));
    s.emit_command(Command::set_scissor_rect(0, 0, 8, 8));
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.emit_command(Command::set_fragment_texture(other.raw(), 0));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 3);

    s.finalize_load_actions();
    s.finalize_store_actions(false);
    s.strip_dead_color_in_clear_only_passes();
    s.cull_dead_clear_only_passes();

    assert_eq!(s.passes().len(), 2);
    assert!(s.passes().iter().all(|p| p.color_texture() == backbuffer()));
}

#[test]
fn rule_f_keeps_clear_only_passes_that_write_something() {
    let mut s = clear_only_passes(4);
    // A stored colour clear, a stored stencil clear, a resolve, a blit.
    s.passes[0].color_load = ColorLoad::Clear {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    s.passes[1].stencil_load = StencilLoad::Clear { value: 1 };
    s.passes[1].depth_flags |= PassDepthFlags::HAS_STENCIL;
    s.passes[1].stencil_store = StoreAction::Store;
    s.passes[2].color_resolve_texture = backbuffer();
    s.passes[3].leading_blits.push(dummy_blit());

    s.strip_dead_color_in_clear_only_passes();
    s.cull_dead_clear_only_passes();

    assert_eq!(s.passes().len(), 4);
}

#[test]
fn rule_f_culls_a_clear_whose_every_store_is_discarded() {
    let mut s = clear_only_passes(2);
    // A colour clear Rule C discards beside a depth plane that only loads,
    // and a depth clear Rule B discards beside a colour target that loads.
    s.passes[0].color_load = ColorLoad::Clear {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    s.passes[0].color_store = StoreAction::DontCare;
    s.passes[0].depth_load = DepthLoad::Load;
    s.passes[1].depth_load = DepthLoad::Clear {
        value: f32::to_bits(1.0),
    };
    s.passes[1].depth_store = StoreAction::DontCare;

    s.cull_dead_clear_only_passes();

    assert!(s.passes().is_empty());
}

// ── Rule J: adjacent passes on identical attachments ──────────

/// Draw into `target`, bind `detour` and bind `target` again with nothing in between, then draw.
fn round_trip(
    s: &mut PassState,
    target: MetalHandle<MTLTextureKind>,
    detour: MetalHandle<MTLTextureKind>,
) {
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        detour,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
}

#[test]
fn rule_j_joins_the_two_passes_a_render_target_round_trip_leaves() {
    let target = tex(0x3000);
    let mut s = fresh();
    round_trip(&mut s, target, tex(0x3100));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(
        s.passes().len(),
        2,
        "the round trip splits the target's pass"
    );
    assert_eq!(s.passes()[1].color_load(), ColorLoad::Load);

    apply_submit_rules(&mut s);

    assert_eq!(s.passes().len(), 1);
    let pass = &s.passes()[0];
    assert_eq!(pass.color_texture(), target);
    assert_eq!(pass.depth_texture(), depth());
    let kinds: Vec<u32> = pass.commands().iter().map(|c| c.cmd).collect();
    let viewport = CommandType::SetViewport as u32;
    let draw = CommandType::DrawPrimitives as u32;
    assert_eq!(
        kinds,
        [viewport, draw, viewport, draw],
        "nothing to restore"
    );
}

#[test]
fn rule_j_joins_a_run_of_passes_into_one() {
    let target = tex(0x3000);
    let mut s = fresh();
    round_trip(&mut s, target, tex(0x3100));
    s.emit_command(dummy_draw());
    round_trip(&mut s, target, tex(0x3200));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 3);

    apply_submit_rules(&mut s);

    assert_eq!(s.passes().len(), 1);
    let draws = s.passes()[0]
        .commands()
        .iter()
        .filter(|c| c.is_draw())
        .count();
    assert_eq!(draws, 4, "the draw of each half, in order");
}

#[test]
fn rule_j_restores_the_fresh_encoder_state_the_second_pass_reads() {
    // The first pass leaves every state the dedup cache starts at its fresh
    // value changed; the second draws without setting them, so a fresh
    // encoder's values go back at the join.
    let target = tex(0x3000);
    let mut s = fresh();
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(Command::set_triangle_fill_mode(TriangleFillMode::Lines));
    s.emit_command(Command::set_depth_bias(0.0, 2.0));
    s.emit_command(Command::set_stencil_reference(7));
    s.emit_command(Command::set_blend_color(0.5, 0.5, 0.5, 0.5));
    s.emit_command(Command::set_visibility_result_mode(
        VisibilityResultMode::Counting,
        16,
    ));
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        tex(0x3100),
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    #[cfg(debug_assertions)]
    let before = s.debug_record_draw_states();

    apply_submit_rules(&mut s);

    assert_eq!(s.passes().len(), 1);
    let pass = &s.passes()[0];
    assert!(pass.has_counting_visibility());
    let first_draw = pass.commands().iter().position(Command::is_draw).unwrap();
    let join = &pass.commands()[first_draw + 1..first_draw + 6];
    let expected = [
        Command::set_triangle_fill_mode(TriangleFillMode::Fill),
        Command::set_depth_bias(0.0, 0.0),
        Command::set_stencil_reference(0),
        Command::set_blend_color(0.0, 0.0, 0.0, 0.0),
        Command::set_visibility_result_mode(VisibilityResultMode::Disabled, 16),
    ];
    for (got, want) in join.iter().zip(&expected) {
        assert!(same_command(got, want), "join restores {}", want.cmd);
    }
    assert_eq!(
        pass.commands()[first_draw + 6].cmd,
        CommandType::SetViewport as u32,
        "the second pass's own commands follow the join"
    );
    #[cfg(debug_assertions)]
    s.debug_assert_draw_states_preserved(&before, &FxHashMap::default());
}

#[test]
fn rule_j_restores_nothing_the_second_pass_sets_before_it_draws() {
    let target = tex(0x3000);
    let mut s = fresh();
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(Command::set_stencil_reference(7));
    s.emit_command(Command::set_cull_mode(CullMode::Back));
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        tex(0x3100),
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(Command::set_stencil_reference(3));
    s.emit_command(Command::set_cull_mode(CullMode::Front));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    #[cfg(debug_assertions)]
    let before = s.debug_record_draw_states();
    let second_len = s.passes()[1].commands().len();
    let first_len = s.passes()[0].commands().len();

    apply_submit_rules(&mut s);

    assert_eq!(s.passes().len(), 1);
    assert_eq!(s.passes()[0].commands().len(), first_len + second_len);
    #[cfg(debug_assertions)]
    s.debug_assert_draw_states_preserved(&before, &FxHashMap::default());
}

#[test]
fn rule_j_keeps_passes_apart_when_the_second_reads_state_it_cannot_restore() {
    // The first pass culls back faces; the second draws before any cull mode
    // is set and so relies on a fresh encoder culling nothing.
    let target = tex(0x3000);
    let mut s = fresh();
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(Command::set_cull_mode(CullMode::Back));
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        tex(0x3100),
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(dummy_draw());
    s.end_current_pass("test");

    apply_submit_rules(&mut s);

    assert_eq!(s.passes().len(), 2);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "pass rules changed the triangle fill mode a surviving draw sees")]
fn draw_state_check_catches_a_join_that_restores_nothing() {
    let target = tex(0x3000);
    let mut s = fresh();
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(Command::set_triangle_fill_mode(TriangleFillMode::Lines));
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        tex(0x3100),
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let before = s.debug_record_draw_states();
    let mut next = s.passes.remove(1);
    s.passes[0].absorb(&mut next, &PassJoin::new());
    s.debug_assert_draw_states_preserved(&before, &FxHashMap::default());
}

#[cfg(debug_assertions)]
#[test]
fn draw_state_check_ignores_bindings_the_first_half_leaves_behind() {
    // The first pass binds a texture and a vertex buffer the second pass's
    // draw never reads; the dedup cache let the second pass bind only what
    // it reads, so the leftovers are no change to what its draw sees.
    let target = tex(0x3000);
    let mut s = fresh();
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(Command::set_fragment_texture(0x7700, 3));
    s.emit_command(Command::set_vertex_buffer(0x7800, 0, 1));
    s.emit_command(dummy_draw());
    s.set_color_render_target(
        tex(0x3100),
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(Command::set_fragment_texture(0x7900, 0));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let before = s.debug_record_draw_states();

    apply_submit_rules(&mut s);

    assert_eq!(s.passes().len(), 1);
    s.debug_assert_draw_states_preserved(&before, &FxHashMap::default());
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(
    expected = "pass rules changed the fragment textures and samplers a surviving draw sees"
)]
fn draw_state_check_still_catches_a_binding_the_draw_had() {
    // Masking the leftovers must not hide a slot the draw did have bound.
    let target = tex(0x3000);
    let mut s = fresh();
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(Command::set_fragment_texture(0x7700, 0));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let before = s.debug_record_draw_states();
    s.passes[0].commands[1] = Command::set_fragment_texture(0x7900, 0);
    s.debug_assert_draw_states_preserved(&before, &FxHashMap::default());
}

/// Assert a render-target round trip with `split` before the second pass keeps two passes.
fn assert_split_keeps_passes_apart(name: &str, split: impl FnOnce(&mut PassState)) {
    let mut s = fresh();
    round_trip(&mut s, tex(0x3000), tex(0x3100));
    split(&mut s);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 2, "{name}: two passes recorded");

    apply_submit_rules(&mut s);

    assert_eq!(s.passes().len(), 2, "{name}: the passes stay apart");
}

#[test]
fn rule_j_keeps_passes_apart_across_a_boundary_the_join_would_change() {
    let target = tex(0x3000);
    assert_split_keeps_passes_apart("a clear", |s| {
        s.clear_color(1, 2, 3, 4);
    });
    assert_split_keeps_passes_apart("a leading blit", |s| {
        s.push_pending_leading_blit(dummy_blit());
    });
    assert_split_keeps_passes_apart("a sample of the target", |s| {
        s.emit_command(Command::set_fragment_texture(target.raw(), 0));
    });
    assert_split_keeps_passes_apart("another mip level", |s| {
        s.set_color_render_target_subresource(
            target,
            &TargetExtent::whole(RenderScale::IDENTITY, (BB_SIZE.0, BB_SIZE.1)),
            RT_FORMAT,
            (0, 1),
        );
    });
    assert_split_keeps_passes_apart("a multisampled companion", |s| {
        s.set_color_msaa(tex(0x3001), MetalHandle::NULL, 4);
    });
    assert_split_keeps_passes_apart("another depth level", |s| {
        s.set_depth_stencil_attachment_level(depth(), 1, BB_SIZE, false, false);
    });
}

#[test]
fn rule_j_keeps_a_resolving_pass_apart_and_moves_a_later_resolve_onto_the_join() {
    let rt = tex(0x3000);
    // The back buffer resolves where a read between the passes pulled it
    // forward, so that pass ends there.
    let mut s = fresh_multisampled();
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.note_msaa_read(backbuffer());
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    apply_submit_rules(&mut s);
    assert_eq!(s.passes().len(), 2, "a resolving pass is not joined");

    // Without that read, the last use takes the one resolve and the join
    // carries it.
    let mut s = fresh_multisampled();
    s.emit_command(dummy_draw());
    s.set_color_render_target(rt, BB_SIZE.0, BB_SIZE.1, RT_FORMAT, RenderScale::IDENTITY);
    s.set_color_render_target(
        backbuffer(),
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        s.render_scale,
    );
    s.set_color_msaa(msaa_backbuffer(), msaa_backbuffer_srgb(), 4);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 2);
    apply_submit_rules(&mut s);
    assert_eq!(s.passes().len(), 1);
    assert_eq!(s.passes()[0].color_resolve_texture(), backbuffer());
}

#[test]
fn rule_j_leaves_the_upload_prefix_alone() {
    let target = tex(0x5000);
    let upload = UploadPassTarget {
        texture: target,
        subresource: (0, 0),
        size: BB_SIZE,
        format: BB_FORMAT,
        rect: (0, 0, 8, 8),
    };
    let mut s = fresh();
    s.push_upload_pass(&upload, &[dummy_draw()], Vec::new());
    s.set_color_render_target(
        target,
        BB_SIZE.0,
        BB_SIZE.1,
        BB_FORMAT,
        RenderScale::IDENTITY,
    );
    s.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes()[1].color_load(), ColorLoad::Load);

    s.merge_adjacent_identical_passes();

    assert_eq!(s.passes().len(), 2);
    assert_eq!(s.upload_pass_count(), 1);
}

#[test]
fn rule_j_moves_the_second_pass_clear_quad_ranges_with_its_commands() {
    let target = tex(0x3000);
    let mut s = fresh();
    round_trip(&mut s, target, tex(0x3100));
    s.emit_command(dummy_draw());
    let start = s.open_color_clear_quad_block();
    s.emit_command(set_pso(0xC1EA));
    s.emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 6));
    s.close_color_clear_quad_block(start);
    s.end_current_pass("test");
    let block: Vec<(u32, u64, u64)> = {
        let pass = &s.passes()[1];
        let (start, end) = pass.color_clear_quad_ranges()[0];
        pass.commands()[start..end]
            .iter()
            .map(|c| (c.cmd, c.param_b, c.param_c))
            .collect()
    };

    s.merge_adjacent_identical_passes();

    assert_eq!(s.passes().len(), 1);
    let pass = &s.passes()[0];
    assert_eq!(pass.color_clear_quad_ranges().len(), 1);
    let (start, end) = pass.color_clear_quad_ranges()[0];
    let moved: Vec<(u32, u64, u64)> = pass.commands()[start..end]
        .iter()
        .map(|c| (c.cmd, c.param_b, c.param_c))
        .collect();
    assert_eq!(moved, block);
}

#[test]
fn every_fresh_state_command_reads_as_a_fresh_value() {
    let changed = [
        Command::set_triangle_fill_mode(TriangleFillMode::Lines),
        Command::set_depth_bias(1.0, 0.0),
        Command::set_stencil_reference(1),
        Command::set_blend_color(1.0, 1.0, 1.0, 1.0),
        Command::set_visibility_result_mode(VisibilityResultMode::Counting, 24),
    ];
    for last in &changed {
        assert!(!sets_fresh_value(last), "{} is a change", last.cmd);
        let restore = fresh_state_command(last).expect("a restorable state");
        assert_eq!(restore.cmd, last.cmd);
        assert!(sets_fresh_value(&restore), "{} restores", last.cmd);
    }
    for kind in [
        CommandType::SetRenderPipelineState,
        CommandType::SetViewport,
        CommandType::SetDepthStencilState,
        CommandType::SetCullMode,
        CommandType::SetScissorRect,
    ] {
        let cmd = Command {
            cmd: kind as u32,
            param_a: 0,
            param_b: 1,
            param_c: 0,
            param_d: 0,
        };
        assert!(
            fresh_state_command(&cmd).is_none(),
            "{kind:?} has no restore"
        );
    }
}

#[test]
fn rule_j_joins_across_a_stencil_plane_nothing_has_written() {
    // The stencil plane of a D24S8 surface no draw writes loads and stores
    // `DontCare` on every pass, so it cannot keep a round trip's passes apart.
    let target = tex(0x3000);
    let mut s = fresh_with_stencil();
    round_trip(&mut s, target, tex(0x3100));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 2);

    apply_submit_rules(&mut s);

    assert_eq!(s.passes().len(), 1);
}

#[test]
fn rule_j_keeps_passes_apart_when_a_loaded_depth_plane_was_not_stored() {
    let target = tex(0x3000);
    for (name, discard) in [
        (
            "depth",
            (|p: &mut Pass| {
                p.depth_store = StoreAction::DontCare;
            }) as fn(&mut Pass),
        ),
        ("stencil", |p: &mut Pass| {
            p.stencil_store = StoreAction::DontCare;
        }),
    ] {
        let mut s = fresh_with_stencil();
        round_trip(&mut s, target, tex(0x3100));
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.passes[1].depth_load = DepthLoad::Load;
        s.passes[1].stencil_load = StencilLoad::Load;
        discard(&mut s.passes[0]);

        s.merge_adjacent_identical_passes();

        assert_eq!(s.passes().len(), 2, "{name}: the passes stay apart");
    }
}

/// A bound mip level of a scaled target records the extent Metal allocated for it.
///
/// Metal sizes a level from the scaled base, `max(1, dimension(base) >> level)`,
/// which can differ by a texel from the scale of the level's reported size. A
/// full-level viewport and scissor have to land on exactly that extent, and the
/// coverage tests have to see the level as covered.
#[test]
fn a_scaled_mip_level_binds_at_the_extent_metal_allocated() {
    let rt = tex(0x3000);
    let mut s = fresh();
    for percent in [50, 67, 75, 99, 100] {
        let scale = RenderScale::from_percent(percent);
        for base in 1..=2048u32 {
            let base_texture = (scale.dimension(base), scale.dimension(base + 1));
            for level in 0..=3u32 {
                let logical = ((base >> level).max(1), ((base + 1) >> level).max(1));
                let texture = (
                    (base_texture.0 >> level).max(1),
                    (base_texture.1 >> level).max(1),
                );
                let extent = TargetExtent::mip_level(scale, logical, base_texture, level);
                s.set_color_render_target_subresource(rt, &extent, RT_FORMAT, (0, level));
                let what = format!("{percent}% of {base} at level {level}");
                assert_eq!(s.current_color_size(), texture, "{what}: recorded extent");
                s.set_viewport(0, 0, logical.0, logical.1, 0.0, 1.0);
                assert_eq!(
                    s.effective_viewport(),
                    (0, 0, texture.0, texture.1),
                    "{what}: full-level viewport",
                );
                assert!(s.viewport_covers_color_attachment(), "{what}: coverage");
                assert_eq!(
                    s.resolved_scissor_rect(true, [0, 0, logical.0, logical.1]),
                    (0, 0, texture.0, texture.1),
                    "{what}: full-level scissor",
                );
            }
        }
    }
}

// ── A small render target 0 over a larger depth surface. The frame's
// ── depth surface is the back buffer's 640x480 one, at the identity scale.

/// A 1x1 render target 0 bound over the frame's depth surface.
fn tiny() -> MetalHandle<MTLTextureKind> {
    tex(0x7000)
}

/// A frame with the 1x1 [`tiny`] target over the 640x480 depth surface, viewport over the depth.
fn tiny_over_depth() -> PassState {
    let mut s = fresh();
    s.set_color_render_target(tiny(), 1, 1, RT_FORMAT, RenderScale::IDENTITY);
    s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
    s
}

#[test]
fn rule_h_keeps_a_colour_target_smaller_than_the_depth_surface() {
    // Stripping the 64x64 target would make the depth surface alone set the
    // render area, so the masked draws would reach texels outside the
    // target's 64x64 that D3D9 never rasterizes.
    let mut s = fresh();
    s.set_color_render_target(tex(0x7100), 64, 64, RT_FORMAT, RenderScale::IDENTITY);
    s.note_draw_color_write_mask(0);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    s.strip_color_from_no_color_draw_passes(&alt);
    assert_eq!(
        s.passes()[0].color_texture(),
        tex(0x7100),
        "the target smaller than the depth surface stays attached"
    );
}

#[test]
fn rule_h_strips_a_colour_target_the_size_of_the_depth_surface() {
    let mut s = fresh();
    s.set_color_render_target(
        tex(0x7100),
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.note_draw_color_write_mask(0);
    s.emit_command(set_pso(PSO_WITH));
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let mut alt = FxHashMap::default();
    alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
    s.strip_color_from_no_color_draw_passes(&alt);
    assert_eq!(s.passes()[0].color_texture(), MetalHandle::NULL);
}

#[test]
fn rule_e_keeps_a_depth_clear_out_of_a_pass_smaller_than_the_depth_surface() {
    // The clear-only pass clears the whole depth surface; the next pass on it
    // rasterizes only the 1x1 target's area, so folding the clear into that
    // pass's load action would leave the rest of the surface uncleared.
    let z = f32::to_bits(0.25);
    let mut s = fresh();
    s.pending_depth_clear = Some(z);
    s.push_depth_clear_pass();
    s.set_color_render_target(tiny(), 1, 1, RT_FORMAT, RenderScale::IDENTITY);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();
    assert_eq!(s.passes().len(), 2, "the depth clear keeps its own pass");
    assert_eq!(s.passes()[0].color_texture(), MetalHandle::NULL);
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::Clear { value: z });
    assert_eq!(s.passes()[1].depth_load(), DepthLoad::Load);
}

#[test]
fn rule_e_still_folds_a_depth_clear_into_a_pass_the_size_of_the_depth_surface() {
    let z = f32::to_bits(0.25);
    let mut s = fresh();
    s.pending_depth_clear = Some(z);
    s.push_depth_clear_pass();
    s.set_color_render_target(
        tex(0x7100),
        BB_SIZE.0,
        BB_SIZE.1,
        RT_FORMAT,
        RenderScale::IDENTITY,
    );
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    s.coalesce_clear_only_passes();
    assert_eq!(
        s.passes().len(),
        1,
        "the depth clear folds into the next pass"
    );
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::Clear { value: z });
}

#[test]
fn a_pending_depth_clear_meeting_a_1x1_target_gets_a_depth_only_pass() {
    let z = f32::to_bits(0.5);
    let mut s = tiny_over_depth();
    assert!(matches!(s.clear_depth(z), DepthClearOutcome::Folded));
    // A draw that writes the 1x1 target opens its pass.
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 2, "a depth-only clear pass comes first");
    assert_eq!(s.passes()[0].color_texture(), MetalHandle::NULL);
    assert_eq!(s.passes()[0].depth_texture(), depth());
    assert_eq!(s.passes()[0].depth_load(), DepthLoad::Clear { value: z });
    assert_eq!(s.passes()[1].color_texture(), tiny());
    assert_eq!(s.passes()[1].depth_load(), DepthLoad::Load);
}

#[test]
fn a_whole_depth_clear_ends_an_open_pass_on_a_1x1_target() {
    // A quad painted into the open pass would be clipped to the 1x1 area.
    let z = f32::to_bits(0.5);
    let mut s = tiny_over_depth();
    s.emit_command(dummy_draw());
    assert!(matches!(s.clear_depth(z), DepthClearOutcome::Folded));
    assert!(s.current_pass_closed(), "the pass on the 1x1 target ended");
    assert_eq!(s.pending_depth_clear, Some(z));
}

#[test]
fn a_region_depth_clear_over_a_1x1_target_opens_a_depth_only_pass() {
    let mut s = tiny_over_depth();
    let (has_color, _) = s
        .begin_region_depth_stencil_clear()
        .expect("a depth surface is bound");
    assert!(!has_color, "the quads go to a pass without colour");
    let pass = s.passes().last().expect("a pass is open");
    assert_eq!(pass.color_texture(), MetalHandle::NULL);
    assert_eq!(pass.depth_texture(), depth());
    assert_eq!(
        pass.depth_size, BB_SIZE,
        "the depth surface sets the extent"
    );
}

// ── Leaving a 1x1 render target 0 out so the depth surface sets the extent.

fn candidate(s: &PassState) -> &'static str {
    match s.rt0_drop_candidate() {
        Rt0DropCandidate::No => "no",
        Rt0DropCandidate::Yes => "yes",
        Rt0DropCandidate::ScaledDepth => "scaled depth",
    }
}

#[test]
fn a_lone_1x1_target_over_a_larger_unscaled_depth_surface_is_a_candidate() {
    assert_eq!(candidate(&tiny_over_depth()), "yes");
}

#[test]
fn every_other_binding_shape_keeps_render_target_0() {
    let mut s = fresh();
    s.set_color_render_target(tex(0x7100), 2, 2, RT_FORMAT, RenderScale::IDENTITY);
    assert_eq!(candidate(&s), "no", "a 2x2 target");

    // Level 8 of a 256x256 texture reports 1x1 but is not a 1x1 resource.
    let mut s = fresh();
    s.set_color_render_target_subresource(
        tex(0x7200),
        &TargetExtent::mip_level(RenderScale::IDENTITY, (1, 1), (256, 256), 8),
        RT_FORMAT,
        (0, 8),
    );
    assert_eq!(
        candidate(&s),
        "no",
        "the 1x1 last level of a larger texture"
    );

    let mut s = tiny_over_depth();
    s.set_extra_color_render_target(1, Some(slot(tex(0x7300), (1, 1))));
    assert_eq!(candidate(&s), "no", "a render target 1 bound");

    let mut s = tiny_over_depth();
    s.set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
    assert_eq!(candidate(&s), "no", "no depth surface");

    let mut s = tiny_over_depth();
    s.set_depth_stencil_attachment(tex(0x7400), (1, 1), false, false);
    s.set_depth_unscaled(true);
    assert_eq!(candidate(&s), "no", "a 1x1 depth surface");

    let mut s = tiny_over_depth();
    s.set_color_msaa(tex(0x7500), MetalHandle::NULL, 4);
    assert_eq!(
        candidate(&s),
        "no",
        "a depth surface at another sample count"
    );

    let mut s = fresh_scaled();
    s.set_color_render_target(tiny(), 1, 1, RT_FORMAT, RenderScale::from_percent(50));
    assert_eq!(candidate(&s), "no", "a scaled render target 0");
}

#[test]
fn a_depth_surface_nobody_vouched_for_reads_as_scaled() {
    let mut s = tiny_over_depth();
    s.set_depth_stencil_attachment(tex(0x7400), (256, 256), false, false);
    assert!(
        !s.current_depth_unscaled(),
        "a new depth binding starts scaled"
    );
    assert_eq!(candidate(&s), "scaled depth");
    s.set_depth_unscaled(true);
    assert_eq!(candidate(&s), "yes");
    // Rebinding the same attachment keeps the declaration.
    s.set_depth_stencil_attachment(tex(0x7400), (256, 256), false, false);
    assert!(s.current_depth_unscaled());
}

#[test]
fn the_frame_depth_surface_is_unscaled_only_at_the_identity_scale() {
    assert!(fresh().current_depth_unscaled());
    let mut s = fresh_scaled();
    assert!(!s.current_depth_unscaled());
    s.set_color_render_target(tiny(), 1, 1, RT_FORMAT, RenderScale::IDENTITY);
    assert_eq!(candidate(&s), "scaled depth");
}

#[test]
fn a_dropped_pass_attaches_the_depth_surface_alone_at_its_extent() {
    let mut s = tiny_over_depth();
    s.set_rt0_dropped(true);
    assert!(!s.pass_binds_color());
    assert_eq!(s.target_extent().texture(), BB_SIZE);
    assert_eq!(s.effective_viewport(), (0, 0, BB_SIZE.0, BB_SIZE.1));
    s.emit_command(dummy_draw());
    let pass = s.passes().last().expect("the draw opened a pass");
    assert_eq!(pass.color_texture(), MetalHandle::NULL);
    assert_eq!(pass.depth_texture(), depth());
    assert_eq!(pass.color_size(), BB_SIZE);
    assert_eq!(pass.depth_size, BB_SIZE);
    s.end_current_pass("test");
    assert!(!s.rt0_dropped(), "the decision ends with its pass");
    assert_eq!(s.target_extent().texture(), (1, 1));
}

#[test]
fn ending_a_pass_clears_the_decision_with_or_without_an_open_pass() {
    let mut s = tiny_over_depth();
    s.set_rt0_dropped(true);
    s.end_current_pass("test");
    assert!(!s.rt0_dropped(), "no pass was open");
    s.set_rt0_dropped(true);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert!(!s.rt0_dropped(), "a pass was open");
}

#[test]
fn changing_the_decision_ends_the_open_pass() {
    let mut s = tiny_over_depth();
    s.set_rt0_dropped(true);
    s.emit_command(dummy_draw());
    s.set_rt0_dropped(false);
    s.emit_command(dummy_draw());
    s.set_rt0_dropped(true);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    let colors: Vec<_> = s.passes().iter().map(Pass::color_texture).collect();
    assert_eq!(colors, [MetalHandle::NULL, tiny(), MetalHandle::NULL]);
    // An unchanged decision keeps the pass.
    let mut s = tiny_over_depth();
    s.set_rt0_dropped(true);
    s.emit_command(dummy_draw());
    s.set_rt0_dropped(true);
    s.emit_command(dummy_draw());
    assert_eq!(s.passes().len(), 1);
}

#[test]
fn a_pending_colour_clear_lands_in_its_own_pass_ahead_of_a_dropped_pass() {
    let z = f32::to_bits(0.5);
    let mut s = tiny_over_depth();
    assert!(matches!(
        s.clear_color(1, 2, 3, 4),
        ColorClearOutcome::Folded
    ));
    assert!(matches!(s.clear_depth(z), DepthClearOutcome::Folded));
    s.push_pending_leading_blit(dummy_blit());
    s.set_rt0_dropped(true);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes().len(), 2);
    let clear = &s.passes()[0];
    assert_eq!(clear.color_texture(), tiny());
    assert_eq!(
        clear.color_load(),
        ColorLoad::Clear {
            r: 1,
            g: 2,
            b: 3,
            a: 4
        }
    );
    assert_eq!(
        clear.depth_texture(),
        MetalHandle::NULL,
        "the colour clear pass has no depth"
    );
    assert_eq!(
        clear.leading_blits().len(),
        1,
        "the first pass pushed takes the blits"
    );
    assert!(!clear.commands().iter().any(Command::is_draw));
    let dropped = &s.passes()[1];
    assert_eq!(dropped.color_texture(), MetalHandle::NULL);
    assert_eq!(dropped.depth_load(), DepthLoad::Clear { value: z });
    assert!(dropped.leading_blits().is_empty());
}

#[test]
fn the_colour_only_clear_pass_carries_the_multisampled_companion() {
    let mut s = tiny_over_depth();
    s.set_depth_stencil_attachment(tex(0x7400), (256, 256), false, false);
    s.set_depth_sample_count(4);
    s.set_depth_unscaled(true);
    s.set_color_msaa(tex(0x7500), MetalHandle::NULL, 4);
    s.clear_color(1, 2, 3, 4);
    s.set_rt0_dropped(true);
    s.emit_command(dummy_draw());
    s.end_current_pass("test");
    assert_eq!(s.passes()[0].color_attachment_texture(), tex(0x7500));
    assert_eq!(s.passes()[1].color_texture(), MetalHandle::NULL);
}

#[test]
fn a_whole_depth_clear_quad_in_a_dropped_pass_declares_no_colour() {
    let mut s = tiny_over_depth();
    s.set_rt0_dropped(true);
    s.emit_command(dummy_draw());
    match s.clear_depth(f32::to_bits(1.0)) {
        DepthClearOutcome::EmitQuad { has_color, .. } => {
            assert!(!has_color, "the quad matches the pass without colour");
        }
        _ => panic!("a clear after a draw paints a quad in the open pass"),
    }
    assert!(!s.current_pass_closed(), "the dropped pass stays open");
}

#[test]
fn a_colour_clear_ends_a_dropped_pass_first() {
    let mut s = tiny_over_depth();
    s.set_rt0_dropped(true);
    s.emit_command(dummy_draw());
    s.end_rt0_dropped_pass("test");
    assert!(s.current_pass_closed());
    assert!(s.pass_binds_color());
    assert_eq!(s.target_extent().texture(), (1, 1));
    // A pass that attaches render target 0 is left open.
    let mut s = tiny_over_depth();
    s.emit_command(dummy_draw());
    s.end_rt0_dropped_pass("test");
    assert!(!s.current_pass_closed());
}
