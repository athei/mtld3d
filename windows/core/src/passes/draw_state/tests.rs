//! Unit tests for the draw-state check: every state class it fingerprints catches a change.
//!
//! Each case records one pass, swaps a command the way a faulty rule would, and runs the
//! check. The fingerprints are sums kept current per command, so the cases also cover a slot
//! set twice, a slot moved to its neighbour, and bindings masked out of the comparison.

use std::panic::{AssertUnwindSafe, catch_unwind};

use mtld3d_shared::{
    NullTextureKind,
    mtl::{CullMode, PrimitiveType, TriangleFillMode, VisibilityResultMode},
};

use super::*;

fn draw() -> Command {
    Command::draw_primitives(PrimitiveType::Triangle, 0, 3)
}

/// A frame of one pass running `commands` and then one draw.
fn one_pass(commands: &[Command]) -> PassState {
    let mut s = PassState::new();
    for &cmd in commands {
        s.emit_command(cmd);
    }
    s.emit_command(draw());
    s.end_current_pass("test");
    assert_eq!(s.passes.len(), 1);
    s
}

/// Record `commands` + draw, put `after` in the place of `commands`, and run the check.
///
/// `None` when the check passes, else its panic message.
fn check_after_swap(commands: &[Command], after: &[Command]) -> Option<String> {
    let mut s = one_pass(commands);
    let before = s.debug_record_draw_states();
    // Keep what the pass opened with, the viewport among it.
    let pass_commands = &mut s.passes[0].commands;
    pass_commands.truncate(pass_commands.len() - commands.len() - 1);
    pass_commands.extend_from_slice(after);
    pass_commands.push(draw());
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        s.debug_assert_draw_states_preserved(&before, &FxHashMap::default());
    }));
    outcome.err().map(|payload| {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|m| (*m).to_owned()))
            .unwrap_or_default()
    })
}

fn assert_caught(class: &str, commands: &[Command], after: &[Command]) {
    let expected = format!("pass rules changed the {class} a surviving draw sees");
    match check_after_swap(commands, after) {
        Some(message) => assert!(
            message.contains(&expected),
            "{class}: reported {message:?}, expected {expected:?}"
        ),
        None => panic!(
            "{class}: the check missed a change from {:?} to {:?}",
            kinds(commands),
            kinds(after)
        ),
    }
}

/// Each command's type and slot, for a failure message.
fn kinds(commands: &[Command]) -> Vec<(u32, u32)> {
    commands.iter().map(|cmd| (cmd.cmd, cmd.param_a)).collect()
}

fn index(slot: usize) -> u32 {
    u32::try_from(slot).unwrap()
}

#[test]
fn every_state_class_catches_a_changed_value() {
    let cases = [
        (
            "viewport",
            Command::set_viewport(0, 0, 640, 480, 0.0, 1.0),
            Command::set_viewport(0, 0, 320, 480, 0.0, 1.0),
        ),
        (
            "depth-stencil state",
            Command::set_depth_stencil_state(0x10),
            Command::set_depth_stencil_state(0x11),
        ),
        (
            "cull mode",
            Command::set_cull_mode(CullMode::Back),
            Command::set_cull_mode(CullMode::Front),
        ),
        (
            "triangle fill mode",
            Command::set_triangle_fill_mode(TriangleFillMode::Lines),
            Command::set_triangle_fill_mode(TriangleFillMode::Fill),
        ),
        (
            "scissor rect",
            Command::set_scissor_rect(0, 0, 64, 64),
            Command::set_scissor_rect(0, 0, 64, 32),
        ),
        (
            "stencil reference",
            Command::set_stencil_reference(7),
            Command::set_stencil_reference(3),
        ),
        (
            "depth bias",
            Command::set_depth_bias(1.0, 2.0),
            Command::set_depth_bias(1.0, 3.0),
        ),
        (
            "blend colour",
            Command::set_blend_color(0.5, 0.5, 0.5, 0.5),
            Command::set_blend_color(0.5, 0.5, 0.5, 0.25),
        ),
        (
            "visibility result mode",
            Command::set_visibility_result_mode(VisibilityResultMode::Counting, 16),
            Command::set_visibility_result_mode(VisibilityResultMode::Boolean, 16),
        ),
        (
            "vertex buffers",
            Command::set_vertex_buffer(0x20, 0, 1),
            Command::set_vertex_buffer(0x20, 64, 1),
        ),
        (
            "fragment buffers",
            Command::set_fragment_buffer(0x30, 0, 1),
            Command::set_fragment_buffer(0x31, 0, 1),
        ),
        (
            "fragment textures and samplers",
            Command::set_fragment_texture(0x40, 2),
            Command::set_fragment_texture(0x41, 2),
        ),
        (
            "vertex textures and samplers",
            Command::set_vertex_texture(0x50, 1),
            Command::set_vertex_texture(0x51, 1),
        ),
    ];
    assert_eq!(cases.len(), CLASSES.len(), "one case per class");
    for ((class, was, now), (name, _, _)) in cases.iter().zip(&CLASSES) {
        assert_eq!(class, name, "cases follow CLASSES");
        assert_caught(class, &[*was], &[*now]);
    }
}

#[test]
fn a_state_the_draw_ran_without_is_caught() {
    // Only the binding classes mask a slot set after the rules but unset before.
    assert_caught("cull mode", &[], &[Command::set_cull_mode(CullMode::Back)]);
    assert_caught(
        "viewport",
        &[],
        &[Command::set_viewport(0, 0, 640, 480, 0.0, 1.0)],
    );
}

#[test]
fn every_binding_slot_catches_a_changed_value() {
    type Bind = fn(u64, u32) -> Command;
    let classes: [(&str, Bind, std::ops::Range<usize>); 6] = [
        (
            "vertex buffers",
            |handle, slot| Command::set_vertex_buffer(handle, 0, slot),
            0..BUFFER_SLOTS,
        ),
        (
            // Fragment slot 0 is the quads' own argument, never inherited.
            "fragment buffers",
            |handle, slot| Command::set_fragment_buffer(handle, 0, slot),
            1..BUFFER_SLOTS,
        ),
        (
            "fragment textures and samplers",
            Command::set_fragment_texture,
            0..LAST_BOUND_MAX_STAGES,
        ),
        (
            "fragment textures and samplers",
            Command::set_fragment_sampler_state,
            0..LAST_BOUND_MAX_STAGES,
        ),
        (
            "vertex textures and samplers",
            Command::set_vertex_texture,
            0..VERTEX_SAMPLER_SLOTS,
        ),
        (
            "vertex textures and samplers",
            Command::set_vertex_sampler_state,
            0..VERTEX_SAMPLER_SLOTS,
        ),
    ];
    for (class, bind, slots) in classes {
        let last = slots.end - 1;
        for slot in slots {
            let was = bind(0x100, index(slot));
            assert_caught(class, &[was], &[bind(0x101, index(slot))]);
            // The same bind one slot over leaves the recorded slot unset.
            let neighbour = if slot == last { slot - 1 } else { slot + 1 };
            assert_caught(class, &[was], &[bind(0x100, index(neighbour))]);
        }
    }
}

#[test]
fn two_slots_trading_values_is_caught() {
    // The class fingerprint is a sum, blind to order, so each slot's
    // position has to be part of its own fingerprint.
    let [a, b] = [0x110, 0x111];
    assert_caught(
        "fragment textures and samplers",
        &[
            Command::set_fragment_texture(a, 0),
            Command::set_fragment_texture(b, 1),
        ],
        &[
            Command::set_fragment_texture(b, 0),
            Command::set_fragment_texture(a, 1),
        ],
    );
}

#[test]
fn a_null_texture_bind_is_caught_on_both_its_slots() {
    // It binds the black texture and the default sampler together.
    let null = Command::set_fragment_null_texture(NullTextureKind::Texture2D, 3);
    let texture = Command::set_fragment_texture(0x60, 3);
    let sampler = Command::set_fragment_sampler_state(0x61, 3);
    assert_caught("fragment textures and samplers", &[null], &[null, texture]);
    assert_caught("fragment textures and samplers", &[null], &[null, sampler]);
    let null = Command::set_vertex_null_texture(NullTextureKind::Texture2D, 1);
    assert_caught(
        "vertex textures and samplers",
        &[null],
        &[null, Command::set_vertex_texture(0x62, 1)],
    );
}

#[test]
fn a_slot_set_back_to_its_recorded_value_is_quiet() {
    // Rebinding a slot takes its old fingerprint out of the class sum.
    let a = Command::set_fragment_texture(0x70, 0);
    let b = Command::set_fragment_texture(0x71, 0);
    assert_eq!(check_after_swap(&[a, b], &[b]), None);
    assert_eq!(check_after_swap(&[b], &[a, b, a, b]), None);
    let cull = Command::set_cull_mode(CullMode::Back);
    assert_eq!(
        check_after_swap(&[cull], &[Command::set_cull_mode(CullMode::Front), cull]),
        None
    );
}

#[test]
fn a_fresh_value_matches_the_state_left_unset() {
    let fresh = Command::set_stencil_reference(0);
    assert_eq!(
        check_after_swap(&[Command::set_stencil_reference(5), fresh], &[]),
        None
    );
    assert_eq!(check_after_swap(&[], &[fresh]), None);
}

#[test]
fn a_binding_the_draw_found_unbound_is_masked_but_not_its_neighbours() {
    let texture = Command::set_fragment_texture(0x80, 0);
    let leftover = Command::set_fragment_texture(0x81, 5);
    let buffer = Command::set_vertex_buffer(0x82, 0, 7);
    assert_eq!(
        check_after_swap(&[texture], &[leftover, buffer, texture]),
        None
    );
    assert_caught(
        "fragment textures and samplers",
        &[texture],
        &[leftover, Command::set_fragment_texture(0x83, 0)],
    );
}

#[test]
fn a_rewritten_pipeline_is_caught() {
    let message = check_after_swap(
        &[Command::set_render_pipeline_state(0x90)],
        &[Command::set_render_pipeline_state(0x91)],
    );
    assert!(
        message.is_some_and(|m| m.contains("changed the pipeline a surviving draw sees")),
        "the pipeline swap was missed"
    );
}
