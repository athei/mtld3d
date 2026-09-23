//! Debug-build check that the submit-time pass rules keep every surviving draw's encoder state.
//!
//! `LastBoundCache::debug_assert_in_sync` runs while commands are recorded, so it cannot see a
//! rule that later drops or rewrites them. Rule H drains colour clear-quad blocks out of the
//! passes it strips, and a later command that deduplicated a state change against a value bound
//! inside such a block would then run with whatever preceded the block. Each pass is one Metal
//! render encoder, so the state a draw sees is a function of the commands before it in its pass:
//! replaying every pass once before the rules and once after them and comparing what each
//! surviving draw sees closes that class for every rule at once.

use mtld3d_shared::{
    Command, CommandType, MetalHandle, mtl::VERTEX_STREAM_SLOTS,
    mtl_handle::MTLRenderPipelineStateKind,
};
use rustc_hash::FxHashMap;
use xxhash_rust::xxh3::Xxh3;

use super::{LAST_BOUND_MAX_STAGES, Pass, PassState, VERTEX_SAMPLER_SLOTS};

/// Entries in Metal's per-stage buffer argument table.
const BUFFER_SLOTS: usize = 31;

const VIEWPORT: usize = 0;
const DEPTH_STENCIL: usize = 1;
const CULL_MODE: usize = 2;
const TRIANGLE_FILL_MODE: usize = 3;
const SCISSOR_RECT: usize = 4;
const STENCIL_REFERENCE: usize = 5;
const DEPTH_BIAS: usize = 6;
const BLEND_COLOR: usize = 7;
const VISIBILITY_RESULT_MODE: usize = 8;
const VERTEX_BUFFERS: usize = 9;
const FRAGMENT_BUFFERS: usize = VERTEX_BUFFERS + BUFFER_SLOTS;
const FRAGMENT_TEXTURES: usize = FRAGMENT_BUFFERS + BUFFER_SLOTS;
const FRAGMENT_SAMPLERS: usize = FRAGMENT_TEXTURES + LAST_BOUND_MAX_STAGES;
const VERTEX_TEXTURES: usize = FRAGMENT_SAMPLERS + LAST_BOUND_MAX_STAGES;
const VERTEX_SAMPLERS: usize = VERTEX_TEXTURES + VERTEX_SAMPLER_SLOTS;
const SLOT_COUNT: usize = VERTEX_SAMPLERS + VERTEX_SAMPLER_SLOTS;

/// The state classes a fingerprint hashes separately, as `(name, first slot, end slot)`.
///
/// The name is what a mismatch reports. Every slot belongs to exactly one class.
const CLASSES: [(&str, usize, usize); 13] = [
    ("viewport", VIEWPORT, VIEWPORT + 1),
    ("depth-stencil state", DEPTH_STENCIL, DEPTH_STENCIL + 1),
    ("cull mode", CULL_MODE, CULL_MODE + 1),
    (
        "triangle fill mode",
        TRIANGLE_FILL_MODE,
        TRIANGLE_FILL_MODE + 1,
    ),
    ("scissor rect", SCISSOR_RECT, SCISSOR_RECT + 1),
    (
        "stencil reference",
        STENCIL_REFERENCE,
        STENCIL_REFERENCE + 1,
    ),
    ("depth bias", DEPTH_BIAS, DEPTH_BIAS + 1),
    ("blend colour", BLEND_COLOR, BLEND_COLOR + 1),
    (
        "visibility result mode",
        VISIBILITY_RESULT_MODE,
        VISIBILITY_RESULT_MODE + 1,
    ),
    ("vertex buffers", VERTEX_BUFFERS, FRAGMENT_BUFFERS),
    ("fragment buffers", FRAGMENT_BUFFERS, FRAGMENT_TEXTURES),
    (
        "fragment textures and samplers",
        FRAGMENT_TEXTURES,
        VERTEX_TEXTURES,
    ),
    ("vertex textures and samplers", VERTEX_TEXTURES, SLOT_COUNT),
];

/// Every draw of a frame with the encoder state it saw, recorded ahead of the pass rules.
///
/// Built by [`PassState::debug_record_draw_states`] and checked by
/// [`PassState::debug_assert_draw_states_preserved`] once the rules have run.
pub struct DrawStateLedger {
    draws: Vec<DrawState>,
}

/// What one draw saw: its pipeline, and a hash of every other state class.
struct DrawState {
    /// Where the draw sat, as `(pass index, command index)`, for the panic message.
    location: (usize, usize),
    pipeline: u64,
    classes: [u64; CLASSES.len()],
}

/// The state one Metal render encoder holds while its commands replay.
///
/// Each slot keeps the bits of the command that last set it, zero while unset (command types
/// start at 1, so a set slot is never all zero). The class hashes are refreshed lazily, only
/// for the classes a command touched since the previous draw.
struct EncoderReplay {
    pipeline: u64,
    slots: [[u64; 4]; SLOT_COUNT],
    class_hashes: [u64; CLASSES.len()],
    dirty: u16,
}

impl PassState {
    /// Record the encoder state every draw of the frame sees, ahead of the submit-time rules.
    ///
    /// Debug builds only; pair with [`Self::debug_assert_draw_states_preserved`].
    #[must_use]
    pub fn debug_record_draw_states(&self) -> DrawStateLedger {
        let mut draws = Vec::new();
        for (pass_index, pass) in self.passes.iter().enumerate() {
            replay_pass(pass, |command_index, pipeline, classes| {
                draws.push(DrawState {
                    location: (pass_index, command_index),
                    pipeline,
                    classes: *classes,
                });
            });
        }
        DrawStateLedger { draws }
    }

    /// Assert every draw that survived the submit-time rules sees the state recorded before them.
    ///
    /// Draws inside colour clear-quad blocks are left out on both sides: Rule H drains those
    /// blocks with the attachment they write. A pipeline matches when it is the recorded
    /// handle or the no-colour sibling `alt` maps that handle to, which is the one rewrite
    /// Rule H makes. Debug builds only.
    ///
    /// # Panics
    ///
    /// Panics when a surviving draw sees a different pipeline or a different value in any
    /// other state class, or when the rules changed the number of surviving draws.
    pub fn debug_assert_draw_states_preserved(
        &self,
        before: &DrawStateLedger,
        alt: &FxHashMap<u64, MetalHandle<MTLRenderPipelineStateKind>>,
    ) {
        let mut next = 0usize;
        for (pass_index, pass) in self.passes.iter().enumerate() {
            replay_pass(pass, |command_index, pipeline, classes| {
                let Some(recorded) = before.draws.get(next) else {
                    panic!(
                        "pass rules added a draw at pass {pass_index} command {command_index} \
                         ({} recorded before the rules)",
                        before.draws.len()
                    );
                };
                let (was_pass, was_command) = recorded.location;
                assert!(
                    pipeline == recorded.pipeline
                        || alt.get(&recorded.pipeline).map(|h| h.raw()) == Some(pipeline),
                    "pass rules changed the pipeline a surviving draw sees: now pass \
                     {pass_index} command {command_index} binds {pipeline:#x}, before the rules \
                     pass {was_pass} command {was_command} bound {:#x}",
                    recorded.pipeline
                );
                for ((name, _, _), (now, was)) in
                    CLASSES.iter().zip(classes.iter().zip(&recorded.classes))
                {
                    assert!(
                        now == was,
                        "pass rules changed the {name} a surviving draw sees: \
                         now pass {pass_index} command {command_index}, before the rules pass \
                         {was_pass} command {was_command}"
                    );
                }
                next += 1;
            });
        }
        assert_eq!(
            next,
            before.draws.len(),
            "pass rules dropped draws outside colour clear-quad blocks"
        );
    }
}

impl EncoderReplay {
    /// A fresh encoder: nothing bound, every class due for a hash.
    const fn new() -> Self {
        Self {
            pipeline: 0,
            slots: [[0; 4]; SLOT_COUNT],
            class_hashes: [0; CLASSES.len()],
            dirty: u16::MAX,
        }
    }

    /// Record the state `cmd` sets.
    fn apply(&mut self, cmd: &Command) {
        let Some(kind) = CommandType::from_repr(cmd.cmd) else {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "draw-state replay: unknown command type {} → ignored", cmd.cmd);
            return;
        };
        let index = cmd.param_a as usize;
        let bits = command_bits(cmd);
        match kind {
            CommandType::SetRenderPipelineState => self.pipeline = cmd.param_b,
            CommandType::SetViewport => self.set(VIEWPORT, bits),
            CommandType::SetDepthStencilState => self.set(DEPTH_STENCIL, bits),
            CommandType::SetCullMode => self.set(CULL_MODE, bits),
            CommandType::SetTriangleFillMode => self.set(TRIANGLE_FILL_MODE, bits),
            CommandType::SetScissorRect => self.set(SCISSOR_RECT, bits),
            CommandType::SetStencilReference => self.set(STENCIL_REFERENCE, bits),
            CommandType::SetDepthBias => self.set(DEPTH_BIAS, bits),
            CommandType::SetBlendColor => self.set(BLEND_COLOR, bits),
            CommandType::SetVisibilityResultMode => self.set(VISIBILITY_RESULT_MODE, bits),
            CommandType::SetVertexBuffer => {
                self.set_slot(VERTEX_BUFFERS, BUFFER_SLOTS, index, bits);
            }
            CommandType::SetVertexBytes | CommandType::SetVertexBytesAt => {
                // Inline bytes at a stream slot are a quad's own vertex
                // argument, or zeros for a stream nothing is bound to. The
                // dedup cache forgets the slot after either, so every later
                // draw that reads the stream binds it again and no draw
                // inherits the bytes. The replay leaves the slot showing the
                // last buffer bound there, the value a drained quad leaves
                // behind too.
                if cmd.param_a >= VERTEX_STREAM_SLOTS {
                    self.set_slot(VERTEX_BUFFERS, BUFFER_SLOTS, index, bits);
                }
            }
            CommandType::SetFragmentBytesAt | CommandType::SetFragmentBuffer => {
                // Fragment slot 0 is the clear and blit quads' own argument,
                // bound right before every quad draw that reads it; no draw
                // inherits it.
                if index != 0 {
                    self.set_slot(FRAGMENT_BUFFERS, BUFFER_SLOTS, index, bits);
                }
            }
            CommandType::SetFragmentTexture => {
                self.set_slot(FRAGMENT_TEXTURES, LAST_BOUND_MAX_STAGES, index, bits);
            }
            CommandType::SetFragmentSamplerState => {
                self.set_slot(FRAGMENT_SAMPLERS, LAST_BOUND_MAX_STAGES, index, bits);
            }
            CommandType::SetFragmentNullTexture => {
                // Binds the black texture and the default sampler together.
                self.set_slot(FRAGMENT_TEXTURES, LAST_BOUND_MAX_STAGES, index, bits);
                self.set_slot(FRAGMENT_SAMPLERS, LAST_BOUND_MAX_STAGES, index, bits);
            }
            CommandType::SetVertexTexture => {
                self.set_slot(VERTEX_TEXTURES, VERTEX_SAMPLER_SLOTS, index, bits);
            }
            CommandType::SetVertexSamplerState => {
                self.set_slot(VERTEX_SAMPLERS, VERTEX_SAMPLER_SLOTS, index, bits);
            }
            CommandType::SetVertexNullTexture => {
                self.set_slot(VERTEX_TEXTURES, VERTEX_SAMPLER_SLOTS, index, bits);
                self.set_slot(VERTEX_SAMPLERS, VERTEX_SAMPLER_SLOTS, index, bits);
            }
            // Draws read the state rather than set it, and a debug group
            // carries none.
            CommandType::DrawPrimitives
            | CommandType::DrawIndexedPrimitives
            | CommandType::DrawIndexedPrimitivesUp
            | CommandType::PushDebugGroup
            | CommandType::PopDebugGroup => {}
        }
    }

    fn set_slot(&mut self, base: usize, count: usize, index: usize, bits: [u64; 4]) {
        if index >= count {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "draw-state replay: bind index {index} past the {count} slots it models → ignored");
            return;
        }
        self.set(base + index, bits);
    }

    fn set(&mut self, slot: usize, bits: [u64; 4]) {
        self.slots[slot] = bits;
        let class = CLASSES
            .iter()
            .position(|&(_, start, end)| (start..end).contains(&slot))
            .expect("every slot belongs to a class");
        self.dirty |= 1 << class;
    }

    /// The per-class hashes of the current state, refreshing the classes touched since last time.
    fn class_hashes(&mut self) -> &[u64; CLASSES.len()] {
        for (class, &(_, start, end)) in CLASSES.iter().enumerate() {
            if self.dirty & (1 << class) != 0 {
                let mut hasher = Xxh3::new();
                for word in self.slots[start..end].iter().flatten() {
                    hasher.update(&word.to_le_bytes());
                }
                self.class_hashes[class] = hasher.digest();
            }
        }
        self.dirty = 0;
        &self.class_hashes
    }
}

/// Replay `pass` on a fresh encoder, calling `on_draw` for each draw outside its clear-quad blocks.
///
/// `on_draw` receives the draw's command index, its pipeline and the per-class state hashes.
fn replay_pass(pass: &Pass, mut on_draw: impl FnMut(usize, u64, &[u64; CLASSES.len()])) {
    let mut encoder = EncoderReplay::new();
    for (index, cmd) in pass.commands.iter().enumerate() {
        if !cmd.is_draw() {
            encoder.apply(cmd);
            continue;
        }
        let in_clear_quad = pass
            .color_clear_quad_ranges
            .iter()
            .any(|&(start, end)| (start..end).contains(&index));
        if !in_clear_quad {
            let pipeline = encoder.pipeline;
            on_draw(index, pipeline, encoder.class_hashes());
        }
    }
}

/// The whole command as four words, so two binds compare equal only when every field does.
fn command_bits(cmd: &Command) -> [u64; 4] {
    [
        (u64::from(cmd.cmd) << 32) | u64::from(cmd.param_a),
        cmd.param_b,
        cmd.param_c,
        cmd.param_d,
    ]
}
