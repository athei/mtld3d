use std::mem::MaybeUninit;

use mtld3d_shared::shader_create::{
    CreateShaderProgramParams, ShaderInputSemantic, ShaderStage, ShaderUsage,
};
use mtld3d_types::{D3D_OK, D3DERR_INVALIDCALL};

use super::ProgramRegistry;

fn params(tokens: &[u32]) -> CreateShaderProgramParams {
    CreateShaderProgramParams {
        runtime: 0,
        tokens_ptr: tokens.as_ptr() as u64,
        semantics_ptr: 0,
        stage: ShaderStage::Vertex,
        token_count: u32::try_from(tokens.len()).expect("small fixture"),
        semantic_capacity: 0,
        result: D3DERR_INVALIDCALL,
        program_id: 0,
        registration: 0,
        max_const_used: 0,
        semantic_count: 0,
        usage: ShaderUsage::empty(),
        color_out_mask: 0,
        padding: [0; 6],
    }
}

#[test]
fn identical_programs_have_distinct_single_use_registrations() {
    let registry = ProgramRegistry::new();
    let tokens = [0xfffe_0300, 0x0000_ffff];
    let mut first = params(&tokens);
    let mut second = params(&tokens);
    // SAFETY: both requests borrow the live token array and have no semantic output.
    assert_eq!(unsafe { registry.parse(&mut first) }, D3D_OK);
    // SAFETY: the token array remains live and immutable throughout the call.
    assert_eq!(unsafe { registry.parse(&mut second) }, D3D_OK);
    assert_eq!(first.program_id, second.program_id);
    assert_ne!(first.registration, second.registration);
    let (id, program) = registry
        .take(first.registration)
        .expect("registered shader");
    assert_eq!(id.raw(), first.program_id);
    assert_eq!(program.bytecode().as_ref(), tokens);
    assert!(registry.take(first.registration).is_none());
    registry.cancel(second.registration);
    assert!(
        registry
            .pending
            .lock()
            .expect("registry lock")
            .programs
            .is_empty()
    );
}

#[test]
fn failures_publish_nothing_and_clear_stale_outputs() {
    let registry = ProgramRegistry::new();
    let tokens = [0xffff_0300, 0x0000_ffff];
    let mut request = params(&tokens);
    request.registration = 99;
    request.program_id = 99;
    // SAFETY: the token array remains live; a pixel program is invalid for this vertex request.
    assert_eq!(unsafe { registry.parse(&mut request) }, D3DERR_INVALIDCALL);
    assert_eq!(request.registration, 0);
    assert_eq!(request.program_id, 0);
    request.tokens_ptr = 0;
    // SAFETY: a null pointer is rejected before accessing the buffer.
    assert_eq!(unsafe { registry.parse(&mut request) }, D3DERR_INVALIDCALL);
    assert!(
        registry
            .pending
            .lock()
            .expect("registry lock")
            .programs
            .is_empty()
    );
}

#[test]
fn semantic_capacity_is_checked_before_registration() {
    let registry = ProgramRegistry::new();
    let tokens = [0xfffe_0300, 0x0200_001f, 5, 0x900f_0000, 0x0000_ffff];
    let mut request = params(&tokens);
    // SAFETY: tokens are readable; zero output capacity requires no writable buffer.
    assert_eq!(unsafe { registry.parse(&mut request) }, D3DERR_INVALIDCALL);
    assert!(
        registry
            .pending
            .lock()
            .expect("registry lock")
            .programs
            .is_empty()
    );
    let mut output = MaybeUninit::<ShaderInputSemantic>::uninit();
    request.semantics_ptr = output.as_mut_ptr() as u64;
    request.semantic_capacity = 1;
    // SAFETY: the output slot is aligned, exclusive and writable, and tokens remain readable.
    assert_eq!(unsafe { registry.parse(&mut request) }, D3D_OK);
    assert_eq!(request.semantic_count, 1);
    // SAFETY: successful parse wrote the only output slot.
    let semantic = unsafe { output.assume_init() };
    assert_eq!(semantic.register_index, 0);
    registry.cancel(request.registration);
}

#[test]
fn repeated_declarations_require_one_output_slot_each() {
    let registry = ProgramRegistry::new();
    let tokens = [
        0xfffe_0300,
        0x0200_001f,
        5,
        0x900f_0000,
        0x0200_001f,
        5,
        0x900f_0000,
        0x0000_ffff,
    ];
    let mut output = [const { MaybeUninit::<ShaderInputSemantic>::uninit() }; 2];
    let mut request = params(&tokens);
    request.semantics_ptr = output.as_mut_ptr() as u64;
    request.semantic_capacity = 1;
    // SAFETY: both arrays remain live, disjoint and exclusively borrowed for the call.
    assert_eq!(unsafe { registry.parse(&mut request) }, D3DERR_INVALIDCALL);
    assert!(
        registry
            .pending
            .lock()
            .expect("registry lock")
            .programs
            .is_empty()
    );
    request.semantic_capacity = 2;
    // SAFETY: both output slots are writable and the tokens remain readable.
    assert_eq!(unsafe { registry.parse(&mut request) }, D3D_OK);
    assert_eq!(request.semantic_count, 2);
    registry.cancel(request.registration);
}

#[test]
fn invalid_address_ranges_are_rejected_without_dereference() {
    assert!(!super::valid_range::<u32>(0, 1));
    assert!(!super::valid_range::<u32>(1, 1));
    assert!(!super::valid_range::<u32>(u64::MAX - 3, 1));
    assert!(super::valid_range::<u32>(4, 1));
}

#[test]
fn null_runtime_handler_clears_registration() {
    let tokens = [0xfffe_0300, 0x0000_ffff];
    let mut request = params(&tokens);
    request.registration = 99;
    request.result = D3D_OK;
    assert_eq!(
        super::create_handler(std::ptr::from_mut(&mut request).cast()),
        D3DERR_INVALIDCALL,
    );
    assert_eq!(request.result, D3DERR_INVALIDCALL);
    assert_eq!(request.registration, 0);
}

#[test]
fn null_cancellation_is_rejected() {
    let mut params = mtld3d_shared::shader_create::CancelShaderProgramParams {
        runtime: 0,
        registration: 1,
    };
    assert_eq!(
        super::cancel_handler(std::ptr::from_mut(&mut params).cast()),
        D3DERR_INVALIDCALL,
    );
}
