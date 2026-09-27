use mtld3d_shared::encoder_wire::{FrameSlab, WireError, WireReader};

use super::{
    AdapterSpoof, ColorSpacePolicy, CursorScale, DeviceCapsFlags, GpuCaps, Mtld3dConfig,
    SoftwareCursorPolicy, WireValue,
};

fn round_trip<T: WireValue>(value: &T) -> T {
    let mut slab = FrameSlab::new();
    slab.push_record(1, |writer| value.write_wire(writer))
        .unwrap();
    let mut outer = WireReader::new(slab.as_bytes());
    let mut record = outer.next_record().unwrap().unwrap();
    let result = T::read_wire(&mut record.payload).unwrap();
    assert!(record.payload.is_empty());
    result
}

#[test]
fn resolved_configuration_preserves_nondefaults() {
    let mut value = Mtld3dConfig::default();
    value.caps_all = !value.caps_all;
    value.main_thread_checker = !value.main_thread_checker;
    value.expand_packed16 = !value.expand_packed16;
    value.deny_float32_filtering = !value.deny_float32_filtering;
    value.managed_memory = !value.managed_memory;
    value.linear_align256 = !value.linear_align256;
    value.hdr_enable = !value.hdr_enable;
    value.color_space = ColorSpacePolicy::Accurate;
    value.cursor_scale = CursorScale::Fixed(7);
    value.cursor_software = SoftwareCursorPolicy::Off;
    value.shader_cache_enable = !value.shader_cache_enable;
    value.shader_async_compile = !value.shader_async_compile;
    value.log_dir = "log_dir/utf8-ä".into();
    value.bytecode_dump_dir = "bytecode_dump_dir/utf8-ä".into();
    value.skip_shaders = vec![0, u64::MAX, 55];
    value.present_gate_file = "present_gate_file/utf8-ä".into();
    value.query_flush_immediate = !value.query_flush_immediate;
    value.query_event_immediate = !value.query_event_immediate;
    value.depth_alias_same_size = !value.depth_alias_same_size;
    value.buffer_ignore_lock_bounds = !value.buffer_ignore_lock_bounds;
    value.vbib_retention_cap_bytes = u64::MAX - 123;
    value.vram_budget_cap_bytes = u64::MAX - 123;
    value.pagebox_pool_cap_bytes = u64::MAX - 123;
    value.present_max_fps = 123;
    value.render_scale_percent = 123;
    value.render_lod_bias = !value.render_lod_bias;
    value.adapter_spoof = AdapterSpoof::Amd;
    value.df_formats = !value.df_formats;
    assert_eq!(round_trip(&value), value);
}

#[test]
fn gpu_caps_preserve_native_overrides() {
    let value = GpuCaps {
        unified_memory: false,
        min_linear_texture_align: 256,
        device_caps: DeviceCapsFlags::RESOLVE_NEEDS_RETIRE,
    };
    let decoded = round_trip(&value);
    assert_eq!(decoded.unified_memory, value.unified_memory);
    assert_eq!(
        decoded.min_linear_texture_align,
        value.min_linear_texture_align
    );
    assert_eq!(decoded.device_caps, value.device_caps);
}

#[test]
fn invalid_enum_tags_are_rejected() {
    let bytes = [255];
    assert_eq!(
        AdapterSpoof::read_wire(&mut WireReader::new(&bytes)),
        Err(WireError::InvalidValue)
    );
    assert_eq!(
        ColorSpacePolicy::read_wire(&mut WireReader::new(&bytes)),
        Err(WireError::InvalidValue)
    );
    assert_eq!(
        SoftwareCursorPolicy::read_wire(&mut WireReader::new(&bytes)),
        Err(WireError::InvalidValue)
    );
    assert_eq!(
        CursorScale::read_wire(&mut WireReader::new(&bytes)),
        Err(WireError::InvalidValue)
    );
}
