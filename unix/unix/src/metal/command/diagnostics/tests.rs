use log::{Level, log_enabled};
use objc2::{
    define_class, extern_methods,
    rc::Retained,
    runtime::{AnyObject, ProtocolObject},
};
use objc2_foundation::{
    NSArray, NSDictionary, NSError, NSLocalizedDescriptionKey, NSObject, NSObjectProtocol, NSString,
};
use objc2_metal::{
    MTLBuffer, MTLCommandBufferEncoderInfo, MTLCommandBufferEncoderInfoErrorKey,
    MTLCommandBufferErrorOption, MTLCommandBufferStatus, MTLCommandEncoderErrorState,
    MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat, MTLResource,
    MTLResourceOptions, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};

use super::{
    BlitSite, DepthTransfer, DepthTransferResample, append_signposts, buffer_role,
    copy_buffer_details, depth_transfer, depth_transfer_resample, diagnostic_descriptor,
    dispatch_region, encoder_state_name, error_details, metadata_string, optional_string,
    resample_details, sequence, status_name, texture_details,
};

define_class!(
    // SAFETY: NSObject has no subclassing requirements. This fixture has no ivars or Drop.
    #[unsafe(super = NSObject)]
    struct CommandEncoderInfoFixture;

    // SAFETY: NSObject supplies the NSObjectProtocol methods.
    unsafe impl NSObjectProtocol for CommandEncoderInfoFixture {}

    // SAFETY: these methods implement the typed protocol signatures with owned objects.
    unsafe impl MTLCommandBufferEncoderInfo for CommandEncoderInfoFixture {
        #[unsafe(method_id(label))]
        fn label(&self) -> Retained<NSString> {
            NSString::from_str("pass\"\n\\label")
        }

        #[unsafe(method_id(debugSignposts))]
        fn debug_signposts(&self) -> Retained<NSArray<NSString>> {
            NSArray::from_retained_slice(&[
                NSString::from_str("first\nmarker"),
                NSString::from_str("second\t\"marker\\"),
            ])
        }

        #[unsafe(method(errorState))]
        fn error_state(&self) -> MTLCommandEncoderErrorState {
            MTLCommandEncoderErrorState::Affected
        }
    }
);

impl CommandEncoderInfoFixture {
    extern_methods!(
        // SAFETY: NSObject's inherited new initializes this subclass, which has no ivars.
        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        fn new() -> Retained<Self>;
    );
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements. This fixture has no ivars or Drop.
    #[unsafe(super = NSObject)]
    struct NilSignpostsEncoderInfoFixture;

    // SAFETY: NSObject supplies the NSObjectProtocol methods.
    unsafe impl NSObjectProtocol for NilSignpostsEncoderInfoFixture {}

    // SAFETY: the methods use the protocol's Objective-C ABI. The object return is
    // deliberately nullable to model missing metadata; no invalid Retained is created.
    unsafe impl MTLCommandBufferEncoderInfo for NilSignpostsEncoderInfoFixture {
        #[unsafe(method_id(label))]
        fn label(&self) -> Retained<NSString> {
            NSString::from_str("init-clear")
        }

        #[unsafe(method_id(debugSignposts))]
        fn debug_signposts(&self) -> Option<Retained<NSArray<NSString>>> {
            None
        }

        #[unsafe(method(errorState))]
        fn error_state(&self) -> MTLCommandEncoderErrorState {
            MTLCommandEncoderErrorState::Faulted
        }
    }
);

impl NilSignpostsEncoderInfoFixture {
    extern_methods!(
        // SAFETY: NSObject's inherited new initializes this subclass, which has no ivars.
        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        fn new() -> Retained<Self>;
    );
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements. This fixture has no ivars or Drop.
    #[unsafe(super = NSObject)]
    struct MissingLabelEncoderInfoFixture;

    // SAFETY: NSObject supplies the NSObjectProtocol methods.
    unsafe impl NSObjectProtocol for MissingLabelEncoderInfoFixture {}

    // SAFETY: the methods use the protocol's Objective-C ABI. Object metadata is
    // deliberately incomplete and consumed only through checked, nullable KVC reads.
    unsafe impl MTLCommandBufferEncoderInfo for MissingLabelEncoderInfoFixture {
        #[unsafe(method_id(label))]
        fn label(&self) -> Option<Retained<NSString>> {
            None
        }

        #[unsafe(method_id(debugSignposts))]
        fn debug_signposts(&self) -> Retained<NSArray<AnyObject>> {
            NSArray::from_slice(&[
                NSString::from_str("before\nmarker").as_ref(),
                NSObject::new().as_ref(),
                NSString::from_str("after\"marker").as_ref(),
            ])
        }

        #[unsafe(method(errorState))]
        fn error_state(&self) -> MTLCommandEncoderErrorState {
            MTLCommandEncoderErrorState::Pending
        }
    }
);

impl MissingLabelEncoderInfoFixture {
    extern_methods!(
        // SAFETY: NSObject's inherited new initializes this subclass, which has no ivars.
        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        fn new() -> Retained<Self>;
    );
}

#[test]
fn nil_signposts_preserve_primary_error_and_encoder_state() {
    let encoder = NilSignpostsEncoderInfoFixture::new();
    let payload = NSArray::<AnyObject>::from_slice(&[encoder.as_ref()]);
    let details = error_details(Some(&fixture_error(Some(payload.as_ref()))));
    assert_eq!(
        details,
        "error=present domain=\"fixture\\n\\\"domain\" code=-9 \
         description=\"driver\\n\\\"detail\\\\\" encoder_info=present encoder_count=1 \
         encoder[0]={label=\"init-clear\" state=4(Faulted) signposts=missing}",
    );
}

#[test]
fn nil_label_and_malformed_signpost_preserve_available_metadata() {
    let encoder = MissingLabelEncoderInfoFixture::new();
    let payload = NSArray::<AnyObject>::from_slice(&[encoder.as_ref()]);
    let details = error_details(Some(&fixture_error(Some(payload.as_ref()))));
    assert_eq!(
        details,
        "error=present domain=\"fixture\\n\\\"domain\" code=-9 \
         description=\"driver\\n\\\"detail\\\\\" encoder_info=present encoder_count=1 \
         encoder[0]={label=missing state=3(Pending) \
         signposts=[\"before\\nmarker\", malformed-non-string, \"after\\\"marker\"]}",
    );
}

#[test]
fn metadata_class_checks_distinguish_missing_and_malformed_values() {
    assert_eq!(metadata_string(None), "missing");
    assert_eq!(
        metadata_string(Some(NSObject::new().as_ref())),
        "malformed-non-string",
    );
    let mut missing = String::new();
    append_signposts(&mut missing, None);
    assert_eq!(missing, "missing");
    let mut malformed = String::new();
    append_signposts(
        &mut malformed,
        Some(NSString::from_str("not an array").as_ref()),
    );
    assert_eq!(malformed, "malformed-non-array");
}

#[test]
fn descriptor_opt_in_retains_resources_and_requests_encoder_status() {
    assert!(diagnostic_descriptor(false).is_none());
    let descriptor = diagnostic_descriptor(true).expect("diagnostics enabled");
    assert!(descriptor.retainedReferences());
    assert_eq!(
        descriptor.errorOptions(),
        MTLCommandBufferErrorOption::EncoderExecutionStatus,
    );
}

#[test]
fn missing_error_and_missing_encoder_key_are_distinct() {
    assert_eq!(
        error_details(None),
        "error=missing encoder_info=unavailable"
    );
    let error = fixture_error(None);
    assert_eq!(
        error_details(Some(&error)),
        "error=present domain=\"fixture\\n\\\"domain\" code=-9 \
         description=\"driver\\n\\\"detail\\\\\" encoder_info=missing-key",
    );
    assert_eq!(super::super::command_buffer_error(Some(&error)).0, 9);
}

#[test]
fn encoder_payload_checks_array_class_and_protocol_per_element() {
    let wrong_type = NSString::from_str("not an array");
    assert!(
        error_details(Some(&fixture_error(Some(wrong_type.as_ref()))))
            .ends_with("encoder_info=malformed-non-array")
    );
    let empty = NSArray::<AnyObject>::new();
    assert!(
        error_details(Some(&fixture_error(Some(empty.as_ref())))).ends_with("encoder_info=empty")
    );

    let encoder = CommandEncoderInfoFixture::new();
    let nonconforming = NSObject::new();
    let payload = NSArray::<AnyObject>::from_slice(&[
        nonconforming.as_ref(),
        encoder.as_ref(),
        nonconforming.as_ref(),
    ]);
    let details = error_details(Some(&fixture_error(Some(payload.as_ref()))));
    assert!(
        details.ends_with(
            "encoder_info=present encoder_count=3 encoder[0]={malformed-nonconforming} \
         encoder[1]={label=\"pass\\\"\\n\\\\label\" state=2(Affected) \
         signposts=[\"first\\nmarker\", \"second\\t\\\"marker\\\\\"]} \
         encoder[2]={malformed-nonconforming}"
        ),
        "{details}"
    );
    assert!(!details.contains(['\n', '\r', '\t']));
}

#[test]
fn empty_signposts_do_not_claim_success() {
    let mut output = String::new();
    append_signposts(&mut output, Some(NSArray::<AnyObject>::new().as_ref()));
    assert_eq!(output, "empty");
}

#[test]
fn all_encoder_states_and_future_values_keep_their_names() {
    for (state, name) in [
        (MTLCommandEncoderErrorState::Unknown, "Unknown"),
        (MTLCommandEncoderErrorState::Completed, "Completed"),
        (MTLCommandEncoderErrorState::Affected, "Affected"),
        (MTLCommandEncoderErrorState::Pending, "Pending"),
        (MTLCommandEncoderErrorState::Faulted, "Faulted"),
        (MTLCommandEncoderErrorState(-9), "unrecognized"),
    ] {
        assert_eq!(encoder_state_name(state), name);
    }
}

#[test]
fn completion_states_include_nonterminal_and_future_values() {
    for (status, name) in [
        (MTLCommandBufferStatus::NotEnqueued, "NotEnqueued"),
        (MTLCommandBufferStatus::Enqueued, "Enqueued"),
        (MTLCommandBufferStatus::Committed, "Committed"),
        (MTLCommandBufferStatus::Scheduled, "Scheduled"),
        (MTLCommandBufferStatus::Completed, "Completed"),
        (MTLCommandBufferStatus::Error, "Error"),
        (MTLCommandBufferStatus(99), "unrecognized"),
    ] {
        assert_eq!(status_name(status), name);
    }
}

#[test]
fn optional_identity_fields_preserve_missing_and_escape_present_strings() {
    assert_eq!(sequence(None), "unavailable");
    assert_eq!(sequence(Some(17)), "0x11");
    assert_eq!(optional_string(None), "missing");
    assert_eq!(optional_string(Some("")), "\"\"");
    assert_eq!(optional_string(Some("a\n\"b\\")), "\"a\\n\\\"b\\\\\"");
    assert_eq!(buffer_role(Some("mtld3d-frame-0x1")), "frame");
    assert_eq!(buffer_role(Some("mtld3d-upload-0x1")), "upload");
    assert_eq!(buffer_role(Some("mtld3d-readback")), "readback");
    assert_eq!(buffer_role(Some("mtld3d-init-clear")), "initialization");
    assert_eq!(buffer_role(Some("mtld3d-init-clear-extra")), "unknown");
    assert_eq!(buffer_role(Some("unexpected")), "unknown");
    assert_eq!(buffer_role(None), "unknown");
}

#[test]
fn absent_copy_ends_and_dispatch_sizes_keep_the_copy_vocabulary() {
    assert_eq!(texture_details(None, 0), "missing");
    assert_eq!(copy_buffer_details(None), "missing");
    assert_eq!(BlitSite::DepthTransfer.to_string(), "depth-transfer");
    assert_eq!(
        dispatch_region(MTLSize {
            width: 3,
            height: 5,
            depth: 1,
        })
        .to_string(),
        "3x5x1",
    );
}

/// The multisample path's record names both kernel ends and the sample it reads.
///
/// The transfer's own ends ride the `texture-copy` record, so what this pins is
/// the private copy the kernel samples, its stencil view, the output planes and
/// the kernel's own argument.
#[test]
fn depth_transfer_resample_names_both_kernel_ends_and_the_sample_it_reads() {
    let Some(fixture) = ResampleFixture::new() else {
        return;
    };
    let details = resample_details(&fixture.pass());
    assert_eq!(
        details,
        format!(
            "site=depth-transfer/1 sample=0 \
             src={{texture={:p} label=\"source\\\"plane\" storage=MTLStorageMode(2) usage=0x11 \
             type=MTLTextureType(4) array_length=1 \
             MTLPixelFormat(260) samples=4 level=0/1 origin=0,0 size=8x4x1}} \
             src_stencil={{texture={:p} label=missing storage=MTLStorageMode(2) usage=0x11 \
             type=MTLTextureType(4) array_length=1 \
             MTLPixelFormat(261) samples=4 level=0/1 origin=0,0 size=8x4x1}} \
             src_planes={{depth={{missing}} stencil={{missing}}}} \
             dst_planes={{depth={{buffer={:p} label=\"depth\\nplane\" \
             storage=MTLStorageMode(2)}} \
             stencil={{buffer={:p} label=missing storage=MTLStorageMode(2)}}}} \
             src_region=8x4x1 region=4x2x1 \
             src_strides={{depth=0 stencil=0}} dst_strides={{depth=64 stencil=256}} \
             grid=1x1x1 threadgroup=8x8x1",
            Retained::as_ptr(&fixture.source),
            Retained::as_ptr(&fixture.stencil_view),
            Retained::as_ptr(&fixture.output_depth),
            Retained::as_ptr(&fixture.output_stencil),
        ),
    );
    assert!(!details.contains(['\n', '\r', '\t']));
}

/// With the target off both entry points return before querying anything.
///
/// Nothing observes a record that is never written, so what this pins is that
/// the disabled calls are reached at all and that the detail helpers stay
/// behind them.
#[test]
fn depth_transfer_records_are_inert_without_the_target() {
    assert!(!log_enabled!(target: super::LOG_TARGET, Level::Debug));
    let Some(fixture) = ResampleFixture::new() else {
        return;
    };
    let Some(queue) = fixture.device.newCommandQueue() else {
        return;
    };
    let Some(cb) = queue.commandBuffer() else {
        return;
    };
    depth_transfer(
        &cb,
        &DepthTransfer {
            source: &fixture.source,
            source_level: 0,
            destination: &fixture.source,
            destination_level: 0,
            width: 8,
            height: 4,
        },
    );
    depth_transfer_resample(&cb, &fixture.pass());
}

/// The live Metal objects one depth-transfer compute pass would bind.
struct ResampleFixture {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    source: Retained<ProtocolObject<dyn MTLTexture>>,
    stencil_view: Retained<ProtocolObject<dyn MTLTexture>>,
    output_depth: Retained<ProtocolObject<dyn MTLBuffer>>,
    output_stencil: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl ResampleFixture {
    fn new() -> Option<Self> {
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
            return None;
        };
        // SAFETY: objc2 typed binding; a class method building a descriptor.
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::Depth32Float_Stencil8,
                8,
                4,
                false,
            )
        };
        desc.setStorageMode(MTLStorageMode::Private);
        desc.setTextureType(MTLTextureType::Type2DMultisample);
        // SAFETY: every Metal device this layer runs on answers for four samples.
        unsafe {
            desc.setSampleCount(4);
        }
        desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView);
        let source = device.newTextureWithDescriptor(&desc)?;
        source.setLabel(Some(&NSString::from_str("source\"plane")));
        let stencil_view = source.newTextureViewWithPixelFormat(MTLPixelFormat::X32_Stencil8)?;
        let plane = |length| {
            device.newBufferWithLength_options(length, MTLResourceOptions::StorageModePrivate)
        };
        let output_depth = plane(128)?;
        output_depth.setLabel(Some(&NSString::from_str("depth\nplane")));
        let output_stencil = plane(512)?;
        Some(Self {
            device,
            source,
            stencil_view,
            output_depth,
            output_stencil,
        })
    }

    fn pass(&self) -> DepthTransferResample<'_> {
        DepthTransferResample {
            source: Some(&self.source),
            source_stencil: Some(&self.stencil_view),
            input_depth: None,
            input_stencil: None,
            output_depth: &self.output_depth,
            output_stencil: Some(&self.output_stencil),
            sizes: [8, 4, 4, 2, 0, 0, 64, 256],
            grid: MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            threadgroup: MTLSize {
                width: 8,
                height: 8,
                depth: 1,
            },
        }
    }
}

fn fixture_error(payload: Option<&AnyObject>) -> Retained<NSError> {
    let description = NSString::from_str("driver\n\"detail\\");
    // SAFETY: Foundation exports this immutable NSString key on every supported macOS.
    let description_key = unsafe { NSLocalizedDescriptionKey };
    let mut keys = vec![description_key];
    let mut values = vec![AsRef::<AnyObject>::as_ref(&*description)];
    if let Some(payload) = payload {
        // SAFETY: Metal exports this immutable Foundation string on every supported macOS.
        keys.push(unsafe { MTLCommandBufferEncoderInfoErrorKey });
        values.push(payload);
    }
    let info = NSDictionary::from_slices(&keys, &values);
    // SAFETY: the user-info description is NSString; encoder-info payloads are deliberately
    // erased fixtures that the diagnostic decoder checks before using their typed APIs.
    unsafe {
        NSError::errorWithDomain_code_userInfo(
            &NSString::from_str("fixture\n\"domain"),
            -9,
            Some(&info),
        )
    }
}
