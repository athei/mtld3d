use objc2::{define_class, extern_methods, rc::Retained, runtime::AnyObject};
use objc2_foundation::{
    NSArray, NSDictionary, NSError, NSLocalizedDescriptionKey, NSObject, NSObjectProtocol, NSString,
};
use objc2_metal::{
    MTLCommandBufferEncoderInfo, MTLCommandBufferEncoderInfoErrorKey, MTLCommandBufferErrorOption,
    MTLCommandBufferStatus, MTLCommandEncoderErrorState,
};

use super::{
    append_signposts, buffer_role, diagnostic_descriptor, encoder_state_name, error_details,
    optional_string, sequence, status_name,
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
    append_signposts(&mut output, &NSArray::new());
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
    assert_eq!(buffer_role(Some("unexpected")), "unknown");
    assert_eq!(buffer_role(None), "unknown");
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
