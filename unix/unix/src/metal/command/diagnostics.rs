use std::fmt::Write;

use block2::RcBlock;
use log::{Level, debug, log_enabled};
use objc2::{
    ProtocolType,
    rc::{Retained, autoreleasepool},
    runtime::{AnyObject, ProtocolObject},
};
use objc2_foundation::{NSArray, NSError, NSObject, NSObjectNSKeyValueCoding, NSString, ns_string};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferDescriptor, MTLCommandBufferEncoderInfo,
    MTLCommandBufferEncoderInfoErrorKey, MTLCommandBufferErrorOption, MTLCommandBufferStatus,
    MTLCommandEncoderErrorState, MTLCommandQueue, MTLDevice, MTLRenderPassAttachmentDescriptor,
    MTLRenderPassDescriptor, MTLResource, MTLTexture,
};

use super::{BlitSite, CopyBufferEndpoint, CopyEndpoint, CopyRegion};

const LOG_TARGET: &str = "mtld3d::unix::command";

pub(super) struct TextureCopy<'a> {
    pub(super) texture: &'a ProtocolObject<dyn MTLTexture>,
    pub(super) endpoint: &'a CopyEndpoint,
    pub(super) slice: usize,
}

pub(super) fn render_pass(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    descriptor: &MTLRenderPassDescriptor,
    pass_index: usize,
    command_count: u32,
) {
    if !log_enabled!(target: LOG_TARGET, Level::Debug) {
        return;
    }
    let mut attachments = String::new();
    let colors = descriptor.colorAttachments();
    for index in 0..4 {
        // SAFETY: the four supported color slots are within Metal's attachment array.
        let color = unsafe { colors.objectAtIndexedSubscript(index) };
        write!(
            attachments,
            " color[{index}]={{{} clear={:?}}}",
            attachment_details(&color),
            color.clearColor(),
        )
        .expect("writing to a String cannot fail");
    }
    let depth = descriptor.depthAttachment();
    let stencil = descriptor.stencilAttachment();
    debug!(
        target: LOG_TARGET,
        "render-pass {} site=pass{pass_index} commands={command_count}{} \
         depth={{{} clear={} resolve_filter={:?}}} \
         stencil={{{} clear={} resolve_filter={:?}}}",
        buffer_identity(cb),
        attachments,
        attachment_details(&depth),
        depth.clearDepth(),
        depth.depthResolveFilter(),
        attachment_details(&stencil),
        stencil.clearStencil(),
        stencil.stencilResolveFilter(),
    );
}

pub(super) fn texture_copy(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    site: BlitSite,
    index: usize,
    source: &TextureCopy<'_>,
    destination: &TextureCopy<'_>,
    region: &CopyRegion,
) {
    if !log_enabled!(target: LOG_TARGET, Level::Debug) {
        return;
    }
    debug!(
        target: LOG_TARGET,
        "texture-copy {} site={site}/{index} src={{{} {} slice={} z=0}} \
         dst={{{} {} slice={} z=0}} region={region}",
        buffer_identity(cb),
        texture_identity(source.texture),
        source.endpoint,
        source.slice,
        texture_identity(destination.texture),
        destination.endpoint,
        destination.slice,
    );
}

pub(super) fn readback(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    source: &TextureCopy<'_>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    destination: &CopyBufferEndpoint,
    region: &CopyRegion,
    pe_destination: (u64, u64),
) {
    if !log_enabled!(target: LOG_TARGET, Level::Debug) {
        return;
    }
    debug!(
        target: LOG_TARGET,
        "readback-copy {} site=readback-blit src={{{} {} slice={} z=0}} \
         dst={{buffer={buffer:p} label={} storage={:?} {destination}}} \
         pe_destination={:#x} pe_length={} region={region}",
        buffer_identity(cb),
        texture_identity(source.texture),
        source.endpoint,
        source.slice,
        optional_string(buffer.label().map(|s| s.to_string()).as_deref()),
        buffer.storageMode(),
        pe_destination.0,
        pe_destination.1,
    );
}

fn buffer_identity(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> String {
    format!(
        "buffer={cb:p} label={} queue={:p}",
        optional_string(cb.label().map(|s| s.to_string()).as_deref()),
        Retained::as_ptr(&cb.commandQueue()),
    )
}

fn texture_identity(texture: &ProtocolObject<dyn MTLTexture>) -> String {
    format!(
        "texture={texture:p} label={} storage={:?} usage={:#x} type={:?} array_length={}",
        optional_string(texture.label().map(|s| s.to_string()).as_deref()),
        texture.storageMode(),
        texture.usage().0,
        texture.textureType(),
        texture.arrayLength(),
    )
}

fn attachment_texture(texture: Option<&ProtocolObject<dyn MTLTexture>>, level: usize) -> String {
    let Some(texture) = texture else {
        return "missing".to_owned();
    };
    let endpoint = CopyEndpoint {
        pixel_format: texture.pixelFormat(),
        sample_count: texture.sampleCount(),
        width: texture.width(),
        height: texture.height(),
        depth: texture.depth(),
        level,
        levels: texture.mipmapLevelCount(),
        origin_x: 0,
        origin_y: 0,
    };
    format!("{} {endpoint}", texture_identity(texture))
}

fn attachment_details(attachment: &MTLRenderPassAttachmentDescriptor) -> String {
    format!(
        "texture={{{}}} level={} slice={} plane={} resolve={{{}}} \
         resolve_level={} resolve_slice={} resolve_plane={} load={:?} store={:?}",
        attachment_texture(attachment.texture().as_deref(), attachment.level()),
        attachment.level(),
        attachment.slice(),
        attachment.depthPlane(),
        attachment_texture(
            attachment.resolveTexture().as_deref(),
            attachment.resolveLevel(),
        ),
        attachment.resolveLevel(),
        attachment.resolveSlice(),
        attachment.resolveDepthPlane(),
        attachment.loadAction(),
        attachment.storeAction(),
    )
}

pub fn command_buffer(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
    let descriptor = diagnostic_descriptor(log_enabled!(target: LOG_TARGET, Level::Debug));
    descriptor.map_or_else(
        || queue.commandBuffer(),
        |descriptor| queue.commandBufferWithDescriptor(&descriptor),
    )
}

/// Observe a creation-time clear without publishing a frame retirement sequence.
pub fn observe_initialization(cb: &ProtocolObject<dyn MTLCommandBuffer>) {
    if !log_enabled!(target: LOG_TARGET, Level::Debug) {
        return;
    }
    let handler = RcBlock::new(
        |cb_ptr: core::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            autoreleasepool(|_| {
                // SAFETY: Metal supplies the completed buffer, valid for this invocation.
                // No reference to it escapes the callback.
                let cb = unsafe { cb_ptr.as_ref() };
                let status = cb.status();
                completion(cb, status, None, "initialization-callback");
                if status == MTLCommandBufferStatus::Error {
                    failure(cb, None, "initialization-callback", cb.error().as_deref());
                }
            });
        },
    );
    // Executable lifetime is separate from the block's captures: D3D CreateDevice sets
    // USED before any clear, so d3d9 detach self-terminates before Wine can unload its
    // statically imported shim and this Unix image. Native unit clients compile this
    // code into their test executable. Surviving direct-shim unload has no such contract.
    // Revisit this observer if D3D unload after CreateDevice can leave the process alive.
    // Process exit may end a callback before it logs; no completion is inferred then.
    // SAFETY: the live buffer has not been committed. Metal copies the valid block at
    // registration, so our handle may drop. The block captures nothing, uses only the
    // invocation's live argument and thread-safe diagnostics, and creates its own pool.
    unsafe { cb.addCompletedHandler(RcBlock::as_ptr(&handler)) };
}

pub fn completion(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    status: MTLCommandBufferStatus,
    seq: Option<u64>,
    site: &str,
) {
    if !log_enabled!(target: LOG_TARGET, Level::Debug) {
        return;
    }
    let label = cb.label().map(|s| s.to_string());
    let queue = cb.commandQueue();
    let device = cb.device();
    debug!(
        target: LOG_TARGET,
        "command-buffer buffer={cb:p} queue={:p} device={:p} registry_id={:#x} \
         device_name={:?} role={} seq={} site={site:?} status={}({}) \
         error_options={:#x} retained_references={} label={} queue_label={}",
        Retained::as_ptr(&queue),
        Retained::as_ptr(&device),
        device.registryID(),
        device.name().to_string(),
        buffer_role(label.as_deref()),
        sequence(seq),
        status.0,
        status_name(status),
        cb.errorOptions().0,
        cb.retainedReferences(),
        optional_string(label.as_deref()),
        optional_string(queue.label().map(|s| s.to_string()).as_deref()),
    );
}

/// Follows the completion envelope with details from the same error the caller handles.
pub fn failure(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    seq: Option<u64>,
    site: &str,
    error: Option<&NSError>,
) {
    if !log_enabled!(target: LOG_TARGET, Level::Debug) {
        return;
    }
    debug!(
        target: LOG_TARGET,
        "command-buffer-error buffer={cb:p} seq={} site={site:?} {}",
        sequence(seq),
        error_details(error),
    );
}

fn diagnostic_descriptor(enabled: bool) -> Option<Retained<MTLCommandBufferDescriptor>> {
    enabled.then(|| {
        let descriptor = MTLCommandBufferDescriptor::new();
        descriptor.setRetainedReferences(true);
        descriptor.setErrorOptions(MTLCommandBufferErrorOption::EncoderExecutionStatus);
        descriptor
    })
}

fn sequence(seq: Option<u64>) -> String {
    seq.map_or_else(|| "unavailable".to_owned(), |seq| format!("{seq:#x}"))
}

fn optional_string(value: Option<&str>) -> String {
    value.map_or_else(|| "missing".to_owned(), |value| format!("{value:?}"))
}

fn buffer_role(label: Option<&str>) -> &'static str {
    match label {
        Some(label) if label.starts_with("mtld3d-frame-") => "frame",
        Some(label) if label.starts_with("mtld3d-upload-") => "upload",
        Some("mtld3d-readback") => "readback",
        Some("mtld3d-init-clear") => "initialization",
        // Labels belong to the constructors; do not guess an unknown buffer's role.
        _ => "unknown",
    }
}

const fn status_name(status: MTLCommandBufferStatus) -> &'static str {
    match status {
        MTLCommandBufferStatus::NotEnqueued => "NotEnqueued",
        MTLCommandBufferStatus::Enqueued => "Enqueued",
        MTLCommandBufferStatus::Committed => "Committed",
        MTLCommandBufferStatus::Scheduled => "Scheduled",
        MTLCommandBufferStatus::Completed => "Completed",
        MTLCommandBufferStatus::Error => "Error",
        // The numeric status is printed beside this name for future SDK values.
        _ => "unrecognized",
    }
}

const fn encoder_state_name(state: MTLCommandEncoderErrorState) -> &'static str {
    match state {
        MTLCommandEncoderErrorState::Unknown => "Unknown",
        MTLCommandEncoderErrorState::Completed => "Completed",
        MTLCommandEncoderErrorState::Affected => "Affected",
        MTLCommandEncoderErrorState::Pending => "Pending",
        MTLCommandEncoderErrorState::Faulted => "Faulted",
        // Preserve future values without folding them into Metal's Unknown state.
        _ => "unrecognized",
    }
}

fn error_details(error: Option<&NSError>) -> String {
    let Some(error) = error else {
        return "error=missing encoder_info=unavailable".to_owned();
    };
    let mut output = format!(
        "error=present domain={:?} code={} description={:?}",
        error.domain().to_string(),
        error.code(),
        error.localizedDescription().to_string(),
    );
    // SAFETY: Metal exports this immutable Foundation string on every supported macOS.
    let key = unsafe { MTLCommandBufferEncoderInfoErrorKey };
    let payload = error.userInfo().objectForKey(key);
    append_encoder_info(&mut output, payload.as_deref());
    output
}

fn append_encoder_info(output: &mut String, payload: Option<&AnyObject>) {
    let Some(payload) = payload else {
        output.push_str(" encoder_info=missing-key");
        return;
    };
    let Some(encoders) = payload.downcast_ref::<NSArray<AnyObject>>() else {
        output.push_str(" encoder_info=malformed-non-array");
        return;
    };
    if encoders.is_empty() {
        output.push_str(" encoder_info=empty");
        return;
    }
    write!(
        output,
        " encoder_info=present encoder_count={}",
        encoders.len()
    )
    .expect("writing to a String cannot fail");
    let Some(protocol) = <dyn MTLCommandBufferEncoderInfo>::protocol() else {
        output.push_str(" encoders=unavailable-protocol");
        return;
    };
    for (index, object) in encoders.iter().enumerate() {
        write!(output, " encoder[{index}]={{").expect("writing to a String cannot fail");
        if !object.class().conforms_to(protocol) {
            output.push_str("malformed-nonconforming}");
            continue;
        }
        // SAFETY: the erased array yielded a retained live object, and its runtime class
        // conforms to MTLCommandBufferEncoderInfo as checked above. The protocol has no
        // extra Rust invariants; the cast transfers this retain without changing lifetime.
        let encoder = unsafe {
            Retained::cast_unchecked::<ProtocolObject<dyn MTLCommandBufferEncoderInfo>>(object)
        };
        let state = encoder.errorState();
        // The protocol's object getters can return nil despite their nonnull bindings.
        // NSObject's typed KVC getter preserves nil and calls these declared accessors.
        let metadata = AsRef::<AnyObject>::as_ref(&*encoder).downcast_ref::<NSObject>();
        let label = metadata.map_or_else(
            || "unavailable-non-nsobject".to_owned(),
            |object| metadata_string(object.valueForKey(ns_string!("label")).as_deref()),
        );
        write!(
            output,
            "label={label} state={}({}) signposts=",
            state.0,
            encoder_state_name(state),
        )
        .expect("writing to a String cannot fail");
        if let Some(object) = metadata {
            append_signposts(
                output,
                object.valueForKey(ns_string!("debugSignposts")).as_deref(),
            );
        } else {
            output.push_str("unavailable-non-nsobject");
        }
        output.push('}');
    }
}

fn metadata_string(value: Option<&AnyObject>) -> String {
    let Some(value) = value else {
        return optional_string(None);
    };
    value.downcast_ref::<NSString>().map_or_else(
        || "malformed-non-string".to_owned(),
        |value| optional_string(Some(&value.to_string())),
    )
}

fn append_signposts(output: &mut String, signposts: Option<&AnyObject>) {
    let Some(signposts) = signposts else {
        output.push_str("missing");
        return;
    };
    let Some(signposts) = signposts.downcast_ref::<NSArray<AnyObject>>() else {
        output.push_str("malformed-non-array");
        return;
    };
    if signposts.is_empty() {
        output.push_str("empty");
        return;
    }
    output.push('[');
    for (index, signpost) in signposts.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        output.push_str(&metadata_string(Some(&signpost)));
    }
    output.push(']');
}

#[cfg(test)]
mod tests;
