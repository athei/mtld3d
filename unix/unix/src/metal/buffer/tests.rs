use mtld3d_core::page_box::PageBox;
use mtld3d_shared::{
    BufferCreateDesc, MetalHandle,
    mtl::{BufferKind, StorageMode},
};
use objc2::rc::{Retained, autoreleasepool};
use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};

use crate::metal::handle::IntoRetained;

#[test]
fn batch_creation_keeps_successes_and_clears_failures_after_pool_drain() {
    autoreleasepool(|_| {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device for buffer test");
        let backing = PageBox::new_zeroed(4096);
        let descriptor = |length| BufferCreateDesc {
            backing_ptr: backing.as_ptr() as u64,
            length,
            id: 41,
            storage_mode: StorageMode::Shared,
            kind: BufferKind::TexStaging,
        };
        let descriptors = [
            descriptor(backing.len() as u64),
            descriptor(0),
            descriptor(backing.len() as u64),
        ];
        let sentinel = device
            .newBufferWithLength_options(16, MTLResourceOptions::StorageModeShared)
            .expect("sentinel buffer");
        // SAFETY: sentinel owns this live buffer until every output slot is overwritten.
        let borrowed = unsafe { MetalHandle::new(Retained::as_ptr(&sentinel) as u64) };
        let mut handles = [borrowed; 3];
        assert!(!autoreleasepool(|_| super::create_buffers(
            &device,
            &descriptors,
            &mut handles,
        )));
        assert!(
            handles[1].is_null(),
            "failed creation clears a populated output"
        );
        for handle in [handles[0], handles[2]] {
            let buffer = handle
                .into_retained()
                .expect("successful output survives pool");
            assert_eq!(buffer.length(), backing.len());
            drop(buffer);
            super::destroy_buffer(handle.raw());
        }
        assert_eq!(
            sentinel.length(),
            16,
            "output replacement does not release caller handles"
        );
        assert!(autoreleasepool(|_| super::create_buffers(
            &device,
            &[],
            &mut []
        )));
    });
}
