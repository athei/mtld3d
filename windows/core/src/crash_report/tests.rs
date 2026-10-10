//! Crash-report routing and the reading of image segments.
//!
//! The routing table pins that a report arriving before `OpenLog` opens the
//! early location (or goes to stderr with the backlog when there is none),
//! and that a signal handler never creates a file the sink would or takes a
//! sink another holder has. The segment tests build Mach-O headers by hand:
//! the layout Wine's loader links with, whose zero-fill segments `dladdr`
//! attributes guest addresses to, and a slid image, so the slide is applied.

use super::*;

/// `LC_UUID`, a load command that is not a segment.
const LC_UUID: u32 = 0x1b;

/// One `LC_SEGMENT_64` as the linker writes it: name, addresses, file extent.
struct Segment {
    name: &'static str,
    vmaddr: u64,
    vmsize: u64,
    fileoff: u64,
    filesize: u64,
}

/// A 64-bit Mach-O header with `segments`, an `LC_UUID` after the first.
fn image(segments: &[Segment]) -> Vec<u8> {
    let mut commands = Vec::new();
    let mut count = 0u32;
    for (index, segment) in segments.iter().enumerate() {
        commands.extend_from_slice(&LC_SEGMENT_64.to_ne_bytes());
        commands.extend_from_slice(&72u32.to_ne_bytes());
        let mut name = [0u8; 16];
        name[..segment.name.len()].copy_from_slice(segment.name.as_bytes());
        commands.extend_from_slice(&name);
        for value in [
            segment.vmaddr,
            segment.vmsize,
            segment.fileoff,
            segment.filesize,
        ] {
            commands.extend_from_slice(&value.to_ne_bytes());
        }
        // maxprot, initprot, nsects, flags.
        commands.extend_from_slice(&[0u8; 16]);
        count += 1;
        if index == 0 {
            commands.extend_from_slice(&LC_UUID.to_ne_bytes());
            commands.extend_from_slice(&24u32.to_ne_bytes());
            commands.extend_from_slice(&[0xab; 16]);
            count += 1;
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(&MH_MAGIC_64.to_ne_bytes());
    // cputype, cpusubtype, filetype.
    out.extend_from_slice(&[0u8; 12]);
    out.extend_from_slice(&count.to_ne_bytes());
    out.extend_from_slice(
        &u32::try_from(commands.len())
            .expect("a test header fits u32")
            .to_ne_bytes(),
    );
    // flags, reserved.
    out.extend_from_slice(&[0u8; 8]);
    out.extend_from_slice(&commands);
    out
}

/// Wine's loader as its link flags lay it out: not slid, based at 8 GiB.
fn wine_loader() -> Vec<u8> {
    image(&[
        Segment {
            name: "__PAGEZERO",
            vmaddr: 0,
            vmsize: 0x1000,
            fileoff: 0,
            filesize: 0,
        },
        Segment {
            name: "WINE_RESERVE",
            vmaddr: 0x1000,
            vmsize: 0x1_ffff_f000,
            fileoff: 0,
            filesize: 0,
        },
        Segment {
            name: "WINE_TOP_DOWN",
            vmaddr: 0x7ff0_0000_0000,
            vmsize: 0x01ff_0000,
            fileoff: 0,
            filesize: 0,
        },
        Segment {
            name: "__TEXT",
            vmaddr: 0x2_0000_0000,
            vmsize: 0x4000,
            fileoff: 0,
            filesize: 0x4000,
        },
        Segment {
            name: "__DATA",
            vmaddr: 0x2_0000_4000,
            vmsize: 0x8000,
            fileoff: 0x4000,
            filesize: 0x1000,
        },
    ])
}

#[test]
fn a_guest_address_in_wines_reserve_is_not_the_loaders_content() {
    let header = wine_loader();
    let site = |addr| fault_site(addr, macho_segments(&header, 0x2_0000_0000));
    // The fault that read as `wine+0xfffffffe88681a46`.
    assert_eq!(site(0x8868_1a46), FaultSite::Reserved);
    assert_eq!(site(0x1000), FaultSite::Reserved);
    assert_eq!(site(0x1_ffff_ffff), FaultSite::Reserved);
    // Wine's top-down heap is reserved the same way.
    assert_eq!(site(0x7ff0_0000_1234), FaultSite::Reserved);
    // The loader's own code and data, its zero-filled tail included, are its content.
    assert_eq!(site(0x2_0000_0f00), FaultSite::Image);
    assert_eq!(site(0x2_0000_4100), FaultSite::Image);
    assert_eq!(site(0x2_0000_b000), FaultSite::Image);
}

#[test]
fn segments_move_with_the_slide() {
    let header = image(&[
        Segment {
            name: "__TEXT",
            vmaddr: 0x1_0000_0000,
            vmsize: 0x4000,
            fileoff: 0,
            filesize: 0x4000,
        },
        Segment {
            name: "RESERVE",
            vmaddr: 0x1_0000_4000,
            vmsize: 0x10000,
            fileoff: 0,
            filesize: 0,
        },
    ]);
    let load = 0x1_0400_0000;
    let segments: Vec<_> = macho_segments(&header, load)
        .map(|s| (s.start, s.size))
        .collect();
    assert_eq!(segments, [(load, 0x4000), (load + 0x4000, 0x10000)]);
    let site = |addr| fault_site(addr, macho_segments(&header, load));
    assert_eq!(site(load + 0x10), FaultSite::Image);
    assert_eq!(site(load + 0x5000), FaultSite::Reserved);
    // The unslid address is nothing of this image's any more.
    assert_eq!(site(0x1_0000_5000), FaultSite::Image);
}

#[test]
fn a_header_that_cannot_be_read_leaves_the_address_to_the_image() {
    let mut header = wine_loader();
    assert_eq!(
        macho_commands_len(&header),
        Some(header.len()),
        "the length covers the header and every command"
    );

    // Not a Mach-O header.
    header[0] ^= 0xff;
    assert_eq!(macho_commands_len(&header), None);
    assert_eq!(macho_segments(&header, 0).count(), 0);
    assert_eq!(
        fault_site(0x8868_1a46, macho_segments(&header, 0)),
        FaultSite::Image
    );
    header[0] ^= 0xff;

    // Commands cut short of what the header promises.
    let short = &header[..header.len() - 8];
    assert_eq!(macho_segments(short, 0).count(), 0);

    // A command whose size would never advance stops the walk.
    let mut stuck = wine_loader();
    stuck[MACHO_HEADER_LEN + 4..MACHO_HEADER_LEN + 8].copy_from_slice(&0u32.to_ne_bytes());
    assert_eq!(macho_segments(&stuck, 0x2_0000_0000).count(), 0);

    assert_eq!(macho_commands_len(&[0u8; 4]), None);
}

#[test]
fn a_report_before_the_location_is_named_opens_the_early_one() {
    for context in [CrashContext::Signal, CrashContext::Thread] {
        assert_eq!(
            crash_route(
                &SinkState::Pending {
                    early_location: true
                },
                &context
            ),
            CrashRoute::EarlyFile
        );
        // Without one, the backlog and the report go to stderr together.
        assert_eq!(
            crash_route(
                &SinkState::Pending {
                    early_location: false
                },
                &context
            ),
            CrashRoute::Stderr
        );
    }
}

#[test]
fn a_signal_handler_never_creates_a_file_nor_waits_for_the_sink() {
    for state in [
        SinkState::Named,
        SinkState::Open,
        SinkState::Stderr,
        SinkState::Busy,
    ] {
        assert_eq!(
            crash_route(&state, &CrashContext::Signal),
            CrashRoute::Descriptor
        );
    }
    for state in [SinkState::Named, SinkState::Open, SinkState::Stderr] {
        assert_eq!(crash_route(&state, &CrashContext::Thread), CrashRoute::Sink);
    }
    assert_eq!(
        crash_route(&SinkState::Busy, &CrashContext::Thread),
        CrashRoute::Descriptor
    );
}
