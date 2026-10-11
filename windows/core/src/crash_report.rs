//! Where a crash report goes, what the address it names lies in, and how a panic opens it.
//!
//! Two decisions the crash paths of `mtld3d.so` take, kept here because both
//! are pure and both are wrong in ways only a crash would otherwise show,
//! and the line `d3d9.dll`'s panic hook writes first.
//!
//! [`crash_route`] picks the destination of a fault report from what the
//! process's log sink knows and from how bad the fault is. Lines wait in a
//! backlog until `Direct3DCreate9` names the log location, so a process that
//! dies before then would lose the report and every line before it; the sink
//! therefore keeps an early location, the default one beside the executable,
//! which a terminal report opens instead of waiting. A first-chance report
//! never does: the fault may be one the game or Wine recovers (a protection
//! probe, a guard page, a handled access violation), and a process that lives
//! on must keep its log where `log.dir` puts it, or have none at all if it
//! never creates a device.
//!
//! [`fault_site`] decides whether an address `dladdr` attributes to a loaded
//! image is that image's content. `dyld` counts every segment of an image,
//! and a segment mapped from no file bytes is address space the image only
//! reserves: Wine's loader keeps the guest's low address space and its own
//! top-down heap as two such segments, so a guest address resolves to the
//! loader binary, at an offset that wraps below its load address.
//!
//! [`PanicLine`] is the line std's default panic hook prints to stderr,
//! which a game a launcher spawned has nowhere to show, formatted for the
//! process's log without allocating.

use core::{fmt::Write, panic::Location};

/// `mach_header_64.magic` of a 64-bit image in this byte order.
const MH_MAGIC_64: u32 = 0xfeed_facf;

/// The load command that maps one segment of a 64-bit image.
const LC_SEGMENT_64: u32 = 0x19;

/// Bytes in `mach_header_64`; the load commands follow it.
pub const MACHO_HEADER_LEN: usize = 32;

/// Bytes of `segment_command_64` up to and including `filesize`.
const SEGMENT_COMMAND_LEN: usize = 56;

/// `segname` of the segment that starts with the image's header.
const TEXT_SEGMENT: &[u8; 16] = b"__TEXT\0\0\0\0\0\0\0\0\0\0";

/// Bytes a [`PanicLine`] holds, its final line break included.
const PANIC_LINE_CAPACITY: usize = 2048;

/// What ends a [`PanicLine`] whose message did not fit, in place of the line break.
const PANIC_LINE_TRUNCATED: &str = " [truncated]\n";

/// What the process's log sink knows when a crash report arrives.
pub enum SinkState {
    /// No location yet: lines wait in the backlog.
    ///
    /// `early_location` says `InitLogger` named the default location, which
    /// a crash report may open ahead of `OpenLog`.
    Pending { early_location: bool },
    /// The location is named and the file not created yet.
    Named,
    /// The file is open.
    Open,
    /// Lines go to stderr.
    Stderr,
    /// Another holder has the sink, possibly the faulting thread itself.
    ///
    /// The report then never waits for it.
    Busy,
}

/// How bad the fault a report describes is.
pub enum Severity {
    /// A fault someone may still recover.
    ///
    /// The first chance of an exception, or a signal handed back to its owner.
    FirstChance,
    /// A fault that ends the process.
    ///
    /// A fault in our own code, an abort, a fatal exception code, a panic.
    Terminal,
}

/// Where a crash report is written from.
pub enum CrashContext {
    /// A signal handler: no allocation, no lock that waits, no `std::fs`.
    Signal,
    /// An ordinary thread, such as the PE exception handler's `WriteLog` call.
    ///
    /// It may allocate and create the file as any line does.
    Thread,
}

/// What a crash report does with the sink.
#[derive(Debug, PartialEq, Eq)]
pub enum CrashRoute {
    /// Write through the sink as any line is written.
    Sink,
    /// Open the early location and write the backlog ahead of the report.
    EarlyFile,
    /// Write the backlog and the report to stderr.
    ///
    /// Every later line of the dying process follows them there.
    Stderr,
    /// Leave the sink alone and write to its descriptor, the open file's or stderr's.
    Descriptor,
}

/// Whether an address inside an image is the image's content or space it only reserves.
#[derive(Debug, PartialEq, Eq)]
pub enum FaultSite {
    /// A segment mapped from the image's file: its code or its data.
    Image,
    /// A segment with no file content: address space the image reserves.
    ///
    /// The image owns the range, not what lives in it, so an offset into the
    /// image names nothing. Under Wine this is the guest's address space.
    Reserved,
}

/// One segment of a loaded image, at its address in this process.
pub struct ImageSegment {
    start: u64,
    size: u64,
    file_offset: u64,
    file_size: u64,
    text: bool,
}

impl ImageSegment {
    /// A segment of `size` bytes at `start`, mapping `file_size` bytes from `file_offset`.
    ///
    /// `text` marks the `__TEXT` segment, the one the header starts.
    #[must_use]
    pub const fn new(start: u64, size: u64, file_offset: u64, file_size: u64, text: bool) -> Self {
        Self {
            start,
            size,
            file_offset,
            file_size,
            text,
        }
    }

    /// True when `addr` lies in the segment.
    const fn contains(&self, addr: u64) -> bool {
        addr >= self.start && addr - self.start < self.size
    }

    /// True for a segment that maps the start of the image's file.
    const fn maps_file_start(&self) -> bool {
        self.file_offset == 0 && self.file_size > 0
    }
}

/// The segments of a 64-bit Mach-O image, read from its header in memory.
///
/// Built by [`macho_segments`]; yields nothing for a header it cannot read.
pub struct MachoSegments<'a> {
    commands: &'a [u8],
    remaining: u32,
    slide: u64,
}

impl MachoSegments<'_> {
    /// The next load command's `cmd` and bytes, or `None` where the commands stop making sense.
    fn take_command(&mut self) -> Option<(u32, &[u8])> {
        let cmd = read_u32(self.commands, 0)?;
        let size = usize::try_from(read_u32(self.commands, 4)?).ok()?;
        if size < 8 || size > self.commands.len() {
            return None;
        }
        let (command, rest) = self.commands.split_at(size);
        self.commands = rest;
        Some((cmd, command))
    }
}

impl Iterator for MachoSegments<'_> {
    type Item = ImageSegment;

    fn next(&mut self) -> Option<ImageSegment> {
        while self.remaining > 0 {
            self.remaining -= 1;
            let slide = self.slide;
            let Some((cmd, command)) = self.take_command() else {
                self.remaining = 0;
                return None;
            };
            if cmd != LC_SEGMENT_64 || command.len() < SEGMENT_COMMAND_LEN {
                continue;
            }
            return Some(ImageSegment::new(
                read_u64(command, 24)?.wrapping_add(slide),
                read_u64(command, 32)?,
                read_u64(command, 40)?,
                read_u64(command, 48)?,
                command.get(8..24) == Some(TEXT_SEGMENT.as_slice()),
            ));
        }
        None
    }
}

/// A panic's thread, location and message, worded as std's default hook prints them.
///
/// `<prefix>thread '<name>' (<id>) panicked at <file>:<line>:<column>:`,
/// then the message on the next line: the layout the default hook writes to
/// stderr, so the log carries what stderr does and a reader that knows one
/// finds the other. A payload that is not a string reads `Box<dyn Any>`, as
/// it does there. The text lives in a fixed buffer, since a panic can come
/// from a thread whose allocator state is not to be trusted; a message too
/// long for it is cut at a character boundary and ends in `[truncated]`.
pub struct PanicLine {
    buf: [u8; PANIC_LINE_CAPACITY],
    len: usize,
}

impl PanicLine {
    /// The line for a panic on thread `thread` (`thread_id`) at `location` with `message`.
    ///
    /// `prefix` opens the first line, the log target the way the other
    /// crash lines name it.
    #[must_use]
    pub fn new(
        prefix: &str,
        thread: &str,
        thread_id: u32,
        location: Option<&Location<'_>>,
        message: Option<&str>,
    ) -> Self {
        let mut buf = [0; PANIC_LINE_CAPACITY];
        let mut cursor = LineCursor::new(&mut buf);
        let message = message.unwrap_or("Box<dyn Any>");
        // An `Err` only says the text ran out of room, which the cursor
        // records, so the rest of the line is skipped and the end marked.
        let _ = match location {
            Some(location) => write!(
                cursor,
                "{prefix}thread '{thread}' ({thread_id}) panicked at {location}:\n{message}"
            ),
            None => write!(
                cursor,
                "{prefix}thread '{thread}' ({thread_id}) panicked at an unknown location:\n{message}"
            ),
        };
        let len = cursor.finish();
        Self { buf, len }
    }

    /// The line's bytes, UTF-8 and ending in a line break.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }
}

/// Where a fault report goes, given what the sink knows, how bad the fault is, and the caller.
///
/// Only a terminal report with no location named yet opens the early one,
/// so it and the backlog reach a file instead of dying with the process;
/// without an early location both go to stderr. A first-chance report goes
/// where any line goes, into the backlog while the location is pending, and
/// a signal handler's to the descriptor the sink already has, stderr until a
/// file is open. A signal handler never creates the file a named location
/// still waits for, and no report waits for a sink another holder has.
#[must_use]
pub const fn crash_route(
    state: &SinkState,
    severity: &Severity,
    context: &CrashContext,
) -> CrashRoute {
    match (state, severity) {
        (SinkState::Busy, _) => CrashRoute::Descriptor,
        (
            SinkState::Pending {
                early_location: true,
            },
            Severity::Terminal,
        ) => CrashRoute::EarlyFile,
        (
            SinkState::Pending {
                early_location: false,
            },
            Severity::Terminal,
        ) => CrashRoute::Stderr,
        _ => match context {
            CrashContext::Thread => CrashRoute::Sink,
            CrashContext::Signal => CrashRoute::Descriptor,
        },
    }
}

/// The byte count of the header and load commands at the start of `header`, or `None`.
///
/// Reads `sizeofcmds` from a 64-bit Mach-O header, so the caller knows how
/// much of the image to hand to [`macho_segments`]. `None` for anything that
/// is not one.
#[must_use]
pub fn macho_commands_len(header: &[u8]) -> Option<usize> {
    if read_u32(header, 0)? != MH_MAGIC_64 {
        return None;
    }
    let commands = usize::try_from(read_u32(header, 20)?).ok()?;
    MACHO_HEADER_LEN.checked_add(commands)
}

/// The segments of the 64-bit Mach-O image whose header and load commands are `image`.
///
/// `load_address` is where the header is mapped. Each segment is moved by
/// the image's slide: the load address less the address its header segment
/// (the one that maps the file from offset 0) asks for.
#[must_use]
pub fn macho_segments(image: &[u8], load_address: u64) -> MachoSegments<'_> {
    let commands = macho_commands_len(image).and_then(|len| image.get(MACHO_HEADER_LEN..len));
    let (Some(commands), Some(count)) = (commands, read_u32(image, 16)) else {
        return MachoSegments {
            commands: &[],
            remaining: 0,
            slide: 0,
        };
    };
    let unslid = MachoSegments {
        commands,
        remaining: count,
        slide: 0,
    };
    let slide = header_segment(unslid).map_or(0, |header| load_address.wrapping_sub(header.start));
    MachoSegments {
        commands,
        remaining: count,
        slide,
    }
}

/// Whether `addr`, inside an image dyld attributes it to, is that image's content.
///
/// An address in a segment with no file bytes is space the image reserves.
/// An address no segment covers (a header that could not be read) counts as
/// the image's, which is what `dladdr` said.
#[must_use]
pub fn fault_site(addr: u64, segments: impl IntoIterator<Item = ImageSegment>) -> FaultSite {
    segments
        .into_iter()
        .find(|segment| segment.contains(addr))
        .map_or(FaultSite::Image, |segment| {
            if segment.file_size == 0 {
                FaultSite::Reserved
            } else {
                FaultSite::Image
            }
        })
}

/// The write position in a [`PanicLine`]'s buffer while [`PanicLine::new`] formats it.
///
/// It keeps room for the end marker until [`Self::finish`], which writes the
/// marker and consumes the cursor, so nothing is written after the end.
struct LineCursor<'a> {
    buf: &'a mut [u8; PANIC_LINE_CAPACITY],
    len: usize,
    truncated: bool,
}

impl<'a> LineCursor<'a> {
    /// A cursor at the start of `buf`.
    const fn new(buf: &'a mut [u8; PANIC_LINE_CAPACITY]) -> Self {
        Self {
            buf,
            len: 0,
            truncated: false,
        }
    }

    /// End the line, with the marker when text was cut, and return its length.
    fn finish(self) -> usize {
        let end = if self.truncated {
            PANIC_LINE_TRUNCATED
        } else {
            "\n"
        };
        self.buf[self.len..self.len + end.len()].copy_from_slice(end.as_bytes());
        self.len + end.len()
    }
}

impl Write for LineCursor<'_> {
    /// Append `text`, or as much of it as fits ahead of the room the end marker keeps.
    fn write_str(&mut self, text: &str) -> core::fmt::Result {
        let room = (PANIC_LINE_CAPACITY - PANIC_LINE_TRUNCATED.len()).saturating_sub(self.len);
        let mut take = text.len().min(room);
        while !text.is_char_boundary(take) {
            take -= 1;
        }
        self.buf[self.len..self.len + take].copy_from_slice(&text.as_bytes()[..take]);
        self.len += take;
        if take < text.len() {
            self.truncated = true;
            return Err(core::fmt::Error);
        }
        Ok(())
    }
}

/// The segment that maps the image's header, the one the slide is measured from.
///
/// `__TEXT` by name first: an image in the dyld shared cache keeps the cache
/// file's offsets in its load commands, so its `__TEXT` does not map offset
/// 0. Failing the name, the segment that maps the start of the file. An
/// image with neither is left unslid; its segments then cover none of its
/// addresses, and [`fault_site`] answers what `dladdr` said.
fn header_segment(segments: MachoSegments<'_>) -> Option<ImageSegment> {
    let mut file_start = None;
    for segment in segments {
        if segment.text {
            return Some(segment);
        }
        if file_start.is_none() && segment.maps_file_start() {
            file_start = Some(segment);
        }
    }
    file_start
}

/// The native-endian `u32` at `offset`, or `None` past the end.
fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let word = bytes.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_ne_bytes(word.try_into().ok()?))
}

/// The native-endian `u64` at `offset`, or `None` past the end.
fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    let word = bytes.get(offset..offset.checked_add(8)?)?;
    Some(u64::from_ne_bytes(word.try_into().ok()?))
}

#[cfg(test)]
mod tests;
