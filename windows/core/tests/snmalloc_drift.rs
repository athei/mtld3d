//! Guards `page_box`'s size model against the production allocator configuration.
//!
//! The same shape assertions run in the PE drift binary. The Windows-only
//! decommit probe there separately checks the per-thread cache budget.

use snmalloc_rs::SnMalloc;

#[path = "snmalloc_drift/shape.rs"]
mod shape;

#[global_allocator]
static ALLOCATOR: SnMalloc = SnMalloc;
