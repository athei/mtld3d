//! Offline disassembler for DXSO bytecode dumped via `MTLD3D_BYTECODE_DUMP`.
//!
//! Usage: `cargo run --example disasm --target x86_64-apple-darwin -- <file.dxso>`
//!
//! Prints the parsed IR as Debug and the MSL the emitter produces, so a
//! captured bytecode blob can be inspected without launching Wine.

use std::{env, fs, process};

use mtld3d_core::dxso::{self, VariantKey};

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: disasm <file.dxso>");
        process::exit(2);
    });
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {path}: {e}");
            process::exit(2);
        }
    };
    if bytes.len() % 4 != 0 {
        eprintln!(
            "file size {} not a multiple of 4 — not a DXSO token stream",
            bytes.len()
        );
        process::exit(2);
    }
    let tokens: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    println!("== tokens ({n} words) ==", n = tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        println!("  [{i:3}] {t:#010x}");
    }
    println!();

    let program = match dxso::parse(&tokens) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("parse failed: {e:?}");
            process::exit(1);
        }
    };
    println!("== parsed IR ==");
    println!("shader_type: {:?}", program.shader_type);
    println!("declarations:");
    for d in &program.declarations {
        println!("  {d:?}");
    }
    println!("def constants:");
    for c in &program.def_constants {
        println!("  {c:?}");
    }
    println!("instructions:");
    for (i, inst) in program.instructions.iter().enumerate() {
        println!("  [{i:3}] {inst:#?}");
    }
    println!();

    let depth_mask: u16 = env::args()
        .nth(2)
        .and_then(|s| {
            s.strip_prefix("0x")
                .map_or_else(|| s.parse().ok(), |hex| u16::from_str_radix(hex, 16).ok())
        })
        .unwrap_or(0);
    let variant = VariantKey {
        depth_sampler_mask: depth_mask,
        ..VariantKey::default()
    };
    println!("== emitted MSL (depth_sampler_mask = {depth_mask:#x}) ==");
    let msl = match program.shader_type {
        dxso::ShaderType::Vertex => dxso::emit_vs_programmable(&program),
        dxso::ShaderType::Pixel => dxso::emit_ps_programmable(&program, variant),
    };
    match msl {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("emit failed: {e:?}"),
    }
}
