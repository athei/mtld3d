//! Unit tests for the benchmark subcommands' argument parsers.

use super::*;

fn args(tokens: &[&str]) -> std::vec::IntoIter<String> {
    tokens
        .iter()
        .map(|t| (*t).to_owned())
        .collect::<Vec<_>>()
        .into_iter()
}

const LEGS: [&str; 12] = [
    "--base-wine",
    "/b/wine",
    "--base-prefix",
    "/b/prefix",
    "--base-stamp",
    "v1",
    "--cand-wine",
    "/c/wine",
    "--cand-prefix",
    "/c/prefix",
    "--cand-stamp",
    "v2",
];

#[test]
fn compare_takes_a_directory_and_its_options() {
    let config = parse_compare(args(&[
        "/ab",
        "--accept",
        "perf.draws_pf, perf.passes_pf",
        "--accept",
        "perf.x",
        "--report",
        "/ab/report.txt",
        "--allow-same-image",
    ]))
    .unwrap();
    assert!(config.options.allow_same_image);
    assert_eq!(config.dir.to_str(), Some("/ab"));
    assert_eq!(
        config.options.accept,
        ["perf.draws_pf", "perf.passes_pf", "perf.x"]
    );
    assert_eq!(
        config.report.as_deref().and_then(|p| p.to_str()),
        Some("/ab/report.txt")
    );
}

#[test]
fn compare_rejects_what_it_does_not_know() {
    assert!(
        parse_compare(args(&[]))
            .unwrap_err()
            .contains("needs the A/B directory")
    );
    assert!(
        parse_compare(args(&["/a", "/b"]))
            .unwrap_err()
            .contains("second")
    );
    assert!(
        parse_compare(args(&["/a", "--same"]))
            .unwrap_err()
            .contains("unknown")
    );
    assert!(
        parse_compare(args(&["/a", "--report"]))
            .unwrap_err()
            .contains("needs a value")
    );
}

#[test]
fn ab_defaults_fill_in_behind_the_mandatory_flags() {
    let mut tokens = LEGS.to_vec();
    tokens.extend(["--out", "/ab", "--", "/e2e.exe"]);
    let config = parse_ab(args(&tokens)).unwrap();
    assert_eq!(config.base.wine.to_str(), Some("/b/wine"));
    assert_eq!(config.base.prefix.to_str(), Some("/b/prefix"));
    assert_eq!(config.base.stamp, "v1");
    assert_eq!(config.cand.stamp, "v2");
    assert_eq!(config.runs, 5);
    assert_eq!(config.timeout, Duration::from_mins(5));
    assert!(config.benches.is_empty());
    assert!(config.config.is_empty());
    assert!(config.report.is_none());
    assert!(!config.options.allow_same_image);
    assert_eq!(config.exes.len(), 1);
}

#[test]
fn ab_reads_every_optional_flag() {
    let mut tokens = LEGS.to_vec();
    tokens.extend([
        "--out",
        "/ab",
        "--runs",
        "7",
        "--bench",
        "wow_112 wow_335a",
        "--bench",
        "query_poll",
        "--config",
        "a=1;b=2",
        "--timeout",
        "90",
        "--accept",
        "perf.draws_pf",
        "--report",
        "/ab/r.txt",
        "--allow-same-image",
        "--",
        "/e2e.exe",
    ]);
    let config = parse_ab(args(&tokens)).unwrap();
    assert_eq!(config.runs, 7);
    assert_eq!(config.benches, ["wow_112", "wow_335a", "query_poll"]);
    assert_eq!(config.config, "a=1;b=2");
    assert_eq!(config.timeout, Duration::from_secs(90));
    assert_eq!(config.options.accept, ["perf.draws_pf"]);
    assert!(config.options.allow_same_image);
}

#[test]
fn ab_names_the_missing_flag() {
    let mut tokens = LEGS.to_vec();
    tokens.extend(["--out", "/ab"]);
    assert!(
        parse_ab(args(&tokens))
            .unwrap_err()
            .contains("no test binary")
    );

    let mut tokens = LEGS[2..].to_vec();
    tokens.extend(["--out", "/ab", "--", "/e2e.exe"]);
    assert!(
        parse_ab(args(&tokens))
            .unwrap_err()
            .contains("missing --base-wine")
    );

    let mut tokens = LEGS.to_vec();
    tokens.extend(["--", "/e2e.exe"]);
    assert!(
        parse_ab(args(&tokens))
            .unwrap_err()
            .contains("missing --out")
    );

    let mut tokens = LEGS.to_vec();
    tokens.extend(["--out", "/ab", "--runs", "0", "--", "/e2e.exe"]);
    assert!(parse_ab(args(&tokens)).unwrap_err().contains("count >= 1"));
}

#[test]
fn ab_needs_three_rounds_at_least() {
    for (runs, ok) in [("2", false), ("3", true)] {
        let mut tokens = LEGS.to_vec();
        tokens.extend(["--out", "/ab", "--runs", runs, "--", "/e2e.exe"]);
        let parsed = parse_ab(args(&tokens));
        assert_eq!(parsed.is_ok(), ok, "--runs {runs}");
        if let Err(reason) = parsed {
            assert!(reason.contains("at least 3"), "{reason}");
        }
    }
}

#[test]
fn ab_takes_both_host_emitters_and_their_corpora() {
    let mut tokens = LEGS.to_vec();
    tokens.extend([
        "--out",
        "/ab",
        "--base-host",
        "/b/emit_corpus",
        "--cand-host",
        "/c/emit_corpus",
        "--host-corpus",
        "/caches/a.bin",
        "--host-corpus",
        "/caches/b.bin",
        "--",
        "/e2e.exe",
    ]);
    let host = parse_ab(args(&tokens)).unwrap().host.expect("a host bench");
    assert_eq!(host.base.to_str(), Some("/b/emit_corpus"));
    assert_eq!(host.cand.to_str(), Some("/c/emit_corpus"));
    assert_eq!(
        host.corpora,
        [
            PathBuf::from("/caches/a.bin"),
            PathBuf::from("/caches/b.bin")
        ]
    );

    let mut tokens = LEGS.to_vec();
    tokens.extend(["--out", "/ab", "--", "/e2e.exe"]);
    assert!(parse_ab(args(&tokens)).unwrap().host.is_none());
}

#[test]
fn ab_rejects_a_host_emitter_for_one_leg_or_corpora_without_one() {
    for extra in [
        &["--base-host", "/b/emit_corpus"][..],
        &["--cand-host", "/c/emit_corpus"][..],
    ] {
        let mut tokens = LEGS.to_vec();
        tokens.extend(["--out", "/ab"]);
        tokens.extend(extra);
        tokens.extend(["--", "/e2e.exe"]);
        let reason = parse_ab(args(&tokens)).unwrap_err();
        assert!(reason.contains("go together"), "{reason}");
    }
    let mut tokens = LEGS.to_vec();
    tokens.extend(["--out", "/ab", "--host-corpus", "/a.bin", "--", "/e2e.exe"]);
    let reason = parse_ab(args(&tokens)).unwrap_err();
    assert!(reason.contains("--host-corpus without"), "{reason}");
}
