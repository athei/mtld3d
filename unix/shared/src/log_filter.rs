/// One-shot logger init shared by all three cdylibs (`d3d9.dll`, `mtld3d.dll`, `mtld3d.so`).
///
/// Each cdylib has its own copy of the `log` / `env_logger` statics, so each
/// calls this from its own entry point (`DllMain` on PE, the `InitLogger`
/// thunk on the unix side). `try_init` is idempotent so repeat calls silently
/// no-op.
///
/// Forces `WriteStyle::Always` on every target: the PE side runs inside
/// Wine where stderr-is-a-TTY auto-detection returns false even when
/// macOS fd 2 is a real terminal, and we want consistent colour across
/// all three linkage units. `NO_COLOR=1` opts out — `env_logger` only reads
/// it under `WriteStyle::Auto`, so we have to pre-resolve the choice here.
pub fn init_logger() {
    let user = std::env::var("RUST_LOG").ok();
    let filter = resolved_log_filter(user.as_deref());
    let style = if std::env::var_os("NO_COLOR").is_some() {
        env_logger::WriteStyle::Never
    } else {
        env_logger::WriteStyle::Always
    };
    let _ = env_logger::Builder::new()
        .parse_filters(&filter)
        .write_style(style)
        .try_init();
}

fn resolved_log_filter(user: Option<&str>) -> String {
    match user {
        Some(s) if !s.is_empty() => format!("info,{s}"),
        _ => "info".into(),
    }
}

#[cfg(test)]
mod tests;
