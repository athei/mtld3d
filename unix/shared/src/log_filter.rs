/// One-shot logger init shared by all three cdylibs (`d3d9.dll`, `mtld3d.dll`, `mtld3d.so`).
///
/// Each cdylib has its own copy of the `log` / `env_logger` statics, so each
/// calls this from its own entry point (`DllMain` on PE, the `InitLogger`
/// thunk on the unix side). `try_init` is idempotent so repeat calls silently
/// no-op.
///
/// Every line goes to the process's log file (see the `OpenLog` thunk), so
/// no target ever colours: `WriteStyle::Never` on all three linkage units
/// keeps escape sequences out of the file.
pub fn init_logger() {
    init(None);
}

/// Register `env_logger` writing every formatted line into `sink`.
///
/// The PE side uses this with a sink that crosses to the unix side, which
/// owns the process's log file; the unix side uses it with the file itself.
/// Filter and style match [`init_logger`].
pub fn init_logger_to(sink: Box<dyn std::io::Write + Send + 'static>) {
    init(Some(sink));
}

fn init(sink: Option<Box<dyn std::io::Write + Send + 'static>>) {
    let user = std::env::var("RUST_LOG").ok();
    let filter = resolved_log_filter(user.as_deref());
    let mut builder = env_logger::Builder::new();
    builder
        .parse_filters(&filter)
        .write_style(env_logger::WriteStyle::Never);
    if let Some(sink) = sink {
        builder.target(env_logger::Target::Pipe(sink));
    }
    let _ = builder.try_init();
}

fn resolved_log_filter(user: Option<&str>) -> String {
    match user {
        Some(s) if !s.is_empty() => format!("info,{s}"),
        _ => "info".into(),
    }
}

#[cfg(test)]
mod tests;
