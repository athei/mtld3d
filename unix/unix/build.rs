fn main() {
    println!("cargo::rustc-check-cfg=cfg(mtld3d_crumb)");
    println!("cargo:rerun-if-env-changed=MTLD3D_CRUMB");
    if std::env::var("MTLD3D_CRUMB").is_ok_and(|v| !v.is_empty() && v != "0") {
        println!("cargo:rustc-cfg=mtld3d_crumb");
    }

    let target = std::env::var("TARGET").unwrap();
    if !target.contains("apple") {
        return;
    }

    println!("cargo:rustc-link-arg-cdylib=-Wl,-install_name,@rpath/mtld3d.so");
}
