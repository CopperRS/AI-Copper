use std::env;
use std::path::Path;
use std::process::Command;

fn main() {

    println!("cargo:rerun-if-changed=cpp/ai_copper.cpp");
    println!("cargo:rerun-if-changed=cpp/CMakeLists.txt");

    let cmake_build_dir = Path::new(&env::var("OUT_DIR").unwrap()).join("cmake_build");
    std::fs::create_dir_all(&cmake_build_dir).unwrap();


    Command::new("cmake")
        .arg("-S")
        .arg("cpp")
        .arg("-B")
        .arg(&cmake_build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .status()
        .unwrap();


    Command::new("cmake")
        .arg("--build")
        .arg(&cmake_build_dir)
        .status()
        .unwrap();

    println!("cargo:rustc-link-lib=dylib=ai_copper");
    println!("cargo:rustc-link-search=native={}", cmake_build_dir.display());

}
