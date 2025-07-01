use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    // Notify Cargo to rerun the build script if these files change
    println!("cargo:rerun-if-changed=cpp/ai_copper.cpp");
    println!("cargo:rerun-if-changed=cpp/CMakeLists.txt");

    // Define the directory where CMake will build the project
    let cmake_build_dir = Path::new(&env::var("OUT_DIR").unwrap()).join("cmake_build");
    // Create the build directory if it doesn't exist
    std::fs::create_dir_all(&cmake_build_dir).unwrap();

    // Run the CMake configuration step
    Command::new("cmake")
        .arg("-S") // Specify the source directory for CMake
        .arg("cpp") // The directory containing the C++ source code and CMakeLists.txt
        .arg("-B") // Specify the build directory for CMake
        .arg(&cmake_build_dir) // Path to the build directory
        .arg("-DCMAKE_BUILD_TYPE=Release") // Set the build type to Release
        .status()
        .unwrap(); // Panic if the command fails

    // Run the CMake build step
    Command::new("cmake")
        .arg("--build") // Build the project
        .arg(&cmake_build_dir) // Path to the build directory
        .status()
        .unwrap(); // Panic if the command fails

    // Link the generated dynamic library to the Rust project
    println!("cargo:rustc-link-lib=dylib=ai_copper");
    // Specify the directory where the dynamic library is located
    println!("cargo:rustc-link-search=native={}", cmake_build_dir.display());
}


































//Como dizia meu amigo Rodrigo Dias de Paula, "A Brenda é safada"
