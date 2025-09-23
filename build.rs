use std::env;
use std::fs;
use std::path::{Path, PathBuf};  
use std::process::Command;

fn copy_dlls_to_target(dll_source_dirs: &[&str], target_dir: &Path) {
    for dir in dll_source_dirs {
        let source_dir = Path::new(dir);
        if source_dir.exists() {
            for entry in fs::read_dir(source_dir).expect("Failed to read DLL source directory") {
                let entry = entry.expect("Failed to get entry");
                let path = entry.path();
                if path.extension().map(|ext| ext.to_ascii_lowercase() == "dll").unwrap_or(false) {
                    let file_name = path.file_name().unwrap();
                    let dest = target_dir.join(file_name);
                    println!("Copying {:?} to {:?}", path, dest);
                    fs::copy(&path, &dest).expect("Failed to copy DLL");
                }
            }
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=cpp/lib.cpp");
    println!("cargo:rerun-if-changed=cpp/CMakeLists.txt");

    let cmake_build_dir = Path::new("cpp").join("build");

    if cmake_build_dir.exists() {
        std::fs::remove_dir_all(&cmake_build_dir).expect("Failed to clean cmake build directory");
    }
    std::fs::create_dir_all(&cmake_build_dir).unwrap();

    let mut cmake_config = Command::new("cmake");
    cmake_config
        .arg("-S")
        .arg("cpp")
        .arg("-B")
        .arg(&cmake_build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Debug"); 

    let torch_path = env::var("LIBTORCH").expect("LIBTORCH environment variable not set. Por favor, defina LIBTORCH com o caminho da sua instalação do libtorch.");
    let tensorflow_path = env::var("TENSORFLOW_ROOT").expect("TENSORFLOW_ROOT environment variable not set. Por favor, defina TENSORFLOW_ROOT com o caminho da sua instalação do TensorFlow.");
    if cfg!(target_os = "windows") {
        cmake_config.arg("-A").arg("x64");
    }
    cmake_config.arg(format!("-DCMAKE_PREFIX_PATH={}", torch_path));
    cmake_config.env("LIBTORCH", &torch_path);
    cmake_config.env("TENSORFLOW_ROOT", &tensorflow_path);

    let config_status = cmake_config.status().expect("Failed to run cmake configuration");
    if !config_status.success() {
        panic!("CMake configuration failed");
    }

    let build_status = Command::new("cmake")
        .arg("--build")
        .arg(&cmake_build_dir)
        .status()
        .expect("Failed to run cmake build");
    if !build_status.success() {
        panic!("CMake build failed");
    }

    let build_dir_abs = cmake_build_dir.canonicalize().unwrap();

    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-search=native={}", build_dir_abs.join("Debug").display());
        println!("cargo:rustc-link-lib=static=ai_copper");

        let torch_lib_dir = Path::new(&torch_path).join("lib");
        println!("cargo:rustc-link-search=native={}", torch_lib_dir.display());
        if torch_lib_dir.exists() {
            for entry in fs::read_dir(&torch_lib_dir).expect("Failed to read libtorch/lib directory") {
                let entry = entry.expect("Failed to get entry");
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "lib" {
                        if let Some(file_name) = path.file_name() {
                            let file_name = file_name.to_string_lossy();

                            if file_name.contains("ittnotify") || file_name.ends_with("d.lib") || file_name.contains("-lited") {
                                continue;
                            }
                            if let Some(file_stem) = path.file_stem() {
                                let lib_name = file_stem.to_string_lossy();
                                let lib_name = lib_name.strip_prefix("lib").unwrap_or(&lib_name);
                                println!("cargo:rustc-link-lib=dylib={}", lib_name);
                            }
                        }
                    }
                }
            }
        }

        let tf_lib_dir = Path::new(&tensorflow_path).join("lib");
        println!("cargo:rustc-link-search=native={}", tf_lib_dir.display());
        if tf_lib_dir.exists() {
            for entry in fs::read_dir(&tf_lib_dir).expect("Failed to read tensorflow/lib directory") {
                let entry = entry.expect("Failed to get entry");
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "lib" {
                        if let Some(file_name) = path.file_name() {
                            let file_name = file_name.to_string_lossy();
                            if file_name.ends_with("d.lib") || file_name.contains("-lited") {
                                continue;
                            }
                            if let Some(file_stem) = path.file_stem() {
                                let lib_name = file_stem.to_string_lossy();
                                let lib_name = lib_name.strip_prefix("lib").unwrap_or(&lib_name);
                                println!("cargo:rustc-link-lib=dylib={}", lib_name);
                            }
                        }
                    }
                }
            }
        }

        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
        let mut target_dir = PathBuf::from(&out_dir);
        for _ in 0..3 {
            target_dir = target_dir.parent().unwrap().to_path_buf();
        }

        let dll_dirs = [
            Path::new(&torch_path).join("lib").to_string_lossy().to_string(),
            Path::new(&tensorflow_path).join("lib").to_string_lossy().to_string(),
        ];
        let dll_dirs_ref: Vec<&str> = dll_dirs.iter().map(|s| s.as_str()).collect();
        copy_dlls_to_target(&dll_dirs_ref, &target_dir);
    } else {
        println!("cargo:rustc-link-search=native={}", build_dir_abs.display());
        println!("cargo:rustc-link-lib=dylib=ai_copper");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir_abs.display());

        let torch_lib_dir = Path::new(&torch_path).join("lib");
        println!("cargo:rustc-link-search=native={}", torch_lib_dir.display());
        if torch_lib_dir.exists() {
            for entry in fs::read_dir(&torch_lib_dir).expect("Failed to read libtorch/lib directory") {
                let entry = entry.expect("Failed to get entry");
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "so" {
                        if let Some(file_stem) = path.file_stem() {
                            let lib_name = file_stem.to_string_lossy();
                            let lib_name = lib_name.strip_prefix("lib").unwrap_or(&lib_name);
                            println!("cargo:rustc-link-lib=dylib={}", lib_name);
                        }
                    }
                }
            }
        }

        let tf_lib_dir = Path::new(&tensorflow_path).join("lib");
        println!("cargo:rustc-link-search=native={}", tf_lib_dir.display());
        if tf_lib_dir.exists() {
            for entry in fs::read_dir(&tf_lib_dir).expect("Failed to read tensorflow/lib directory") {
                let entry = entry.expect("Failed to get entry");
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "so" {
                        if let Some(file_stem) = path.file_stem() {
                            let lib_name = file_stem.to_string_lossy();
                            let lib_name = lib_name.strip_prefix("lib").unwrap_or(&lib_name);
                            println!("cargo:rustc-link-lib=dylib={}", lib_name);
                        }
                    }
                }
            }
        }


        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
        let mut target_dir = PathBuf::from(&out_dir);
        for _ in 0..3 {
            target_dir = target_dir.parent().unwrap().to_path_buf();
        }
        let so_dirs = [
            Path::new(&torch_path).join("lib").to_string_lossy().to_string(),
            Path::new(&tensorflow_path).join("lib").to_string_lossy().to_string(),
        ];
        let so_dirs_ref: Vec<&str> = so_dirs.iter().map(|s| s.as_str()).collect();
        copy_dlls_to_target(&so_dirs_ref, &target_dir);
        println!("cargo:rustc-env=LD_LIBRARY_PATH={}", build_dir_abs.display());
    }
}

