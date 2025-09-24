use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Copia arquivos dinâmicos (.dll ou .so) para o diretório de destino
fn copy_dynamic_libs_to_target(lib_dirs: &[&str], target_dir: &Path, exts: &[&str]) {
    for dir in lib_dirs {
        let source_dir = Path::new(dir);
        if source_dir.exists() {
            for entry in fs::read_dir(source_dir).expect("Falha ao ler diretório de libs") {
                let entry = entry.expect("Falha ao acessar entrada de diretório");
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if exts.contains(&ext.to_string_lossy().as_ref()) {
                        let file_name = path.file_name().unwrap();
                        let dest = target_dir.join(file_name);
                        println!("Copiando {:?} -> {:?}", path, dest);
                        let _ = fs::copy(&path, &dest);
                    }
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
        let _ = fs::remove_dir_all(&cmake_build_dir);
    }
    fs::create_dir_all(&cmake_build_dir).unwrap();

    let torch_path = env::var("LIBTORCH").expect("Variável LIBTORCH não definida.");
    let tensorflow_path = env::var("TENSORFLOW_ROOT").expect("Variável TENSORFLOW_ROOT não definida.");

    // Configuração do CMake
    let mut cmd = Command::new("cmake");
    let cmake_config = cmd
        .arg("-S").arg("cpp")
        .arg("-B").arg(&cmake_build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Debug")
        .arg(format!("-DCMAKE_PREFIX_PATH={}", torch_path))
        .env("LIBTORCH", &torch_path)
        .env("TENSORFLOW_ROOT", &tensorflow_path);

    if cfg!(target_os = "windows") {
        cmake_config.arg("-A").arg("x64");
    }

    let config_status = cmake_config.status().expect("Falha ao rodar configuração CMake");
    if !config_status.success() {
        panic!("CMake configuration failed");
    }

    let build_status = Command::new("cmake")
        .arg("--build").arg(&cmake_build_dir)
        .status()
        .expect("Falha ao rodar build CMake");
    if !build_status.success() {
        panic!("CMake build failed");
    }

    let build_dir_abs = cmake_build_dir.canonicalize().unwrap();

    // --- LINKAGEM ---
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-search=native={}", build_dir_abs.join("Debug").display());
        println!("cargo:rustc-link-lib=dylib=ai_copper");

        let torch_lib_dir = Path::new(&torch_path).join("lib");
        let tf_lib_dir = Path::new(&tensorflow_path).join("lib");

        println!("cargo:rustc-link-search=native={}", torch_lib_dir.display());
        println!("cargo:rustc-link-search=native={}", tf_lib_dir.display());

        for lib_dir in [&torch_lib_dir, &tf_lib_dir] {
            if lib_dir.exists() {
                for entry in fs::read_dir(lib_dir).unwrap() {
                    let entry = entry.unwrap();
                    let path = entry.path();
                    if let Some(ext) = path.extension() {
                        if ext == "lib" {
                            let fname = path.file_name().unwrap().to_string_lossy();
                            // NÃO pular protobuf
                            if fname.contains("d.lib") || fname.contains("lite") || fname.contains("ittnotify") {
                                continue;
                            }
                            let stem = path.file_stem().unwrap().to_string_lossy();
                            let libname = stem.strip_prefix("lib").unwrap_or(&stem);
                            println!("cargo:rustc-link-lib=dylib={}", libname);
                        }
                    }
                }
            }
        }

        // --- Copiar DLLs ---
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let mut target_dir = out_dir.clone();
        for _ in 0..3 { target_dir = target_dir.parent().unwrap().to_path_buf(); }

        let dll_dirs = [
            build_dir_abs.join("Debug"),
            Path::new(&torch_path).join("bin"),
            tf_lib_dir,
        ];
        let dll_dirs_str: Vec<String> = dll_dirs.iter().map(|p| p.to_string_lossy().into_owned()).collect();
        let dll_dirs_str_refs: Vec<&str> = dll_dirs_str.iter().map(|s| s.as_str()).collect();
        copy_dynamic_libs_to_target(&dll_dirs_str_refs, &target_dir, &["dll"]);
    } else {
        // Linux
        println!("cargo:rustc-link-search=native={}", build_dir_abs.display());
        println!("cargo:rustc-link-lib=dylib=ai_copper");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir_abs.display());

        let torch_lib_dir = Path::new(&torch_path).join("lib");
        let tf_lib_dir = Path::new(&tensorflow_path).join("lib");

        println!("cargo:rustc-link-search=native={}", torch_lib_dir.display());
        println!("cargo:rustc-link-search=native={}", tf_lib_dir.display());

        for lib_dir in [&torch_lib_dir, &tf_lib_dir] {
            if lib_dir.exists() {
                for entry in fs::read_dir(lib_dir).unwrap() {
                    let entry = entry.unwrap();
                    let path = entry.path();
                    if let Some(ext) = path.extension() {
                        if ext == "so" {
                            let stem = path.file_stem().unwrap().to_string_lossy();
                            let libname = stem.strip_prefix("lib").unwrap_or(&stem);
                            println!("cargo:rustc-link-lib=dylib={}", libname);
                        }
                    }
                }
            }
        }

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let mut target_dir = out_dir.clone();
        for _ in 0..3 { target_dir = target_dir.parent().unwrap().to_path_buf(); }

        let so_dirs = [torch_lib_dir, tf_lib_dir];
        let so_dirs_str: Vec<String> = so_dirs.iter().map(|p| p.to_string_lossy().into_owned()).collect();
        let so_dirs_str_refs: Vec<&str> = so_dirs_str.iter().map(|s| s.as_str()).collect();
        copy_dynamic_libs_to_target(&so_dirs_str_refs, &target_dir, &["so"]);
    }
}
