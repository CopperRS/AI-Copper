use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Copia todos os arquivos de um diretório de origem para o diretório de destino
fn copy_all_files(source_dir: &Path, target_dir: &Path) {
    if !source_dir.exists() {
        println!("cargo:warning=Source directory {:?} does not exist", source_dir);
        return;
    }

    fs::create_dir_all(target_dir).unwrap();

    for entry in fs::read_dir(source_dir).expect("Failed to read source directory") {
        let entry = entry.expect("Invalid entry in source directory");
        let path = entry.path();
        let file_name = path.file_name().unwrap();
        let dest = target_dir.join(file_name);

        println!("cargo:warning=Copying {:?} -> {:?}", path, dest);

        if let Err(e) = fs::copy(&path, &dest) {
            println!("cargo:warning=Failed to copy {:?}: {}", file_name, e);
        } else {
            println!("cargo:warning=Successfully copied {:?}", file_name);
        }
    }
}

/// Copia uma biblioteca específica se existir
fn copy_specific_lib(source_path: &Path, target_dir: &Path) {
    if source_path.exists() {
        let file_name = source_path.file_name().unwrap();
        let dest = target_dir.join(file_name);
        println!("cargo:warning=Copying {:?} -> {:?}", source_path, dest);

        if let Err(e) = fs::copy(source_path, &dest) {
            println!("cargo:warning=Failed to copy {:?}: {}", file_name, e);
        } else {
            println!("cargo:warning=Successfully copied {:?}", file_name);
        }
    } else {
        println!("cargo:warning=Library not found at {:?}", source_path);
    }
}

/// Linka todas as bibliotecas encontradas em um diretório (.lib para Windows, .so para Linux)
fn link_all_libs_in_dir(dir: &Path, extension: &str) {
    if !dir.exists() {
        println!("cargo:warning=Library directory {:?} not found", dir);
        return;
    }

    let mut seen = HashSet::new();
    for entry in fs::read_dir(dir).expect("Failed to read library directory") {
        let entry = entry.expect("Invalid entry in library directory");
        let path = entry.path();

        if let Some(ext) = path.extension() {
            if ext == extension {
                if let Some(file_stem) = path.file_stem() {
                    let mut lib_name = file_stem.to_string_lossy().to_string();
                    if lib_name.starts_with("lib") && lib_name.len() > 3 {
                        lib_name = lib_name[3..].to_string();
                    }

                    if seen.insert(lib_name.clone()) {
                        println!("cargo:rustc-link-lib=dylib={}", lib_name);
                        println!("cargo:warning=Linked library: {}", lib_name);
                    }
                }
            }
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=cpp/lib.cpp");
    println!("cargo:rerun-if-changed=cpp/CMakeLists.txt");

    let cmake_build_dir = Path::new("cpp/build");
    if cmake_build_dir.exists() {
        println!("cargo:warning=Cleaning previous build directory...");
        for _ in 0..5 {
            if fs::remove_dir_all(&cmake_build_dir).is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
    fs::create_dir_all(&cmake_build_dir).unwrap();

    let torch_path = env::var("LIBTORCH").expect("LIBTORCH environment variable not set");
    let tf_path = env::var("TENSORFLOW_ROOT").expect("TENSORFLOW_ROOT environment variable not set");

    // Configura o CMake com gerador apropriado
    let mut cmake_args = vec![
        "-S".to_string(), "cpp".to_string(),
        "-B".to_string(), cmake_build_dir.display().to_string(),
        "-DCMAKE_BUILD_TYPE=Debug".to_string(),
        format!("-DCMAKE_PREFIX_PATH={}", torch_path),
    ];

    if cfg!(target_os = "windows") {
        cmake_args.push("-G".to_string());
        cmake_args.push("Visual Studio 17 2022".to_string());
        cmake_args.push("-A".to_string());
        cmake_args.push("x64".to_string());
    } else {
        cmake_args.push("-G".to_string());
        cmake_args.push("Unix Makefiles".to_string());
    }

    let cmake_status = Command::new("cmake")
        .args(&cmake_args)
        .env("LIBTORCH", &torch_path)
        .env("TENSORFLOW_ROOT", &tf_path)
        .status()
        .expect("Failed to run CMake configuration");

    if !cmake_status.success() {
        panic!("CMake configuration failed");
    }

    // Compila o projeto C++
    let build_args = vec![
        "--build".to_string(), cmake_build_dir.display().to_string(),
        "--config".to_string(), "Debug".to_string(),
    ];

    let build_status = Command::new("cmake")
        .args(&build_args)
        .status()
        .expect("Failed to run CMake build");

    if !build_status.success() {
        panic!("CMake build failed");
    }

    let build_dir_abs = cmake_build_dir.canonicalize().unwrap();

    // Configuração por sistema operacional
    if cfg!(target_os = "windows") {
        let lib_dir = build_dir_abs.join("Debug");
        let torch_lib = Path::new(&torch_path).join("lib");
        let tf_lib = Path::new(&tf_path).join("lib");

        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        link_all_libs_in_dir(&torch_lib, "lib");
        link_all_libs_in_dir(&tf_lib, "lib");

        println!("cargo:rustc-link-lib=dylib=ai_copper");
        println!("cargo:rustc-link-search=native={}", torch_lib.display());
        println!("cargo:rustc-link-search=native={}", tf_lib.display());

        // Calcula o target/debug do projeto consumidor a partir do OUT_DIR
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let target_dir = out_dir
            .ancestors()
            .nth(4) // Subir 4 níveis para alcançar o diretório raiz do projeto
            .unwrap()
            .join("debug");

        // Copia todos os arquivos dos diretórios relevantes
        copy_all_files(&lib_dir, &target_dir);
        copy_all_files(&torch_lib, &target_dir);
        copy_all_files(&tf_lib, &target_dir);

        // Copia explícita de todas as DLLs críticas
        let tf_dll_path = Path::new(&tf_path).join("lib").join("tensorflow.dll");
        copy_specific_lib(&tf_dll_path, &target_dir);
        let torch_dll_path = Path::new(&torch_path).join("lib").join("torch.dll");
        copy_specific_lib(&torch_dll_path, &target_dir);

        // Verificação detalhada das DLLs críticas
        let ai_copper_path = target_dir.join("ai_copper.dll");
        if !ai_copper_path.exists() {
            println!("cargo:warning=ai_copper.dll not found in {:?}", target_dir);
        } else {
            println!("cargo:warning=ai_copper.dll found in {:?}", target_dir);
        }
        let torch_path_check = target_dir.join("torch.dll");
        if !torch_path_check.exists() {
            println!("cargo:warning=torch.dll not found in {:?}", target_dir);
        } else {
            println!("cargo:warning=torch.dll found in {:?}", target_dir);
        }
        let tf_path_check = target_dir.join("tensorflow.dll");
        if !tf_path_check.exists() {
            println!("cargo:warning=tensorflow.dll not found in {:?}", target_dir);
        } else {
            println!("cargo:warning=tensorflow.dll found in {:?}", target_dir);
        }
    } else {
        let torch_lib = Path::new(&torch_path).join("lib");
        let tf_lib = Path::new(&tf_path).join("lib");

        println!("cargo:rustc-link-search=native={}", build_dir_abs.display());
        link_all_libs_in_dir(&torch_lib, "so");
        link_all_libs_in_dir(&tf_lib, "so");

        println!("cargo:rustc-link-lib=dylib=ai_copper");
        println!("cargo:rustc-link-search=native={}", torch_lib.display());
        println!("cargo:rustc-link-search=native={}", tf_lib.display());

        // Calcula o target/debug do projeto consumidor a partir do OUT_DIR
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let target_dir = out_dir
            .ancestors()
            .nth(3) // Subir 3 níveis para alcançar o diretório raiz do projeto
            .unwrap()
            .join("target")
            .join("debug");
        fs::create_dir_all(&target_dir).unwrap();

        // Copia todos os arquivos dos diretórios relevantes
        copy_all_files(&build_dir_abs, &target_dir);
        copy_all_files(&torch_lib, &target_dir);
        copy_all_files(&tf_lib, &target_dir);

        // Copia explícita das bibliotecas compartilhadas críticas
        let tf_so_path = Path::new(&tf_path).join("lib").join("libtensorflow.so");
        copy_specific_lib(&tf_so_path, &target_dir);
        let torch_so_path = Path::new(&torch_path).join("lib").join("libtorch.so");
        copy_specific_lib(&torch_so_path, &target_dir);

        // Verificação detalhada das bibliotecas críticas
        let ai_copper_path = target_dir.join("libai_copper.so");
        if !ai_copper_path.exists() {
            println!("cargo:warning=libai_copper.so not found in {:?}", target_dir);
        } else {
            println!("cargo:warning=libai_copper.so found in {:?}", target_dir);
        }
        let torch_path_check = target_dir.join("libtorch.so");
        if !torch_path_check.exists() {
            println!("cargo:warning=libtorch.so not found in {:?}", target_dir);
        } else {
            println!("cargo:warning=libtorch.so found in {:?}", target_dir);
        }
        let tf_path_check = target_dir.join("libtensorflow.so");
        if !tf_path_check.exists() {
            println!("cargo:warning=libtensorflow.so not found in {:?}", target_dir);
        } else {
            println!("cargo:warning=libtensorflow.so found in {:?}", target_dir);
        }

        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", target_dir.display());
    }
}