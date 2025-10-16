use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// URLs para download das bibliotecas
const LIBTORCH_WINDOWS_URL: &str = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.1.0%2Bcpu.zip";
const LIBTORCH_LINUX_URL: &str = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip";
const TENSORFLOW_WINDOWS_URL: &str = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-2.10.0.zip";
const TENSORFLOW_LINUX_URL: &str = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.10.0.tar.gz";

/// Baixa um arquivo da URL e salva no destino
fn download_file(url: &str, dest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use reqwest::blocking::Client;
    use std::io::Write;

    println!("cargo:warning=Baixando de: {}", url);
    println!("cargo:warning=Salvando em: {:?}", dest);

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()?;
    
    let response = client.get(url).send()?;
    
    if !response.status().is_success() {
        return Err(format!("Falha no download: HTTP {}", response.status()).into());
    }

    fs::create_dir_all(dest.parent().unwrap())?;
    let mut file = fs::File::create(dest)?;
    let content = response.bytes()?;
    file.write_all(&content)?;

    println!("cargo:warning=Download concluído!");
    Ok(())
}

/// Extrai um arquivo ZIP
fn extract_zip(zip_path: &Path, dest_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use zip::ZipArchive;
    
    println!("cargo:warning=Extraindo ZIP: {:?}", zip_path);
    let file = fs::File::open(zip_path)?;
    let mut archive = ZipArchive::new(file)?;
    
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = dest_dir.join(file.name());
        
        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                fs::create_dir_all(p)?;
            }
            let mut outfile = fs::File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;
        }
    }
    
    println!("cargo:warning=Extração ZIP concluída!");
    Ok(())
}

/// Extrai um arquivo TAR.GZ
fn extract_tar_gz(tar_path: &Path, dest_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use flate2::read::GzDecoder;
    use tar::Archive;
    
    println!("cargo:warning=Extraindo TAR.GZ: {:?}", tar_path);
    let file = fs::File::open(tar_path)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    archive.unpack(dest_dir)?;
    
    println!("cargo:warning=Extração TAR.GZ concluída!");
    Ok(())
}

/// Garante que uma biblioteca está disponível (baixa se necessário)
fn ensure_library(lib_name: &str, lib_dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if lib_dir.exists() {
        println!("cargo:warning={} já existe em: {:?}", lib_name, lib_dir);
        set_environment_variables(lib_name, lib_dir)?;
        return Ok(lib_dir.to_path_buf());
    }

    println!("cargo:warning={} não encontrado, iniciando download...", lib_name);

    // Define diretório de destino baseado no sistema operacional
    let (dest_root, temp_dir) = if cfg!(target_os = "windows") {
        (PathBuf::from("C:\\"), PathBuf::from(env::temp_dir()).join("ai_copper_downloads"))
    } else {
        (PathBuf::from("/opt"), PathBuf::from(env::temp_dir()).join("ai_copper_downloads"))
    };

    fs::create_dir_all(&temp_dir)?;

    match lib_name {
        "libtorch" if cfg!(target_os = "windows") => {
            let final_dir = dest_root.join("libtorch");
            let zip_file = temp_dir.join("libtorch.zip");
            
            if !zip_file.exists() {
                download_file(LIBTORCH_WINDOWS_URL, &zip_file)?;
            }
            
            // Extrai temporariamente
            let temp_extract = temp_dir.join("libtorch_extract");
            fs::create_dir_all(&temp_extract)?;
            extract_zip(&zip_file, &temp_extract)?;
            
            // Move para C:\libtorch (remove subpasta duplicada se existir)
            let extracted_libtorch = temp_extract.join("libtorch");
            if extracted_libtorch.exists() {
                move_directory(&extracted_libtorch, &final_dir)?;
            } else {
                move_directory(&temp_extract, &final_dir)?;
            }
            
            // Limpa arquivos temporários
            let _ = fs::remove_file(&zip_file);
            let _ = fs::remove_dir_all(&temp_extract);
            
            println!("cargo:warning=LibTorch instalado em: {:?}", final_dir);
            set_environment_variables(lib_name, &final_dir)?;
            Ok(final_dir)
        }
        "libtorch" => {
            let final_dir = dest_root.join("libtorch");
            let tar_file = temp_dir.join("libtorch.tar.gz");
            
            if !tar_file.exists() {
                download_file(LIBTORCH_LINUX_URL, &tar_file)?;
            }
            
            // Extrai temporariamente
            let temp_extract = temp_dir.join("libtorch_extract");
            fs::create_dir_all(&temp_extract)?;
            extract_tar_gz(&tar_file, &temp_extract)?;
            
            // Move para /opt/libtorch
            let extracted_libtorch = temp_extract.join("libtorch");
            if extracted_libtorch.exists() {
                move_directory(&extracted_libtorch, &final_dir)?;
            } else {
                move_directory(&temp_extract, &final_dir)?;
            }
            
            // Limpa arquivos temporários
            let _ = fs::remove_file(&tar_file);
            let _ = fs::remove_dir_all(&temp_extract);
            
            println!("cargo:warning=LibTorch instalado em: {:?}", final_dir);
            set_environment_variables(lib_name, &final_dir)?;
            Ok(final_dir)
        }
        "tensorflow" if cfg!(target_os = "windows") => {
            let final_dir = dest_root.join("libtensorflow");
            let zip_file = temp_dir.join("tensorflow.zip");
            
            if !zip_file.exists() {
                download_file(TENSORFLOW_WINDOWS_URL, &zip_file)?;
            }
            
            // Extrai diretamente em C:\libtensorflow
            fs::create_dir_all(&final_dir)?;
            extract_zip(&zip_file, &final_dir)?;
            
            // Limpa arquivos temporários
            let _ = fs::remove_file(&zip_file);
            
            println!("cargo:warning=TensorFlow instalado em: {:?}", final_dir);
            set_environment_variables(lib_name, &final_dir)?;
            Ok(final_dir)
        }
        "tensorflow" => {
            let final_dir = dest_root.join("libtensorflow");
            let tar_file = temp_dir.join("tensorflow.tar.gz");
            
            if !tar_file.exists() {
                download_file(TENSORFLOW_LINUX_URL, &tar_file)?;
            }
            
            fs::create_dir_all(&final_dir)?;
            extract_tar_gz(&tar_file, &final_dir)?;
            
            // Limpa arquivos temporários
            let _ = fs::remove_file(&tar_file);
            
            println!("cargo:warning=TensorFlow instalado em: {:?}", final_dir);
            set_environment_variables(lib_name, &final_dir)?;
            Ok(final_dir)
        }
        _ => Err(format!("Biblioteca desconhecida: {}", lib_name).into())
    }
}

/// Move um diretório completo de origem para destino
fn move_directory(source: &Path, dest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if dest.exists() {
        println!("cargo:warning=Removendo diretório existente: {:?}", dest);
        fs::remove_dir_all(dest)?;
    }
    
    println!("cargo:warning=Movendo {:?} para {:?}", source, dest);
    
    // Tenta mover diretamente primeiro (mais rápido)
    if let Err(_) = fs::rename(source, dest) {
        // Se falhar, copia e remove
        copy_directory_recursive(source, dest)?;
        fs::remove_dir_all(source)?;
    }
    
    Ok(())
}

/// Copia recursivamente um diretório
fn copy_directory_recursive(source: &Path, dest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(dest)?;
    
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        let dest_path = dest.join(entry.file_name());
        
        if path.is_dir() {
            copy_directory_recursive(&path, &dest_path)?;
        } else {
            fs::copy(&path, &dest_path)?;
        }
    }
    
    Ok(())
}

/// Configura variáveis de ambiente do sistema
fn set_environment_variables(lib_name: &str, lib_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "windows")]
    {
        let lib_dir = lib_path.join("lib");
        let bin_dir = lib_path.join("bin");
        let include_dir = lib_path.join("include");
        
        match lib_name {
            "libtorch" => {
                println!("cargo:warning=Configurando variáveis de ambiente para LibTorch...");
                
                // 1. TORCH_HOME - Raiz da instalação do libtorch
                set_env_var("TORCH_HOME", &lib_path.display().to_string())?;
                
                // 2. LD_LIBRARY_PATH - Caminho das libs (.so) no Linux/macOS
                let ld_lib_path = lib_dir.display().to_string();
                set_env_var("LD_LIBRARY_PATH", &ld_lib_path)?;
                
                // 3. LIBRARY_PATH - Caminho de linkagem (para compilação)
                set_env_var("LIBRARY_PATH", &ld_lib_path)?;
                
                // 4. CPATH - Caminho dos headers (.h)
                if include_dir.exists() {
                    set_env_var("CPATH", &include_dir.display().to_string())?;
                }
                
                // 5. PATH - (Windows) Caminho das DLLs
                add_to_path_windows(&lib_dir)?;
                if bin_dir.exists() {
                    add_to_path_windows(&bin_dir)?;
                }
                
                // 6. CMAKE_PREFIX_PATH - Para CMake encontrar automaticamente o pacote Torch
                set_env_var("CMAKE_PREFIX_PATH", &lib_path.display().to_string())?;
                
                println!("cargo:warning=✓ TORCH_HOME={}", lib_path.display());
                println!("cargo:warning=✓ LD_LIBRARY_PATH={}", ld_lib_path);
                println!("cargo:warning=✓ LIBRARY_PATH={}", ld_lib_path);
                if include_dir.exists() {
                    println!("cargo:warning=✓ CPATH={}", include_dir.display());
                }
                println!("cargo:warning=✓ CMAKE_PREFIX_PATH={}", lib_path.display());
                println!("cargo:warning=✓ PATH atualizado com lib e bin");
            }
            "tensorflow" => {
                println!("cargo:warning=Configurando variáveis de ambiente para TensorFlow...");
                
                // 1. TENSORFLOW_ROOT - Raiz da instalação do TensorFlow
                set_env_var("TENSORFLOW_ROOT", &lib_path.display().to_string())?;
                
                // 2. LD_LIBRARY_PATH - Caminho das libs dinâmicas no Linux/macOS
                let ld_lib_path = lib_dir.display().to_string();
                set_env_var("LD_LIBRARY_PATH", &ld_lib_path)?;
                
                // 3. LIBRARY_PATH - Caminho de linkagem (para compilação)
                set_env_var("LIBRARY_PATH", &ld_lib_path)?;
                
                // 4. CPATH - Caminho dos headers (.h)
                if include_dir.exists() {
                    set_env_var("CPATH", &include_dir.display().to_string())?;
                }
                
                // 5. PATH - (Windows) Caminho das DLLs
                add_to_path_windows(&lib_dir)?;
                if bin_dir.exists() {
                    add_to_path_windows(&bin_dir)?;
                }
                
                println!("cargo:warning=✓ TENSORFLOW_ROOT={}", lib_path.display());
                println!("cargo:warning=✓ LD_LIBRARY_PATH={}", ld_lib_path);
                println!("cargo:warning=✓ LIBRARY_PATH={}", ld_lib_path);
                if include_dir.exists() {
                    println!("cargo:warning=✓ CPATH={}", include_dir.display());
                }
                println!("cargo:warning=✓ PATH atualizado com lib e bin");
            }
            _ => {}
        }
        
        println!("cargo:warning=Variáveis de ambiente configuradas. Reinicie o terminal para que tenham efeito.");
    }
    
    #[cfg(not(target_os = "windows"))]
    {
        // Linux/macOS
        let lib_dir = lib_path.join("lib");
        let include_dir = lib_path.join("include");
        
        match lib_name {
            "libtorch" => {
                println!("cargo:warning=Variáveis configuradas no ambiente de build:");
                println!("cargo:warning=Para persistir, adicione ao ~/.bashrc ou ~/.zshrc:");
                println!("cargo:warning=  export TORCH_HOME={}", lib_path.display());
                println!("cargo:warning=  export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH", lib_dir.display());
                println!("cargo:warning=  export LIBRARY_PATH={}:$LIBRARY_PATH", lib_dir.display());
                if include_dir.exists() {
                    println!("cargo:warning=  export CPATH={}:$CPATH", include_dir.display());
                }
                println!("cargo:warning=  export CMAKE_PREFIX_PATH={}", lib_path.display());
            }
            "tensorflow" => {
                println!("cargo:warning=Variáveis configuradas no ambiente de build:");
                println!("cargo:warning=Para persistir, adicione ao ~/.bashrc ou ~/.zshrc:");
                println!("cargo:warning=  export TENSORFLOW_ROOT={}", lib_path.display());
                println!("cargo:warning=  export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH", lib_dir.display());
                println!("cargo:warning=  export LIBRARY_PATH={}:$LIBRARY_PATH", lib_dir.display());
                if include_dir.exists() {
                    println!("cargo:warning=  export CPATH={}:$CPATH", include_dir.display());
                }
            }
            _ => {}
        }
    }
    
    Ok(())
}

/// Define uma variável de ambiente do usuário (Windows)
#[cfg(target_os = "windows")]
fn set_env_var(var_name: &str, value: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;
    
    let output = Command::new("powershell")
        .args(&[
            "-Command",
            &format!("[Environment]::SetEnvironmentVariable('{}', '{}', 'User')", var_name, value)
        ])
        .output()?;
    
    if !output.status.success() {
        println!("cargo:warning=Falha ao definir {}: {}", var_name, String::from_utf8_lossy(&output.stderr));
    }
    
    Ok(())
}

/// Adiciona um diretório ao PATH do Windows
#[cfg(target_os = "windows")]
fn add_to_path_windows(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;
    
    if !dir.exists() {
        return Ok(());
    }
    
    let dir_str = dir.display().to_string();
    
    println!("cargo:warning=Adicionando ao PATH: {}", dir_str);
    
    // Obtém o PATH atual do usuário
    let output = Command::new("powershell")
        .args(&[
            "-Command",
            "[Environment]::GetEnvironmentVariable('Path', 'User')"
        ])
        .output()?;
    
    let current_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    
    // Verifica se já está no PATH
    if current_path.split(';').any(|p| p.trim() == dir_str) {
        println!("cargo:warning=Diretório já está no PATH: {}", dir_str);
        return Ok(());
    }
    
    // Adiciona ao PATH
    let new_path = if current_path.is_empty() {
        dir_str
    } else {
        format!("{};{}", current_path, dir_str)
    };
    
    let output = Command::new("powershell")
        .args(&[
            "-Command",
            &format!("[Environment]::SetEnvironmentVariable('Path', '{}', 'User')", new_path)
        ])
        .output()?;
    
    if output.status.success() {
        println!("cargo:warning=PATH atualizado com sucesso");
    } else {
        println!("cargo:warning=Falha ao atualizar PATH: {}", String::from_utf8_lossy(&output.stderr));
    }
    
    Ok(())
}


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

    // Verifica features habilitadas
    let use_libtorch = cfg!(feature = "libtorch");
    let use_tensorflow = cfg!(feature = "tensorflow");

    if !use_libtorch && !use_tensorflow {
        panic!("Pelo menos um backend (libtorch ou tensorflow) deve estar habilitado!");
    }

    let cmake_build_dir = Path::new("cpp/build");
    if cmake_build_dir.exists() {
        println!("cargo:warning=Limpando diretório de build anterior...");
        for _ in 0..5 {
            if fs::remove_dir_all(&cmake_build_dir).is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
    fs::create_dir_all(&cmake_build_dir).unwrap();

    // Garante que as bibliotecas estejam disponíveis
    let torch_path = if use_libtorch {
        // Verifica primeiro em C:\libtorch (instalação automática)
        let system_torch_path = if cfg!(target_os = "windows") {
            PathBuf::from("C:\\libtorch")
        } else {
            PathBuf::from("/opt/libtorch")
        };
        
        // Tenta usar variável de ambiente ou usa o caminho do sistema
        match env::var("LIBTORCH") {
            Ok(path) => {
                println!("cargo:warning=Usando LIBTORCH de variável de ambiente: {}", path);
                PathBuf::from(path)
            }
            Err(_) => {
                if system_torch_path.exists() {
                    println!("cargo:warning=Usando LibTorch instalado em: {:?}", system_torch_path);
                    set_environment_variables("libtorch", &system_torch_path)
                        .unwrap_or_else(|e| println!("cargo:warning=Erro ao configurar variáveis: {}", e));
                    system_torch_path
                } else {
                    println!("cargo:warning=LIBTORCH não definido, fazendo download automático...");
                    ensure_library("libtorch", &system_torch_path)
                        .expect("Falha ao baixar LibTorch")
                }
            }
        }
    } else {
        PathBuf::from("deps/libtorch") // Caminho dummy se não estiver usando
    };

    let tf_path = if use_tensorflow {
        // Verifica primeiro em C:\libtensorflow (instalação automática)
        let system_tf_path = if cfg!(target_os = "windows") {
            PathBuf::from("C:\\libtensorflow")
        } else {
            PathBuf::from("/opt/libtensorflow")
        };
        
        // Tenta usar variável de ambiente ou usa o caminho do sistema
        match env::var("TENSORFLOW_ROOT") {
            Ok(path) => {
                println!("cargo:warning=Usando TENSORFLOW_ROOT de variável de ambiente: {}", path);
                PathBuf::from(path)
            }
            Err(_) => {
                if system_tf_path.exists() {
                    println!("cargo:warning=Usando TensorFlow instalado em: {:?}", system_tf_path);
                    set_environment_variables("tensorflow", &system_tf_path)
                        .unwrap_or_else(|e| println!("cargo:warning=Erro ao configurar variáveis: {}", e));
                    system_tf_path
                } else {
                    println!("cargo:warning=TENSORFLOW_ROOT não definido, fazendo download automático...");
                    ensure_library("tensorflow", &system_tf_path)
                        .expect("Falha ao baixar TensorFlow")
                }
            }
        }
    } else {
        PathBuf::from("deps/tensorflow") // Caminho dummy se não estiver usando
    };

    // Configura o CMake com gerador apropriado
    let mut cmake_args = vec![
        "-S".to_string(), "cpp".to_string(),
        "-B".to_string(), cmake_build_dir.display().to_string(),
        "-DCMAKE_BUILD_TYPE=Release".to_string(),
    ];

    if use_libtorch {
        cmake_args.push(format!("-DCMAKE_PREFIX_PATH={}", torch_path.display()));
    }

    if use_libtorch {
        cmake_args.push(format!("-DCMAKE_PREFIX_PATH={}", torch_path.display()));
    }

    if cfg!(target_os = "windows") {
        cmake_args.push("-G".to_string());
        cmake_args.push("Visual Studio 17 2022".to_string());
        cmake_args.push("-A".to_string());
        cmake_args.push("x64".to_string());
    } else {
        cmake_args.push("-G".to_string());
        cmake_args.push("Unix Makefiles".to_string());
    }

    let mut cmake_cmd = Command::new("cmake");
    cmake_cmd.args(&cmake_args);
    
    if use_libtorch {
        cmake_cmd.env("LIBTORCH", &torch_path);
    }
    if use_tensorflow {
        cmake_cmd.env("TENSORFLOW_ROOT", &tf_path);
    }

    let cmake_status = cmake_cmd
        .status()
        .expect("Failed to run CMake configuration");

    if !cmake_status.success() {
        panic!("CMake configuration failed");
    }

    // Compila o projeto C++
    let build_args = vec![
        "--build".to_string(), cmake_build_dir.display().to_string(),
        "--config".to_string(), "Release".to_string(),
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
        if use_libtorch {
            let torch_lib = torch_path.join("lib");

            let renames = vec![
                (torch_lib.join("libprotobuf.lib"), torch_lib.join("protobuf.lib")),
                (torch_lib.join("libprotoc.lib"), torch_lib.join("protoc.lib")),
                (torch_lib.join("libittnotify.lib"), torch_lib.join("ittnotify.lib")),
                (torch_lib.join("libprotobuf-lite.lib"), torch_lib.join("protobuf-lite.lib")),
            ];

            for (from, to) in renames {
                if from.exists() && !to.exists() {
                    if let Err(e) = fs::copy(&from, &to) {
                        println!("cargo:warning=Failed to copy {:?} to {:?}: {}", from, to, e);
                    } else {
                        println!("cargo:warning=Copied {:?} to {:?}", from, to);
                    }
                }
            }

            println!("cargo:rustc-link-search=native={}", torch_lib.display());
            link_all_libs_in_dir(&torch_lib, "lib");
        }

        // O CMake gera os arquivos em cpp/build/Release/Release no Windows
        let lib_dir = build_dir_abs.join("Release").join("Release");
        
        if use_tensorflow {
            let tf_lib = tf_path.join("lib");
            println!("cargo:rustc-link-search=native={}", tf_lib.display());
            link_all_libs_in_dir(&tf_lib, "lib");
        }

        println!("cargo:rustc-link-lib=dylib=ai_copper_cpp");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());

        // Calcula o target/debug do projeto consumidor a partir do OUT_DIR
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let target_dir = out_dir
            .ancestors()
            .nth(4) // Subir 4 níveis para alcançar o diretório raiz do projeto
            .unwrap()
            .join("debug");

        // Copia todos os arquivos dos diretórios relevantes
        copy_all_files(&lib_dir, &target_dir);
        
        if use_libtorch {
            let torch_lib = torch_path.join("lib");
            copy_all_files(&torch_lib, &target_dir);
            
            // Copia explícita de DLLs críticas do LibTorch
            let torch_dll_path = torch_lib.join("torch.dll");
            copy_specific_lib(&torch_dll_path, &target_dir);
            
            let torch_cpu_dll = torch_lib.join("torch_cpu.dll");
            copy_specific_lib(&torch_cpu_dll, &target_dir);
            
            let c10_dll = torch_lib.join("c10.dll");
            copy_specific_lib(&c10_dll, &target_dir);
        }
        
        if use_tensorflow {
            let tf_lib = tf_path.join("lib");
            copy_all_files(&tf_lib, &target_dir);
            
            // Copia explícita de DLLs do TensorFlow
            let tf_dll_path = tf_lib.join("tensorflow.dll");
            copy_specific_lib(&tf_dll_path, &target_dir);
        }

        // Verificação detalhada das DLLs críticas
        let ai_copper_cpp_path = target_dir.join("ai_copper_cpp.dll");
        if !ai_copper_cpp_path.exists() {
            println!("cargo:warning=ai_copper_cpp.dll not found in {:?}", target_dir);
        } else {
            println!("cargo:warning=ai_copper_cpp.dll found in {:?}", target_dir);
        }
        
        if use_libtorch {
            let torch_path_check = target_dir.join("torch.dll");
            if !torch_path_check.exists() {
                println!("cargo:warning=torch.dll not found in {:?}", target_dir);
            } else {
                println!("cargo:warning=torch.dll found in {:?}", target_dir);
            }
        }
        
        if use_tensorflow {
            let tf_path_check = target_dir.join("tensorflow.dll");
            if !tf_path_check.exists() {
                println!("cargo:warning=tensorflow.dll not found in {:?}", target_dir);
            } else {
                println!("cargo:warning=tensorflow.dll found in {:?}", target_dir);
            }
        }
    } else {
        if use_libtorch {
            let torch_lib = torch_path.join("lib");
            println!("cargo:rustc-link-search=native={}", torch_lib.display());
            link_all_libs_in_dir(&torch_lib, "so");
        }
        
        if use_tensorflow {
            let tf_lib = tf_path.join("lib");
            println!("cargo:rustc-link-search=native={}", tf_lib.display());
            link_all_libs_in_dir(&tf_lib, "so");
        }

        println!("cargo:rustc-link-lib=dylib=ai_copper");
        println!("cargo:rustc-link-search=native={}", build_dir_abs.display());

        // Calcula o target/debug do projeto consumidor a partir do OUT_DIR
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let target_dir = out_dir
            .ancestors()
            .nth(3) // Subir 3 níveis para alcançar o diretório raiz do projeto
            .unwrap()
            .join("debug");

        // Copia todos os arquivos dos diretórios relevantes
        copy_all_files(&build_dir_abs, &target_dir);
        
        if use_libtorch {
            let torch_lib = torch_path.join("lib");
            copy_all_files(&torch_lib, &target_dir);
            
            // Copia explícita das bibliotecas compartilhadas críticas do LibTorch
            let torch_so_path = torch_lib.join("libtorch.so");
            copy_specific_lib(&torch_so_path, &target_dir);
            
            let torch_cpu_so = torch_lib.join("libtorch_cpu.so");
            copy_specific_lib(&torch_cpu_so, &target_dir);
            
            let c10_so = torch_lib.join("libc10.so");
            copy_specific_lib(&c10_so, &target_dir);
        }
        
        if use_tensorflow {
            let tf_lib = tf_path.join("lib");
            copy_all_files(&tf_lib, &target_dir);
            
            // Copia explícita das bibliotecas do TensorFlow
            let tf_so_path = tf_lib.join("libtensorflow.so");
            copy_specific_lib(&tf_so_path, &target_dir);
        }

        // Verificação detalhada das bibliotecas críticas
        let ai_copper_path = target_dir.join("libai_copper.so");
        if !ai_copper_path.exists() {
            println!("cargo:warning=libai_copper.so not found in {:?}", target_dir);
        } else {
            println!("cargo:warning=libai_copper.so found in {:?}", target_dir);
        }
        
        if use_libtorch {
            let torch_path_check = target_dir.join("libtorch.so");
            if !torch_path_check.exists() {
                println!("cargo:warning=libtorch.so not found in {:?}", target_dir);
            } else {
                println!("cargo:warning=libtorch.so found in {:?}", target_dir);
            }
        }
        
        if use_tensorflow {
            let tf_path_check = target_dir.join("libtensorflow.so");
            if !tf_path_check.exists() {
                println!("cargo:warning=libtensorflow.so not found in {:?}", target_dir);
            } else {
                println!("cargo:warning=libtensorflow.so found in {:?}", target_dir);
            }
        }

        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", target_dir.display());
    }
}