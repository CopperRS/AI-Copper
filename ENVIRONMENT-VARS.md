# 🔧 Referência de Variáveis de Ambiente

Este documento lista todas as variáveis de ambiente configuradas automaticamente pelo AI-Copper.

## 📋 Visão Geral

O sistema configura automaticamente **todas as variáveis necessárias** para compilar e executar aplicações com LibTorch e TensorFlow, conforme as especificações das APIs C/C++.

## 🔥 LibTorch (PyTorch C++ API)

### Variáveis Configuradas

| Variável | Obrigatória? | Descrição | Valor (Windows) | Valor (Linux) |
|----------|--------------|-----------|-----------------|---------------|
| `TORCH_HOME` | ✅ Sim | Raiz da instalação do LibTorch | `C:\libtorch` | `/opt/libtorch` |
| `LD_LIBRARY_PATH` | ✅ Sim | Caminho das bibliotecas dinâmicas (`.so` no Linux, `.dll` no Windows) | `C:\libtorch\lib` | `/opt/libtorch/lib` |
| `LIBRARY_PATH` | ✅ Sim | Caminho de linkagem para compilação | `C:\libtorch\lib` | `/opt/libtorch/lib` |
| `CPATH` | ✅ Sim | Caminho dos headers (`.h`) | `C:\libtorch\include` | `/opt/libtorch/include` |
| `CMAKE_PREFIX_PATH` | ✅ Sim (ao compilar) | Para CMake encontrar automaticamente o pacote Torch | `C:\libtorch` | `/opt/libtorch` |
| `PATH` | ✅ Sim | Caminhos das DLLs e executáveis | `C:\libtorch\lib;C:\libtorch\bin` | `/opt/libtorch/lib:/opt/libtorch/bin` |

### Uso no Código

```cpp
// CMakeLists.txt
find_package(Torch REQUIRED)
# CMAKE_PREFIX_PATH será usado automaticamente

target_link_libraries(your_app ${TORCH_LIBRARIES})
```

```rust
// build.rs
let torch_path = env::var("TORCH_HOME").expect("TORCH_HOME not set");
println!("cargo:rustc-link-search=native={}/lib", torch_path);
```

## 🧠 TensorFlow (C API)

### Variáveis Configuradas

| Variável | Obrigatória? | Descrição | Valor (Windows) | Valor (Linux) |
|----------|--------------|-----------|-----------------|---------------|
| `TENSORFLOW_ROOT` | ✅ Sim | Raiz da instalação do TensorFlow | `C:\libtensorflow` | `/opt/libtensorflow` |
| `LD_LIBRARY_PATH` | ✅ Sim | Caminho das bibliotecas dinâmicas (`.so` no Linux, `.dll` no Windows) | `C:\libtensorflow\lib` | `/opt/libtensorflow/lib` |
| `LIBRARY_PATH` | ✅ Sim | Caminho de linkagem para compilação | `C:\libtensorflow\lib` | `/opt/libtensorflow/lib` |
| `CPATH` | ✅ Sim | Caminho dos headers (`.h`) | `C:\libtensorflow\include` | `/opt/libtensorflow/include` |
| `PATH` | ✅ Sim | Caminhos das DLLs e executáveis | `C:\libtensorflow\lib;C:\libtensorflow\bin` | `/opt/libtensorflow/lib:/opt/libtensorflow/bin` |

### Uso no Código

```cpp
// CMakeLists.txt
include_directories($ENV{TENSORFLOW_ROOT}/include)
link_directories($ENV{TENSORFLOW_ROOT}/lib)
target_link_libraries(your_app tensorflow)
```

```rust
// build.rs
let tf_path = env::var("TENSORFLOW_ROOT").expect("TENSORFLOW_ROOT not set");
println!("cargo:rustc-link-search=native={}/lib", tf_path);
```

## 🚀 Verificação

### Windows (PowerShell)

```powershell
# Verificar TODAS as variáveis de uma vez
Write-Host "=== LibTorch ===" -ForegroundColor Yellow
Write-Host "TORCH_HOME: $env:TORCH_HOME"
Write-Host "LD_LIBRARY_PATH: $env:LD_LIBRARY_PATH"
Write-Host "LIBRARY_PATH: $env:LIBRARY_PATH"
Write-Host "CPATH: $env:CPATH"
Write-Host "CMAKE_PREFIX_PATH: $env:CMAKE_PREFIX_PATH"

Write-Host "`n=== TensorFlow ===" -ForegroundColor Yellow
Write-Host "TENSORFLOW_ROOT: $env:TENSORFLOW_ROOT"
Write-Host "LD_LIBRARY_PATH: $env:LD_LIBRARY_PATH"
Write-Host "LIBRARY_PATH: $env:LIBRARY_PATH"
Write-Host "CPATH: $env:CPATH"

Write-Host "`n=== PATH ===" -ForegroundColor Yellow
$env:PATH -split ';' | Select-String "libtorch|tensorflow"
```

Ou use o script de teste:

```powershell
.\test-auto-install.ps1
```

### Linux (Bash)

```bash
# Verificar TODAS as variáveis de uma vez
echo "=== LibTorch ==="
echo "TORCH_HOME: $TORCH_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "LIBRARY_PATH: $LIBRARY_PATH"
echo "CPATH: $CPATH"
echo "CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"

echo -e "\n=== TensorFlow ==="
echo "TENSORFLOW_ROOT: $TENSORFLOW_ROOT"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "LIBRARY_PATH: $LIBRARY_PATH"
echo "CPATH: $CPATH"

echo -e "\n=== PATH ==="
echo $PATH | tr ':' '\n' | grep -E 'libtorch|tensorflow'
```

## 🔄 Reconfiguração

Se as variáveis não estiverem configuradas ou precisarem ser atualizadas:

### Automático (durante o build)

```bash
cargo clean
cargo build --features libtorch,tensorflow
```

### Manual (Windows)

```powershell
# Execute o script de configuração
.\setup-env.ps1

# Reinicie o terminal
```

### Manual (Linux)

Adicione ao `~/.bashrc` ou `~/.zshrc`:

```bash
# LibTorch
export TORCH_HOME=/opt/libtorch
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/opt/libtorch/lib:$LIBRARY_PATH
export CPATH=/opt/libtorch/include:$CPATH
export CMAKE_PREFIX_PATH=/opt/libtorch

# TensorFlow
export TENSORFLOW_ROOT=/opt/libtensorflow
export LD_LIBRARY_PATH=/opt/libtensorflow/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/opt/libtensorflow/lib:$LIBRARY_PATH
export CPATH=/opt/libtensorflow/include:$CPATH
```

Depois recarregue:

```bash
source ~/.bashrc
```

## ⚠️ Troubleshooting

### Erro: "DLL não encontrada" (Windows)

**Causa**: PATH não contém os diretórios das DLLs

**Solução**:
```powershell
.\setup-env.ps1
# Reinicie o terminal
```

### Erro: "Shared library not found" (Linux)

**Causa**: LD_LIBRARY_PATH não configurado

**Solução**:
```bash
export LD_LIBRARY_PATH=/opt/libtorch/lib:/opt/libtensorflow/lib:$LD_LIBRARY_PATH
```

### Erro: "cmake could not find Torch"

**Causa**: CMAKE_PREFIX_PATH não configurado

**Solução**:
```bash
export CMAKE_PREFIX_PATH=/opt/libtorch  # Linux
$env:CMAKE_PREFIX_PATH = "C:\libtorch"  # Windows
```

### Erro: "header file not found"

**Causa**: CPATH não configurado

**Solução**:
```bash
export CPATH=/opt/libtorch/include:/opt/libtensorflow/include:$CPATH  # Linux
$env:CPATH = "C:\libtorch\include;C:\libtensorflow\include"           # Windows
```

## 📚 Referências Oficiais

### LibTorch
- [Installing C++ Distributions of PyTorch](https://pytorch.org/cppdocs/installing.html)
- [PyTorch C++ API Documentation](https://pytorch.org/cppdocs/)

### TensorFlow
- [Install TensorFlow for C](https://www.tensorflow.org/install/lang_c)
- [TensorFlow C API Reference](https://www.tensorflow.org/api_docs/c)

## 🎯 Resumo Rápido

**Após a instalação automática, estas variáveis estarão configuradas:**

✅ `TORCH_HOME` e `TENSORFLOW_ROOT` → Raízes das instalações  
✅ `LD_LIBRARY_PATH` e `LIBRARY_PATH` → Para encontrar bibliotecas  
✅ `CPATH` → Para encontrar headers  
✅ `CMAKE_PREFIX_PATH` → Para CMake  
✅ `PATH` → Para DLLs e executáveis  

**Reinicie o terminal** após a primeira instalação para que tudo funcione corretamente!
