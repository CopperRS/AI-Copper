# 🔄 Fluxo de Instalação Automática

## 📋 Resumo Executivo

**Em um PC zerado**, quando você executar `cargo build --features libtorch,tensorflow`:

✅ **Baixa** automaticamente LibTorch e TensorFlow  
✅ **Instala** em `C:\libtorch` e `C:\libtensorflow` (sem subpastas duplicadas)  
✅ **Configura** todas as variáveis de ambiente automaticamente  
✅ **Atualiza** o PATH do sistema  
✅ **Pronto** para usar! Sem configuração manual necessária  

---

## 🎯 Estrutura Final (Garantida)

### ✅ Correto (Como será instalado)

```
C:\
├── libtorch/                    ← Instalação do LibTorch
│   ├── bin/
│   ├── include/
│   │   └── torch/
│   │       ├── csrc/
│   │       ├── script.h
│   │       └── torch.h
│   ├── lib/
│   │   ├── torch.dll
│   │   ├── torch_cpu.dll
│   │   ├── c10.dll
│   │   └── *.lib
│   └── share/
│       └── cmake/
│           └── Torch/
│
└── libtensorflow/               ← Instalação do TensorFlow
    ├── include/
    │   └── tensorflow/
    │       └── c/
    │           └── c_api.h
    └── lib/
        ├── tensorflow.dll
        └── tensorflow.lib
```

### ❌ Incorreto (NÃO será criado)

```
C:\
├── libtorch/
│   └── libtorch/               ← SUBPASTA DUPLICADA (NÃO!)
│       ├── bin/
│       └── lib/
│
└── libtensorflow/
    └── libtensorflow/          ← SUBPASTA DUPLICADA (NÃO!)
        └── lib/
```

---

## 🚀 Processo Passo a Passo

### 1️⃣ **Detecção Inicial**

```rust
// O sistema verifica se já existe
if system_torch_path.exists() {
    // Usa a instalação existente
    println!("✓ LibTorch encontrado em C:\libtorch");
} else {
    // Inicia download e instalação
    println!("⬇ Baixando LibTorch...");
}
```

### 2️⃣ **Download**

```
⬇ Baixando LibTorch (2.1.0) - ~1.2GB
  URL: https://download.pytorch.org/libtorch/cpu/...
  Destino temporário: C:\Users\{User}\AppData\Local\Temp\ai_copper_downloads\

⬇ Baixando TensorFlow (2.10.0) - ~200MB
  URL: https://storage.googleapis.com/tensorflow/...
  Destino temporário: C:\Users\{User}\AppData\Local\Temp\ai_copper_downloads\
```

### 3️⃣ **Extração Inteligente**

```rust
// Extrai em pasta temporária
extract_zip(&zip_file, &temp_extract)?;

// Detecta se há subpasta duplicada
let extracted_libtorch = temp_extract.join("libtorch");
if extracted_libtorch.exists() {
    // Remove a duplicação movendo o conteúdo interno
    move_directory(&extracted_libtorch, &final_dir)?;
} else {
    // Usa diretamente
    move_directory(&temp_extract, &final_dir)?;
}
```

**Resultado**: `C:\libtorch` conterá **diretamente** `bin/`, `lib/`, `include/` - sem duplicação!

### 4️⃣ **Limpeza**

```rust
// Remove arquivos temporários
fs::remove_file(&zip_file)?;
fs::remove_dir_all(&temp_extract)?;
```

### 5️⃣ **Configuração Automática de Variáveis**

Para **LibTorch**:
```
✓ TORCH_HOME = C:\libtorch
✓ LD_LIBRARY_PATH = C:\libtorch\lib
✓ LIBRARY_PATH = C:\libtorch\lib
✓ CPATH = C:\libtorch\include
✓ CMAKE_PREFIX_PATH = C:\libtorch
✓ PATH += C:\libtorch\lib;C:\libtorch\bin
```

Para **TensorFlow**:
```
✓ TENSORFLOW_ROOT = C:\libtensorflow
✓ LD_LIBRARY_PATH = C:\libtensorflow\lib
✓ LIBRARY_PATH = C:\libtensorflow\lib
✓ CPATH = C:\libtensorflow\include
✓ PATH += C:\libtensorflow\lib;C:\libtensorflow\bin
```

---

## 📊 Cenários de Teste

### Cenário 1: PC Completamente Zerado

```powershell
# Passo 1: Clone o repositório
git clone https://github.com/CopperRS/AI-Copper.git
cd AI-Copper

# Passo 2: Build (primeira vez)
cargo build --features libtorch,tensorflow

# O que acontece:
# ⬇ Baixando LibTorch 2.1.0...
# ⬇ Baixando TensorFlow 2.10.0...
# 📦 Extraindo para C:\libtorch...
# 📦 Extraindo para C:\libtensorflow...
# ⚙️ Configurando variáveis de ambiente...
# ✅ Compilação concluída!

# Passo 3: Verificar
.\test-auto-install.ps1

# Resultado esperado:
# ✓ C:\libtorch existe
# ✓ C:\libtorch\lib existe
# ✓ C:\libtorch\include existe
# ✓ Todas as variáveis configuradas
# ✓ PATH atualizado
```

**Tempo estimado**: 10-20 minutos (dependendo da velocidade da internet)

### Cenário 2: Bibliotecas Já Instaladas

```powershell
# Build subsequente
cargo build --features libtorch,tensorflow

# O que acontece:
# ✓ LibTorch encontrado em C:\libtorch
# ✓ TensorFlow encontrado em C:\libtensorflow
# ⚙️ Verificando variáveis de ambiente...
# ✅ Compilação concluída!
```

**Tempo estimado**: 2-5 minutos (apenas compilação)

### Cenário 3: Reinstalação Forçada

```powershell
# Remover instalações existentes
Remove-Item -Recurse -Force C:\libtorch
Remove-Item -Recurse -Force C:\libtensorflow

# Build novamente
cargo clean
cargo build --features libtorch,tensorflow

# Processo completo é executado novamente
```

---

## 🔍 Verificação da Estrutura

### Comando para Verificar a Estrutura

```powershell
# Verificar que NÃO há subpastas duplicadas
Write-Host "=== Estrutura LibTorch ===" -ForegroundColor Cyan
Get-ChildItem C:\libtorch -Directory | Select-Object Name

# Saída esperada:
# Name
# ----
# bin
# include
# lib
# share

# NÃO deve aparecer outra pasta "libtorch" aqui!

Write-Host "`n=== Estrutura TensorFlow ===" -ForegroundColor Cyan
Get-ChildItem C:\libtensorflow -Directory | Select-Object Name

# Saída esperada:
# Name
# ----
# include
# lib

# NÃO deve aparecer outra pasta "libtensorflow" aqui!
```

### Verificar DLLs Diretamente Acessíveis

```powershell
# LibTorch - DLLs devem estar em C:\libtorch\lib
Test-Path C:\libtorch\lib\torch.dll
Test-Path C:\libtorch\lib\torch_cpu.dll
Test-Path C:\libtorch\lib\c10.dll

# TensorFlow - DLLs devem estar em C:\libtensorflow\lib
Test-Path C:\libtensorflow\lib\tensorflow.dll

# Todos devem retornar: True
```

---

## ✅ Garantias do Sistema

### 1. **Sem Subpastas Duplicadas**

```rust
// Código que garante isso:
let extracted_libtorch = temp_extract.join("libtorch");
if extracted_libtorch.exists() {
    // Se o ZIP contém libtorch/libtorch/..., pegamos apenas o conteúdo interno
    move_directory(&extracted_libtorch, &final_dir)?;
}
```

### 2. **Instalação Única**

```rust
// Só baixa se não existir
if lib_dir.exists() {
    println!("cargo:warning={} já existe em: {:?}", lib_name, lib_dir);
    set_environment_variables(lib_name, lib_dir)?;
    return Ok(lib_dir.to_path_buf());
}
```

### 3. **Limpeza Automática**

```rust
// Remove temporários após instalação
let _ = fs::remove_file(&zip_file);
let _ = fs::remove_dir_all(&temp_extract);
```

### 4. **Variáveis Persistentes**

```rust
// Configuradas no registro do Windows (User)
[Environment]::SetEnvironmentVariable('TORCH_HOME', 'C:\libtorch', 'User')
// Persistem entre reinicializações
```

---

## 🎓 Exemplo de Uso Após Instalação

```rust
// Seu código Rust
use ai_copper::prelude::*;

fn main() {
    // LibTorch funciona imediatamente
    let tensor = Tensor::randn(&[2, 3], Backend::LibTorch);
    println!("{:?}", tensor);
    
    // TensorFlow funciona imediatamente
    let tensor2 = Tensor::zeros(&[3, 3], Backend::TensorFlow);
    println!("{:?}", tensor2);
}
```

**Nenhuma configuração manual necessária!** 🎉

---

## 📞 Suporte

Se algo não funcionar conforme esperado:

1. **Execute o teste**:
   ```powershell
   .\test-auto-install.ps1
   ```

2. **Verifique a estrutura**:
   ```powershell
   Get-ChildItem C:\libtorch -Recurse -Depth 1
   Get-ChildItem C:\libtensorflow -Recurse -Depth 1
   ```

3. **Reconfigure variáveis**:
   ```powershell
   .\setup-env.ps1
   ```

4. **Reinicie o terminal**

---

## 🏆 Resumo Final

| Aspecto | Status |
|---------|--------|
| Download automático | ✅ Sim |
| Instalação em C:\ | ✅ Sim |
| Sem subpastas duplicadas | ✅ Garantido |
| Variáveis de ambiente | ✅ Automático |
| PATH atualizado | ✅ Automático |
| Configuração manual | ❌ Não necessária |
| Funciona em PC zerado | ✅ Sim |

**Conclusão**: Basta executar `cargo build` e tudo funcionará! 🚀
