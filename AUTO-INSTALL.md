# Instalação Automática de Dependências

Este documento descreve o sistema de instalação automática das bibliotecas LibTorch e TensorFlow.

## 📦 Como Funciona

O AI-Copper agora baixa e instala automaticamente as dependências necessárias durante o build:

### Windows
- **LibTorch**: Instalado em `C:\libtorch`
- **TensorFlow**: Instalado em `C:\libtensorflow`

### Linux
- **LibTorch**: Instalado em `/opt/libtorch`
- **TensorFlow**: Instalado em `/opt/libtensorflow`

## 🚀 Processo de Instalação

### 1. Build Automático

Quando você executa `cargo build`, o script `build.rs`:

1. **Verifica** se as bibliotecas já existem nos caminhos do sistema
2. Se não existirem:
   - **Baixa** os arquivos compactados para uma pasta temporária
   - **Extrai** o conteúdo
   - **Move** para o diretório raiz do sistema (sem subpastas duplicadas)
   - **Remove** arquivos temporários
3. **Configura** automaticamente as variáveis de ambiente:
   - `LIBTORCH` → Caminho da instalação do LibTorch
   - `TENSORFLOW_ROOT` → Caminho da instalação do TensorFlow
   - Atualiza o `PATH` com os diretórios `lib` e `bin`

### 2. Configuração Manual (Opcional)

Se preferir configurar manualmente as variáveis de ambiente, execute:

```powershell
.\setup-env.ps1
```

**Nota**: Execute como Administrador para configurar variáveis do sistema (recomendado).

## 🔧 Variáveis de Ambiente

### LibTorch
O sistema configura automaticamente as seguintes variáveis:

| Variável | Descrição | Exemplo (Windows) |
|----------|-----------|-------------------|
| `TORCH_HOME` | Diretório raiz da instalação | `C:\libtorch` |
| `LD_LIBRARY_PATH` | Caminho das bibliotecas dinâmicas (.so/.dll) | `C:\libtorch\lib` |
| `LIBRARY_PATH` | Caminho de linkagem para compilação | `C:\libtorch\lib` |
| `CPATH` | Caminho dos headers (.h) | `C:\libtorch\include` |
| `CMAKE_PREFIX_PATH` | Caminho para CMake encontrar o pacote | `C:\libtorch` |
| `PATH` | Caminhos das DLLs/executáveis | Inclui `lib` e `bin` |

### TensorFlow
O sistema configura automaticamente as seguintes variáveis:

| Variável | Descrição | Exemplo (Windows) |
|----------|-----------|-------------------|
| `TENSORFLOW_ROOT` | Diretório raiz da instalação | `C:\libtensorflow` |
| `LD_LIBRARY_PATH` | Caminho das bibliotecas dinâmicas (.so/.dll) | `C:\libtensorflow\lib` |
| `LIBRARY_PATH` | Caminho de linkagem para compilação | `C:\libtensorflow\lib` |
| `CPATH` | Caminho dos headers (.h) | `C:\libtensorflow\include` |
| `PATH` | Caminhos das DLLs/executáveis | Inclui `lib` e `bin` |

## ✅ Verificando a Instalação

### Windows (PowerShell)
```powershell
# Verificar variáveis de ambiente LibTorch
$env:TORCH_HOME
$env:LD_LIBRARY_PATH
$env:LIBRARY_PATH
$env:CPATH
$env:CMAKE_PREFIX_PATH

# Verificar variáveis de ambiente TensorFlow
$env:TENSORFLOW_ROOT
$env:LD_LIBRARY_PATH
$env:LIBRARY_PATH
$env:CPATH

# Verificar PATH
$env:PATH

# Verificar se as pastas existem
Test-Path C:\libtorch
Test-Path C:\libtensorflow

# Listar DLLs instaladas
Get-ChildItem C:\libtorch\lib\*.dll
Get-ChildItem C:\libtensorflow\lib\*.dll

# Executar teste completo
.\test-auto-install.ps1
```

### Linux (Bash)
```bash
# Verificar variáveis de ambiente LibTorch
echo $TORCH_HOME
echo $LD_LIBRARY_PATH
echo $LIBRARY_PATH
echo $CPATH
echo $CMAKE_PREFIX_PATH

# Verificar variáveis de ambiente TensorFlow
echo $TENSORFLOW_ROOT
echo $LD_LIBRARY_PATH
echo $LIBRARY_PATH
echo $CPATH

# Verificar se as pastas existem
ls -la /opt/libtorch
ls -la /opt/libtensorflow

# Listar bibliotecas instaladas
ls -la /opt/libtorch/lib/*.so
ls -la /opt/libtensorflow/lib/*.so
```

## 🔄 Reinstalação

Se precisar reinstalar as bibliotecas:

1. **Delete** as pastas:
   - Windows: `C:\libtorch` e `C:\libtensorflow`
   - Linux: `/opt/libtorch` e `/opt/libtensorflow`

2. **Execute** novamente:
   ```bash
   cargo clean
   cargo build --features libtorch,tensorflow
   ```

## 📝 URLs de Download

### LibTorch
- **Windows**: https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.1.0%2Bcpu.zip
- **Linux**: https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip

### TensorFlow
- **Windows**: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-2.10.0.zip
- **Linux**: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.10.0.tar.gz

## ⚠️ Requisitos

### Windows
- **Permissões**: Pode requerer privilégios de administrador para escrever em `C:\`
- **PowerShell**: Versão 5.1 ou superior
- **Espaço**: ~2GB para LibTorch, ~200MB para TensorFlow

### Linux
- **Permissões**: Pode requerer `sudo` para escrever em `/opt/`
- **Espaço**: ~2GB para LibTorch, ~200MB para TensorFlow

## 🛠️ Solução de Problemas

### Erro de Permissão

**Windows**:
```powershell
# Execute o PowerShell como Administrador
cargo build --features libtorch,tensorflow
```

**Linux**:
```bash
# Use sudo para permitir escrita em /opt/
sudo cargo build --features libtorch,tensorflow
```

### DLLs não encontradas (Windows)

Se você receber erros sobre DLLs faltando:

1. Verifique o PATH:
   ```powershell
   $env:PATH -split ';' | Select-String "libtorch|tensorflow"
   ```

2. Execute o script de configuração:
   ```powershell
   .\setup-env.ps1
   ```

3. **Reinicie** o terminal ou IDE

### Bibliotecas não encontradas (Linux)

Se você receber erros sobre bibliotecas compartilhadas:

1. Verifique o LD_LIBRARY_PATH:
   ```bash
   echo $LD_LIBRARY_PATH
   ```

2. Adicione ao perfil do shell (~/.bashrc ou ~/.zshrc):
   ```bash
   export LD_LIBRARY_PATH=/opt/libtorch/lib:/opt/libtensorflow/lib:$LD_LIBRARY_PATH
   ```

3. Recarregue:
   ```bash
   source ~/.bashrc
   ```

## 🎯 Integração com CI/CD

Para ambientes de integração contínua:

```yaml
# Exemplo para GitHub Actions
steps:
  - name: Cache dependencies
    uses: actions/cache@v3
    with:
      path: |
        C:\libtorch
        C:\libtensorflow
      key: ${{ runner.os }}-ai-copper-deps-v1

  - name: Build
    run: cargo build --release --features libtorch,tensorflow
```

## 📚 Referências

- [PyTorch C++ API](https://pytorch.org/cppdocs/)
- [TensorFlow C API](https://www.tensorflow.org/install/lang_c)
- [AI-Copper Documentation](./docs/index.md)
