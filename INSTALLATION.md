# 📥 Guia de Instalação - AI Copper

## ✨ Instalação Automática (Recomendado)

A partir da versão 0.1.3, o **AI Copper** faz o download automático das bibliotecas necessárias!

### Uso Básico

Adicione ao seu `Cargo.toml`:

```toml
[dependencies]
ai_copper = "0.1.3"
```

**Pronto!** Na primeira compilação, a biblioteca irá:
1. ✅ Baixar automaticamente o LibTorch 2.1.0
2. ✅ Baixar automaticamente o TensorFlow 2.10.0
3. ✅ Compilar e configurar tudo para você

### Escolhendo Backends Específicos

Se você quiser usar apenas um dos backends, pode desabilitar o outro:

```toml
[dependencies]
# Apenas LibTorch
ai_copper = { version = "0.1.3", default-features = false, features = ["libtorch"] }

# Apenas TensorFlow
ai_copper = { version = "0.1.3", default-features = false, features = ["tensorflow"] }

# Ambos (padrão)
ai_copper = "0.1.3"
```

## 🔧 Instalação Manual (Opcional)

Se você já tem as bibliotecas instaladas localmente ou prefere usar versões específicas, pode definir variáveis de ambiente:

### Windows

```powershell
# LibTorch
$env:LIBTORCH = "C:\path\to\libtorch"

# TensorFlow
$env:TENSORFLOW_ROOT = "C:\path\to\tensorflow"
```

### Linux/Mac

```bash
# LibTorch
export LIBTORCH=/path/to/libtorch

# TensorFlow
export TENSORFLOW_ROOT=/path/to/tensorflow
```

## 📋 Requisitos

### Windows
- Visual Studio 2022 (ou 2019) com C++ tools
- CMake 3.15+
- Conexão com internet (para download automático)

### Linux
- GCC 7.0+
- CMake 3.15+
- Conexão com internet (para download automático)

## 🚀 Primeiro Uso

Após adicionar ao `Cargo.toml`, compile seu projeto:

```bash
cargo build
```

Na primeira vez, o processo pode demorar alguns minutos:
- ⏱️ Download do LibTorch: ~200 MB
- ⏱️ Download do TensorFlow: ~60 MB
- ⏱️ Compilação da biblioteca C++

As bibliotecas são salvas em `deps/` e reutilizadas nas próximas compilações!

## 💡 Exemplo Rápido

```rust
use ai_copper::Tensor;

fn main() {
    // Cria um tensor 3x3 com valores aleatórios
    let tensor = Tensor::rand(3, 3);
    
    // Operações matemáticas
    let result = tensor.sin().exp();
    
    // Estatísticas
    println!("Média: {}", result.mean());
    println!("Desvio Padrão: {}", result.std());
}
```

## 🔍 Troubleshooting

### Erro: "Failed to download"
- Verifique sua conexão com internet
- Pode ser necessário configurar proxy se estiver em rede corporativa

### Erro: "CMake not found"
- Instale o CMake: https://cmake.org/download/
- Adicione ao PATH do sistema

### Erro: "Visual Studio not found" (Windows)
- Instale o Visual Studio 2022 com "Desktop development with C++"
- Ou use o Visual Studio Build Tools

### Downloads Muito Lentos
Se os downloads estiverem muito lentos, você pode:
1. Baixar manualmente:
   - [LibTorch Windows](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.1.0%2Bcpu.zip)
   - [LibTorch Linux](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip)
   - [TensorFlow Windows](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-2.10.0.zip)
   - [TensorFlow Linux](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.10.0.tar.gz)

2. Extrair para `deps/libtorch` e `deps/tensorflow` na raiz do projeto

3. Compilar novamente

## 📦 Cache das Bibliotecas

As bibliotecas baixadas ficam em:
```
seu_projeto/
  └── deps/
      ├── libtorch/
      └── tensorflow/
```

Para limpar o cache e forçar novo download:
```bash
# Windows
Remove-Item -Recurse -Force deps

# Linux/Mac
rm -rf deps
```

## 🌐 Versões Suportadas

| Biblioteca | Versão | Platform |
|-----------|--------|----------|
| LibTorch  | 2.1.0  | Windows, Linux |
| TensorFlow| 2.10.0 | Windows, Linux |

## 📝 Notas

- Os downloads são feitos apenas uma vez
- As bibliotecas são reutilizadas entre compilações
- O diretório `deps/` está no `.gitignore` (não é versionado)
- Você pode compartilhar seu projeto sem incluir as bibliotecas grandes
