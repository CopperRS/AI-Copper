#  AI Copper

**AI Copper** é uma biblioteca Rust unificada que combina as capacidades do **LibTorch** (PyTorch C++) e **TensorFlow C API** em uma interface única. 

Crie modelos de machine learning e deep learning usando o melhor das duas bibliotecas!

## 🎨 Características

### 🔥 Dual Backend Support
- **LibTorch Backend**: Acesso completo às funcionalidades do PyTorch em C++
- **TensorFlow Backend**: Suporte nativo para TensorFlow C API
- **API Unificada**: Troque entre backends sem alterar seu código

### 🎯 Funcionalidades Principais

#### Tensor Operations
- ✅ Criação de tensores (zeros, ones, rand, randn, eye, from_values)
- ✅ Operações aritméticas (+, -, *, /)
- ✅ Operações matriciais (matmul, transpose)
- ✅ Estatísticas (sum, mean, max, min, std, var, argmax, argmin)
- ✅ Funções matemáticas (sin, cos, exp, log, sqrt, abs, pow)
- ✅ Funções de ativação (relu, sigmoid, tanh)
- ✅ Transformações (map, reshape, zeros_like, ones_like)
- ✅ Conversão entre backends

#### Neural Networks (LibTorch)
- ✅ Camadas Linear
- ✅ Funções de perda (MSE Loss, Cross Entropy Loss)
- ✅ Funções de ativação (ReLU, Sigmoid, Tanh)
- ✅ Otimizadores (SGD, Adam)
- ✅ Backpropagation automática
- ✅ Treinamento de modelos

#### TensorFlow Integration
- ✅ Carregar modelos SavedModel
- ✅ Executar inferência
- ✅ Manipulação de tensores multi-dimensionais
- ✅ Operações tensoriais básicas

## 📦 Instalação

####   Adicione ao seu `Cargo.toml`:



```toml
[dependencies]
ai_copper = { git = "" }
```



### Build do Projeto
> [!NOTE]
> ❗❗ IMPORTANTE: Em toda Build, o terminal deve estar em modo de adminstrador. ❗❗

> [!NOTE]
> O primeiro build pode demorar, pois as bibliotecas são baixadas e instaladas automaticamente durante o primeiro build.
> - **Windows**: `C:\libtorch` & `C:\libtensorflow`
> - **Linux**: `/libtorch` & `/libtensorflow`
> - **MacOS**; `/libtorch` & `/libtensorflow`
>
> As variáveis de ambiente são configuradas automaticamente. Para mais detalhes, consulte [AUTO-INSTALL.md](AUTO-INSTALL.md). 

```bash
# Apenas LibTorch
cargo build --features libtorch
```
```bash
# Apenas TensorFlow
cargo build --features tensorflow
```
```bash
# Com ambos os backends
cargo build
```

Para instruções detalhadas de instalação manual, veja o [Guia de Instalação](INSTALLATION.md).

## 🦀 Utilização na Copper

- **[Instalação da Copper Lang](https://github.com/liy77/copper-lang.git)**
- Crie a pasta do projeto
- Crie o arquivo .crs
- Crie o arquivo Cargo.toml
- A estrutura do projeto deve ficar assim:

```
📦 nome-do-projeto
├─ example.crs
└─ cargo.toml
```

- Dentro do cargo.toml, cole o seguinte código:

```toml
[package]
name = "nome-da-pasta"
version = "0.1.0"
edition = "2024"

[dependencies]
ai_copper = { git = "https://github.com/CopperRS/AI-Copper.git", branch = "main" }

[[bin]]
name = "example"
path = "example.crs"
```

```bash
# Para Buildar e Rodar
  cforge run example.crs
```

## 📚 Documentação

- **[INSTALLATION.md](INSTALLATION.md)** - Guia completo de instalação e troubleshooting
- **[QUICKSTART.md](QUICKSTART.md)** - Exemplos práticos e início rápido
- **[Documentação da API](docs/index.md)** - Referência completa da API
- **[Exemplos](examples.md)** - Exemplos de uso da biblioteca no Rust
- **[Copper Usage](examples/copper.crs)** - Exemplos de uso da biblioteca no Copper


# Para Contribuidores
> [!NOTE]
> Linux & MacOS ainda estão em desenvolvimento, aguarde novos patches.

## 💻 Requisitos Windows

- Rust ( 2021 ou superior )
- CMake
- Compilador C++ compatível
- Visual Studio Installer ( Ferramentas de desenvolvimento desktop com C++ )

## 🐧 Requisitos Linux 

- Rust ( 2021 ou superior )
- Clang
- Libclang
- CMake
- Libssl-dev & OpenSSL devel
- pkg-config
- Tar, Unzip & Bsdtar
- g++


## 🍎 Requisitos MacOS

- Rust ( 2021 ou superior )
- homebrew
- Xcode
- Apple clang via xcode clt
- libclang & clang
- OpenSSL
- CMake
- pkg-config
- tar, unzip & curl


##  💻 Compilar Windows

```bash
# Clone o repositório
git clone https://github.com/CopperRS/AI-Copper.git
cd AI_Copper

# Compile
cargo build 

# Execute exemplos
cargo run --example advanced_features

# Execute
cargo run
```

## 🐧 Compilar Linux 

```bash
# Clone o repositório
git clone https://github.com/CopperRS/AI-Copper.git
cd AI_Copper

# Compile
cargo build 

# Execute exemplos
cargo run --example advanced_features

# Execute
cargo run
```

##  🍎 Compilar MacOS

```bash
# Clone o repositório
git clone https://github.com/CopperRS/AI-Copper.git
cd AI_Copper

# Compile
cargo build 

# Execute exemplos
cargo run --example advanced_features

# Execute
cargo run
```

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja em **[LICENSE](LICENSE)** para detalhes.

## ⚡ Contribuidores

<table>
  <tbody>
    <tr style="align-items: center">
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Spectrevz"><img src="https://avatars.githubusercontent.com/u/142043449?v=4?s=100" width="100px;" style="border-radius: 50%"; alt="Spectrevz"/> <br /><sub><b>Spectrevz</b></sub></a><br />
      </td>
      <td align="center"  valign="top" width="14.28%"><a href="https://github.com/Moonx0207"><img src="https://avatars.githubusercontent.com/u/214397746?v=4?s=100" width="100px;" style="border-radius: 50%";  alt="Moonx0207"/> <br /><sub><b>Moonx0207</b></sub></a><br />
      </td>
    </tr>
  </tbody>
</table>
