# 🚀 AI Copper

**AI Copper** é uma biblioteca Rust unificada que combina as capacidades do **LibTorch** (PyTorch C++) e **TensorFlow C API** em uma interface única e conveniente. Crie modelos de machine learning e deep learning usando o melhor dos dois frameworks!

## ✨ Características

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

Adicione ao seu `Cargo.toml`:

```toml
[dependencies]
ai_copper = "0.1.3"
```

**🎉 Novidade!** As bibliotecas (LibTorch e TensorFlow) são baixadas automaticamente na primeira compilação. Não é mais necessário configurar variáveis de ambiente!

Para mais detalhes sobre instalação, veja [INSTALLATION.md](INSTALLATION.md).

### Backends Opcionais

```toml
# Apenas LibTorch
ai_copper = { version = "0.1.3", default-features = false, features = ["libtorch"] }

# Apenas TensorFlow  
ai_copper = { version = "0.1.3", default-features = false, features = ["tensorflow"] }

# Ambos (padrão)
ai_copper = "0.1.3"
```

## ✨ Novidades v0.1.3

Esta versão adiciona **24 novas funções** focadas em Deep Learning moderno e **download automático de dependências**:

### 🚀 Download Automático
- ✅ LibTorch 2.1.0 baixado automaticamente
- ✅ TensorFlow 2.10.0 baixado automaticamente
- ✅ Sem necessidade de configurar variáveis de ambiente
- ✅ Suporte a backends opcionais via features

### 🔥 Funções de Ativação
```rust
use ai_copper::Tensor;

let x = Tensor::from_values(&[-2.0, -1.0, 0.0, 1.0, 2.0], 1, 5);
let relu = x.relu();      // [0, 0, 0, 1, 2]
let sigmoid = x.sigmoid(); // Valores entre 0 e 1
let tanh = x.tanh();       // Valores entre -1 e 1
```

### 📐 Funções Matemáticas
```rust
let data = Tensor::rand(3, 3);
let sin_data = data.sin();      // Seno
let exp_data = data.exp();      // e^x
let sqrt_data = data.sqrt();    // Raiz quadrada
let pow_data = data.pow(2.0);   // x^2
```

### 🎲 Criação Avançada de Tensores
```rust
let normal = Tensor::randn(3, 3);    // Distribuição normal
let identity = Tensor::eye(5);       // Matriz identidade 5x5
let zeros = normal.zeros_like();     // Zeros com mesma forma
let ones = normal.ones_like();       // Uns com mesma forma
```

### 📊 Estatísticas Avançadas
```rust
let data = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
println!("Desvio Padrão: {}", data.std());  // ~1.414
println!("Variância: {}", data.var());       // ~2.0
println!("Argmax: {}", data.argmax());       // 4 (índice do máximo)
println!("Argmin: {}", data.argmin());       // 0 (índice do mínimo)
```

### 🧠 Neural Networks Melhorados
```rust
use ai_copper::{Linear, Optimizer};

// Otimizador Adam (NOVO!)
let model = Linear::new(784, 10);
let optimizer = Optimizer::adam(&model, 0.001);

// Classificação com CrossEntropy (NOVO!)
let predictions = model.forward(&input).relu();
let loss = predictions.cross_entropy_loss(&labels);
loss.backward();
optimizer.step();
```

## 🎓 Exemplos de Uso

### 1. Usando LibTorch Diretamente

```rust
use ai_copper::Tensor;

fn main() {
    // Criar tensores
    let t1 = Tensor::ones(2, 3);
    let t2 = Tensor::rand(2, 3);
    let t3 = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    
    // Operações
    println!("Soma: {}", t1.sum());
    println!("Média: {}", t1.mean());
    
    // Operações aritméticas
    let t4 = t1 + t2;
    t4.print();
    
    // Transposta
    let t5 = t3.transpose();
    
    // Multiplicação de matrizes
    let a = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Tensor::from_values(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = a.matmul(&b);
}
```

### 2. Usando TensorFlow

```rust
use ai_copper::{FlowTensors, TensorFlowModel};

fn main() {
    // Versão
    println!("TensorFlow: {}", FlowTensors::version_tf());
    
    // Criar tensores
    let t1 = FlowTensors::ones(&[2, 3]).unwrap();
    let t2 = FlowTensors::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    
    // Operações
    println!("Soma: {}", t1.sum());
    println!("Média: {}", t2.mean());
    
    // Transposta
    let t3 = t2.transpose().unwrap();
    
    // Carregar modelo
    let model = TensorFlowModel::load("model_path", "serve").unwrap();
    let outputs = model.run(&["input"], &[&t1], &["output"]).unwrap();
}
```

### 3. API Unificada - O Melhor dos Dois Mundos

```rust
use ai_copper::{UnifiedTensor, Backend, Device};

fn main() {
    let device = Device::CPU;
    
    // Usar LibTorch
    let t1 = UnifiedTensor::ones(2, 3, Backend::LibTorch, device);
    
    // Usar TensorFlow
    let t2 = UnifiedTensor::rand(2, 3, Backend::TensorFlow, device);
    
    // Converter entre backends
    let t3 = t2.to_backend(Backend::LibTorch);
    
    // Operações funcionam independente do backend
    println!("Soma: {}", t3.sum());
    println!("Média: {}", t3.mean());
    
    // Operações aritméticas
    let t4 = UnifiedTensor::from_values(&[1.0, 2.0, 3.0, 4.0], 2, 2, Backend::LibTorch, device);
    let t5 = UnifiedTensor::from_values(&[5.0, 6.0, 7.0, 8.0], 2, 2, Backend::LibTorch, device);
    let t6 = t4 + t5;
}
```

### 4. Treinamento de Modelos

```rust
use ai_copper::{Tensor, Linear, Optimizer};

fn main() {
    // Dados: y = 2*x + 1
    let x = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0], 4, 1);
    let y = Tensor::from_values(&[3.0, 5.0, 7.0, 9.0], 4, 1);
    
    // Modelo
    let linear = Linear::new(1, 1);
    let optimizer = Optimizer::sgd(&linear, 0.01);
    
    // Treinar
    for epoch in 0..100 {
        let pred = linear.forward(&x);
        let loss = pred.mse_loss(&y);
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss.as_slice()[0]);
        }
        
        loss.backward();
        optimizer.step();
    }
    
    // Testar
    let test = Tensor::from_values(&[5.0], 1, 1);
    let result = linear.forward(&test);
    result.print();
}
```

## 📚 Documentação

- **[INSTALLATION.md](INSTALLATION.md)** - Guia completo de instalação e troubleshooting
- **[QUICKSTART.md](QUICKSTART.md)** - Exemplos práticos e início rápido
- **[Documentação da API](docs/index.md)** - Referência completa da API

### Scripts de Teste

```bash
# Windows
.\test-installation.ps1

# Linux/Mac
chmod +x test-installation.sh
./test-installation.sh
```

## 📚 API Completa

### Tensor Operations (LibTorch)

```rust
// Criação
Tensor::ones(rows, cols)
Tensor::zeros(rows, cols)
Tensor::rand(rows, cols)
Tensor::from_values(&values, rows, cols)

// Operações
tensor.sum()
tensor.mean()
tensor.max()
tensor.min()
tensor.transpose()
tensor.reshape(new_rows, new_cols)
tensor.matmul(&other)
tensor.map(|x| x * 2.0)

// Aritmética
t1 + t2
t1 - t2
t1 * t2
t1 / t2

// Neural Networks
Linear::new(in_features, out_features)
linear.forward(&input)
tensor.mse_loss(&target)
tensor.backward()

Optimizer::sgd(&linear, learning_rate)
optimizer.step()
```

### TensorFlow Tensors

```rust
// Criação
FlowTensors::new(&values, &dims)
FlowTensors::zeros(&dims)
FlowTensors::ones(&dims)

// Operações
tensor.sum()
tensor.mean()
tensor.max()
tensor.min()
tensor.transpose()
tensor.reshape(&new_dims)
tensor.map(|x| x * 2.0)
tensor.data()
tensor.dims()

// Modelo
TensorFlowModel::load(path, tags)
model.run(&input_names, &inputs, &output_names)
```

### Unified API

```rust
// Criação
UnifiedTensor::ones(rows, cols, backend, device)
UnifiedTensor::zeros(rows, cols, backend, device)
UnifiedTensor::rand(rows, cols, backend, device)
UnifiedTensor::from_values(&values, rows, cols, backend, device)

// Conversão
tensor.to_backend(Backend::TensorFlow)
tensor.backend()

// Operações (mesmo API independente do backend)
tensor.sum()
tensor.mean()
tensor.max()
tensor.min()
tensor.transpose()
tensor.map(|x| x * 2.0)
tensor.shape()
```

## 🏗️ Estrutura do Projeto

```
ai_copper/
├── src/
│   ├── lib.rs                     # Módulo principal
│   ├── unified/                   # API Unificada
│   │   ├── mod.rs
│   │   ├── tensor.rs             # UnifiedTensor
│   │   ├── backend.rs            # Backend enum
│   │   └── device.rs             # Device enum
│   ├── tensor_libortch/          # LibTorch backend
│   │   ├── mod.rs
│   │   ├── tensor.rs             # Tensor, Linear, Optimizer
│   │   ├── operators.rs          # Sobrecarga de operadores
│   │   └── ffi.rs                # FFI bindings
│   └── tensor_tensorflow/        # TensorFlow backend
│       ├── mod.rs
│       ├── tensors_flow.rs       # FlowTensors, Model
│       └── ffi.rs                # FFI bindings
├── cpp/
│   └── lib.cpp                   # Implementação C++
├── examples/
│   └── basic_usage.rs            # Exemplos de uso
├── Cargo.toml
└── README.md
```

## 🔧 Requisitos

- Rust 2021 edition ou superior
- LibTorch (para funcionalidades LibTorch)
- TensorFlow C API (para funcionalidades TensorFlow)
- CMake (para compilação C++)
- Compilador C++ compatível

## 🚀 Como Compilar

```bash
# Clone o repositório
git clone https://github.com/seu_usuario/ai_copper.git
cd ai_copper

# Compile
cargo build --release

# Execute exemplos
cargo run --example basic_usage

# Execute testes
cargo test
```

## 📖 Documentação

Gere a documentação completa com:

```bash
cargo doc --open
```

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.

## 👨‍💻 Autores

**Ryan Lima** - [ryan2020gary@gmail.com](mailto:ryan2020gary@gmail.com)
**Rodrigo Dias** - [rodrigods.dev@gmail.com](mailto:rodrigods.dev@gmail.com)

## 🌟 Agradecimentos

- PyTorch Team pela excelente biblioteca LibTorch
- TensorFlow Team pela TensorFlow C API
- Comunidade Rust pelo ecossistema incrível

---

**Made with ❤️ and Rust 🦀**


**`ai_copper`** is a library developed for the **`Copper language`**, written in Rust with revamped C++ functions, which provides connections to **`PyTorch's`** libtorch C++ library and **`TensorFlow's`** C++ library. It allows you to create and manipulate tensors, perform machine learning operations, and use pre-trained TensorFlow models directly in Copper. The library is designed to facilitate integration between Copper, Rust, and C++ in projects that use libtorch and TensorFlow for machine learning implementations.

## Features

- Create tensors directly in Rust using `libtorch`.
- Support for basic tensor manipulation operations.
- Easy integration with PyTorch for running machine learning models in Rust.
- Support for TensorFlow C++ library integration.

## Requirements

- **Rust**: The library is designed to work with the latest version of Rust.
- **C++**: A C++ compiler (such as g++ or Clang) must be installed on the system.
- **CMake**: You must have [CMake](https://cmake.org/download/) installed and available in your PATH.
- **MSVC/Build Tools (Windows)**: On Windows, you need the [Microsoft Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (MSVC) with the C++ workload enabled.
- **libtorch**: The PyTorch C++ library (CPU version) must be installed on your system.
- **TensorFlow C++**: The TensorFlow C++ library (CPU version) must be installed on your system (optional, for TensorFlow integration)(https://www.tensorflow.org/install/lang_c?hl=pt-br).
- **Copper**: The Copper language must be configured in the environment to use the library.

## Environment variables

### Windows

- `LIBTORCH` — Path to the root of libtorch (e.g., `C:\libtorch`)
- `TENSORFLOW_ROOT` — Path to the root of TensorFlow C++ (e.g., `C:\libtensorflow`)

How to set (temporary, for the current terminal only):

```powershell
$env:LIBTORCH = "C:\libtorch"
$env:TENSORFLOW_ROOT = "C:\libtensorflow"
```

Or set permanently via Control Panel → System → Advanced → Environment Variables.

> The build.rs will automatically copy the required DLLs from `libtorch/lib` and `libtensorflow/lib` to the executable directory.

### Linux

Set the environment variables:

```bash
export LIBTORCH=/home/youruser/libtorch
export TENSORFLOW_ROOT=/home/youruser/libtensorflow
```

For temporary use, run the commands above in the terminal before building.

To make it permanent, add to the end of your `~/.bashrc` or `~/.zshrc`:

```bash
export LIBTORCH=/home/youruser/libtorch
export TENSORFLOW_ROOT=/home/youruser/libtensorflow
```

> The build.rs will automatically copy the required .so libraries from `libtorch/lib` and `libtensorflow/lib` to the executable directory.

**Attention:** If the variables are not set correctly, the build will fail or the executable will not find the required libraries at runtime.

## Installation

**Windows**

To add `ai_copper` to your Rust project, simply include the following line in your `Cargo.toml:`

```toml
[dependencies]
ai_copper = { git = "https://github.com/CopperRS/ai_copper.git" }
```

Then, run:

```
cargo build
cargo run
```

> The build script automatically copies the necessary DLLs from libtorch and TensorFlow folders to your target directory.  
> You do **not** need to manually copy DLLs or adjust the PATH after setup.

---

**Linux**

To use `ai_copper` on Linux, you need to clone the repository and build it locally to generate the shared library (`libai_copper.so`). Follow these steps:

1. Clone the Repository:

```bash
git clone https://github.com/CopperRS/ai_copper.git
cd ai_copper
```

2. Install to cmake and g++ or clang

**`Terminal`**

```
#g++
sudo apt-get install build-essential

#clang
sudo apt-get install clang

#cmake
sudo apt-get install cmake
```

3. Create the .so file to use the lib.

**`Terminal`**

```
cd cpp
mkdir build && cd build
cmake ..
cmake --build .
cd ../..
cargo build
```

4. Add as a Local Dependency: In your project's `Cargo.toml`, add `ai_copper` as a path dependency, pointing to the cloned repository:

```toml
[dependencies]
ai_copper = { path = "/path/to/ai_copper" }
```

Replace `/path/to/ai_copper` with the actual path where you cloned the repository

5. Build the Project: Run the following command in your project directory to build the project and generate the `libai_copper.so` file

```bash
cargo build
```

This will create the shared library in `/path/to/ai_copper/cpp/build`.

6. Run the Project: Before running your project, set the `LD_LIBRARY_PATH` to include the directory containing `libai_copper.so`, libtorch, and TensorFlow libs:

**_If you haven't defined the variables permanently, you can temporarily set them to run at runtime._**

```bash
export LIBTORCH_PATH=/home/yourname/libtorch
export TENSORFLOW_ROOT=/home/yourname/libtensorflow
export LD_LIBRARY_PATH=/path/to/ai_copper/cpp/build:$LIBTORCH_PATH/lib:$TENSORFLOW_ROOT/lib:$LD_LIBRARY_PATH
cargo run
```

---

**Notes**

- Ensure that the TensorFlow C++ package you use contains the DLLs (`.dll`) on Windows or shared objects (`.so`) on Linux inside the `lib` folder.
- The build script of `ai_copper` will handle copying DLLs automatically on Windows.
- On Linux, it is essential to have your `LD_LIBRARY_PATH` correctly set so that runtime linking works.
- This setup allows you to use both PyTorch (`libtorch`) and TensorFlow C++ APIs seamlessly within Rust using `ai_copper`.

