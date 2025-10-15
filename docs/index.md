<<<<<<< HEAD
# Introducion 

This library was created to help you create and train neural networks and use pre-trained models for your future projects.

This documentation will teach you how to use the library's basic commands, from creating simple tensors to neural networks.


# Import

Read the **README.MD** to install, here only the lib import

**Rust**
```
    use ai_copper::{Tensor, operators, Optimizer, Linear};
=======
# 🚀 AI Copper Documentation

**AI Copper** is a unified Rust library that combines the capabilities of **LibTorch** (PyTorch C++) and **TensorFlow C API** into a single, convenient interface. Create machine learning and deep learning models using the best of both frameworks!

This documentation will teach you how to use the library's basic commands, from creating simple tensors to neural networks.

## ✨ Features

### 🔥 Dual Backend Support
- **LibTorch Backend**: Complete access to PyTorch functionalities in C++
- **TensorFlow Backend**: Native support for TensorFlow C API
- **Unified API**: Switch between backends without changing your code

### 🎯 Main Functionalities

#### Tensor Operations
- ✅ Tensor creation (zeros, ones, rand, randn, eye, from_values)
- ✅ Arithmetic operations (+, -, *, /)
- ✅ Matrix operations (matmul, transpose)
- ✅ Statistics (sum, mean, max, min, std, var, argmax, argmin)
- ✅ Mathematical functions (sin, cos, exp, log, sqrt, abs, pow)
- ✅ Activation functions (relu, sigmoid, tanh)
- ✅ Transformations (map, reshape, zeros_like, ones_like)
- ✅ Conversion between backends

#### Neural Networks (LibTorch)
- ✅ Linear layers
- ✅ Loss functions (MSE Loss, Cross Entropy Loss)
- ✅ Activation functions (ReLU, Sigmoid, Tanh)
- ✅ Optimizers (SGD, Adam)
- ✅ Automatic backpropagation
- ✅ Model training

#### TensorFlow Integration
- ✅ Load SavedModel models
- ✅ Run inference
- ✅ Multi-dimensional tensor manipulation
- ✅ Basic tensor operations

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ai_copper = "0.1.3"
```

## ✨ What's New in v0.1.3

This version adds **24 new functions** focused on modern Deep Learning:

### 🔥 Activation Functions
```rust
use ai_copper::Tensor;

let x = Tensor::from_values(&[-2.0, -1.0, 0.0, 1.0, 2.0], 1, 5);
let relu = x.relu();      // [0, 0, 0, 1, 2]
let sigmoid = x.sigmoid(); // Values between 0 and 1
let tanh = x.tanh();       // Values between -1 and 1
```

### 📐 Mathematical Functions
```rust
let data = Tensor::rand(3, 3);
let sin_data = data.sin();      // Sine
let exp_data = data.exp();      // e^x
let sqrt_data = data.sqrt();    // Square root
let pow_data = data.pow(2.0);   // x^2
```

### 🎲 Advanced Tensor Creation
```rust
let normal = Tensor::randn(3, 3);    // Normal distribution
let identity = Tensor::eye(5);       // 5x5 identity matrix
let zeros = normal.zeros_like();     // Zeros with same shape
let ones = normal.ones_like();       // Ones with same shape
```

### 📊 Advanced Statistics
```rust
let data = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
println!("Standard Deviation: {}", data.std());  // ~1.414
println!("Variance: {}", data.var());       // ~2.0
println!("Argmax: {}", data.argmax());       // 4 (maximum index)
println!("Argmin: {}", data.argmin());       // 0 (minimum index)
```

### 🧠 Improved Neural Networks
```rust
use ai_copper::{Linear, Optimizer};

// Adam Optimizer (NEW!)
let model = Linear::new(784, 10);
let optimizer = Optimizer::adam(&model, 0.001);

// Classification with CrossEntropy (NEW!)
let predictions = model.forward(&input).relu();
let loss = predictions.cross_entropy_loss(&labels);
loss.backward();
optimizer.step();
```

## 🎓 Usage Examples

### 1. Using LibTorch Directly

```rust
use ai_copper::Tensor;

fn main() {
    // Create tensors
    let t1 = Tensor::ones(2, 3);
    let t2 = Tensor::rand(2, 3);
    let t3 = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    
    // Operations
    println!("Sum: {}", t1.sum());
    println!("Mean: {}", t1.mean());
    
    // Arithmetic operations
    let t4 = t1 + t2;
    t4.print();
    
    // Transpose
    let t5 = t3.transpose();
    
    // Matrix multiplication
    let a = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Tensor::from_values(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = a.matmul(&b);
}
```

### 2. Using TensorFlow

```rust
use ai_copper::{FlowTensors, TensorFlowModel};

fn main() {
    // Version
    println!("TensorFlow: {}", FlowTensors::version_tf());
    
    // Create tensors
    let t1 = FlowTensors::ones(&[2, 3]).unwrap();
    let t2 = FlowTensors::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    
    // Operations
    println!("Sum: {}", t1.sum());
    println!("Mean: {}", t2.mean());
    
    // Transpose
    let t3 = t2.transpose().unwrap();
    
    // Load model
    let model = TensorFlowModel::load("model_path", "serve").unwrap();
    let outputs = model.run(&["input"], &[&t1], &["output"]).unwrap();
}
```

### 3. Unified API - The Best of Both Worlds

```rust
use ai_copper::{UnifiedTensor, Backend, Device};

fn main() {
    let device = Device::CPU;
    
    // Use LibTorch
    let t1 = UnifiedTensor::ones(2, 3, Backend::LibTorch, device);
    
    // Use TensorFlow
    let t2 = UnifiedTensor::rand(2, 3, Backend::TensorFlow, device);
    
    // Convert between backends
    let t3 = t2.to_backend(Backend::LibTorch);
    
    // Operations work regardless of backend
    println!("Sum: {}", t3.sum());
    println!("Mean: {}", t3.mean());
    
    // Arithmetic operations
    let t4 = UnifiedTensor::from_values(&[1.0, 2.0, 3.0, 4.0], 2, 2, Backend::LibTorch, device);
    let t5 = UnifiedTensor::from_values(&[5.0, 6.0, 7.0, 8.0], 2, 2, Backend::LibTorch, device);
    let t6 = t4 + t5;
}
```

### 4. Model Training

```rust
use ai_copper::{Tensor, Linear, Optimizer};

fn main() {
    // Data: y = 2*x + 1
    let x = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0], 4, 1);
    let y = Tensor::from_values(&[3.0, 5.0, 7.0, 9.0], 4, 1);
    
    // Model
    let linear = Linear::new(1, 1);
    let optimizer = Optimizer::sgd(&linear, 0.01);
    
    // Train
    for epoch in 0..100 {
        let pred = linear.forward(&x);
        let loss = pred.mse_loss(&y);
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss.as_slice()[0]);
        }
        
        loss.backward();
        optimizer.step();
    }
    
    // Test
    let test = Tensor::from_values(&[5.0], 1, 1);
    let result = linear.forward(&test);
    result.print();
}
```

## 📚 Complete API Reference

### Tensor Operations (LibTorch)

```rust
// Creation
Tensor::ones(rows, cols)
Tensor::zeros(rows, cols)
Tensor::rand(rows, cols)
Tensor::randn(rows, cols)           // NEW v0.1.3
Tensor::eye(size)                   // NEW v0.1.3
Tensor::from_values(&values, rows, cols)

// Statistics
tensor.sum()
tensor.mean()
tensor.max()
tensor.min()
tensor.std()                        // NEW v0.1.3
tensor.var()                        // NEW v0.1.3
tensor.argmax()                     // NEW v0.1.3
tensor.argmin()                     // NEW v0.1.3

// Transformations
tensor.transpose()
tensor.reshape(new_rows, new_cols)
tensor.zeros_like()                 // NEW v0.1.3
tensor.ones_like()                  // NEW v0.1.3
tensor.map(|x| x * 2.0)

// Matrix Operations
tensor.matmul(&other)

// Mathematical Functions
tensor.sin()                        // NEW v0.1.3
tensor.cos()                        // NEW v0.1.3
tensor.exp()                        // NEW v0.1.3
tensor.log()                        // NEW v0.1.3
tensor.sqrt()                       // NEW v0.1.3
tensor.abs()                        // NEW v0.1.3
tensor.pow(exponent)                // NEW v0.1.3

// Activation Functions
tensor.relu()                       // NEW v0.1.3
tensor.sigmoid()                    // NEW v0.1.3
tensor.tanh()                       // NEW v0.1.3

// Arithmetic
t1 + t2
t1 - t2
t1 * t2
t1 / t2

// Neural Networks
Linear::new(in_features, out_features)
linear.forward(&input)
tensor.mse_loss(&target)
tensor.cross_entropy_loss(&target)  // NEW v0.1.3
tensor.backward()

Optimizer::sgd(&linear, learning_rate)
Optimizer::adam(&linear, learning_rate)  // NEW v0.1.3
optimizer.step()
```

### TensorFlow Tensors

```rust
// Creation
FlowTensors::new(&values, &dims)
FlowTensors::zeros(&dims)
FlowTensors::ones(&dims)

// Operations
tensor.sum()
tensor.mean()
tensor.max()
tensor.min()
tensor.transpose()
tensor.reshape(&new_dims)
tensor.map(|x| x * 2.0)
tensor.data()
tensor.dims()

// Model
TensorFlowModel::load(path, tags)
model.run(&input_names, &inputs, &output_names)
```

### Unified API

```rust
// Creation
UnifiedTensor::ones(rows, cols, backend, device)
UnifiedTensor::zeros(rows, cols, backend, device)
UnifiedTensor::rand(rows, cols, backend, device)
UnifiedTensor::from_values(&values, rows, cols, backend, device)

// Conversion
tensor.to_backend(Backend::TensorFlow)
tensor.backend()

// Operations (same API regardless of backend)
tensor.sum()
tensor.mean()
tensor.max()
tensor.min()
tensor.transpose()
tensor.map(|x| x * 2.0)
tensor.shape()
```

## Import

**Rust**
```rust
use ai_copper::{Tensor, Linear, Optimizer};
use ai_copper::{FlowTensors, TensorFlowModel};
use ai_copper::{UnifiedTensor, Backend, Device};
>>>>>>> origin/master
```

**Copper**
```
<<<<<<< HEAD
    In development
```

# Compatibility

This lib is fully compatible with Rust and especially Copper.

# Order

The documentation for the commands and how to use them are separated into their respective folders.
=======
In development
```

## Compatibility

This library is fully compatible with Rust and especially designed for Copper.

## 📖 Documentation Structure

The documentation for specific commands and usage is organized into folders:
- `docs_libtorch/` - LibTorch-specific documentation
  - `1_tensor/Tensors.md` - Detailed tensor operations guide
  
See the detailed guides for comprehensive examples and advanced usage patterns.

## 🔧 Requirements

- Rust 2021 edition or higher
- LibTorch (for LibTorch functionalities)
- TensorFlow C API (for TensorFlow functionalities)
- CMake (for C++ compilation)
- Compatible C++ compiler

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Authors

**Ryan Lima** - [ryan2020gary@gmail.com](mailto:ryan2020gary@gmail.com)  
**Rodrigo Dias** - [rodrigods.dev@gmail.com](mailto:rodrigods.dev@gmail.com)

## 🌟 Acknowledgments

- PyTorch Team for the excellent LibTorch library
- TensorFlow Team for the TensorFlow C API
- Rust Community for the amazing ecosystem

---

**Made with ❤️ and Rust 🦀**
>>>>>>> origin/master

