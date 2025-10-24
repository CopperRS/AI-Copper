
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