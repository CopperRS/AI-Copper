# 📋 A fazer

> Links da documentação e lista de tarefas

---

## 📚 Documentação

### LibTorch

🔗 [LibTorch Official Documentation](https://pytorch.org/cppdocs/)

### TensorFlow

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
- [ ] **Tarefa 1:** Description pending
- [ ] Add tensor operations
- [ ] Implement device management

### 🌊 TensorFlow - FOCO


- [ ✅ ] Add (+), Sub (-), Mul (*), Div (/) - Operadores aritméticos
- [ ✅ ] MatMul - Multiplicação de matrizes
Nota: stubs de implementação e exemplos foram adicionados ao backend TensorFlow (`src/tensor_tensorflow/tensors_flow.rs`) e demonstrados em `examples/advanced_features.rs`.
- [ ✅ ] RandomUniform, RandomNormal
- [ ✅ ] TruncatedNormal, RandomGamma
**OTIMIZADORES (Training Ops):**

- [ ✅ ] GradientDescent (SGD)
- [ ✅ ] Adam, Adagrad, RMSProp
- [ ✅ ] Momentum, Adadelta, Ftrl

**SPARSE TENSORS (Sparse Ops):**

- [ ✅ ] SparseAdd, SparseTensorDenseMatMul
- [ ✅ ] SparseConcat, SparseSlice
- [ ✅ ] SparseReshape

**OPERAÇÕES DE ARRAY (Array Ops):**

- [ ✅ ] Concat - Concatenar tensors
- [ ✅ ] Stack/Unstack - Empilhar/desempilhar
- [ ✅ ] Split - Dividir tensor
- [ ✅ ] Slice - Fatiar tensor
- [ ✅ ] Gather/GatherNd - Coletar elementos
- [ ✅ ] Transpose N-dimensional - Transpor qualquer dimensão
- [ ✅ ] Fill - Preencher com valor
- [ ✅ ] Pad/PadV2 - Padding
- [ ✅ ] Reverse - Reverter tensor
- [ ✅ ] OneHot - Codificação one-hot
- [ ✅ ] Where - Selecionar por condição

**REDES NEURAIS (NN Ops):**

- [ ✅ ] Relu, Relu6, Elu, Selu - Ativações
- [ ✅ ] Softmax, LogSoftmax - Normalização
- [ ✅ ] Conv2D/Conv3D - Convolução
- [ ✅ ] MaxPool/AvgPool - Pooling
- [ ✅ ] BatchNormalization - Normalização
- [ ✅ ] SoftmaxCrossEntropy - Loss functions
- [ ✅ ] BiasAdd - Adicionar bias
- [ ✅ ] Dropout - Regularização

### **Structs**

- `FlowTensors` - tensor do TensorFlow
- `TensorFlowModel` - modelo SavedModel

### **Métodos de FlowTensors**

```rust
// Criação
new(values: &[f32], dims: &[i64])
zeros(dims: &[i64])
ones(dims: &[i64])

// Acesso
data() -> Option<&[f32]>
dims() -> &[i64]

// Estatísticas
sum() -> f32
mean() -> f32
max() -> f32
min() -> f32

// Transformações
transpose() -> Option<FlowTensors>  // apenas 2D
reshape(new_dims: &[i64]) -> Option<FlowTensors>
map<F>(f: F) -> Option<FlowTensors>

// Utilidade
version_tf() -> String
```

### **Métodos de TensorFlowModel**

```rust
load(model_path: &str, tags: &str) -> Option<Self>
run(input_names: &[&str], input_tensors: &[&FlowTensors], output_names: &[&str]) -> Option<Vec<FlowTensors>>
```

### **FFI (C++ Bindings)**

```rust
VersionTF()
LoadSavedModel()
RunSession()
CreateTFTensor()
GetTensorData()
FreeTFTensor()
FreeModel()
```

### **Integração Unificada**

```rust
UnifiedTensor::TensorFlow(FlowTensors)
// Suporta: zeros, ones, rand, from_values, to_backend, shape, as_slice, print
```

### **Tipos de dados**

- [ ✅ ] f32 (já implementado)
- [ ✅ ] f64 (double), i32, i64, i8, i16, u8, u16
- [ ✅ ] bool, complex64/128, string

---

#### 🚀 A fazer

**OPERAÇÕES DE IMAGEM (Image Ops):**

- [ ] Resize (Bilinear/Bicubic/NearestNeighbor)
- [ ] CropAndResize
- [ ] Decode/Encode (Jpeg/Png)
- [ ] AdjustContrast/Hue/Saturation
- [ ] RGBToHSV/HSVToRGB

**FUNCIONALIDADES AVANÇADAS:**

- [ ] Clone nativo para FlowTensors
- [ ] Gradientes/Autograd (GradientTape)
- [ ] GPU support (Device management)
- [ ] Broadcasting automático
- [ ] JIT compilation (XLA)
- [ ] Control flow (if/while/for)
- [ ] Save/Load checkpoints
- [ ] Dataset API

**TESTES & DOCS:**

- [ ] Testes unitários completos
- [ ] Benchmarks de performance

### **SEMPRE** atualize a documentação e essa lista
