# 📋 A fazer

> Links da documentação e lista de tarefas

---

## 📚 Documentação

### LibTorch

🔗 [LibTorch Official Documentation](https://pytorch.org/cppdocs/)

### TensorFlow

🔗 [TensorFlow C++ API Documentation](https://www.tensorflow.org/api_docs/cc)

---

## ✅ Tarefas

### 🔥 LibTorch

#### ✨ Já implementado

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3

#### 🚀 A fazer

- [ ] **Tarefa 1:** Description pending
- [ ] Add tensor operations
- [ ] Implement device management
- [ ] Add examples and tests

---

### 🌊 TensorFlow - FOCO

#### ✨ Já implementado

**OPERAÇÕES MATEMÁTICAS (Math Ops):**

- [ ✅ ] Add (+), Sub (-), Mul (*), Div (/) - Operadores aritméticos
- [ ✅ ] MatMul - Multiplicação de matrizes
- [ ✅ ] BatchMatMul/V2/V3 - Multiplicação em batch
- [ ✅ ] Pow, Sqrt, Square, Abs - Funções matemáticas básicas
- [ ✅ ] Exp, Log, Log1p, Sigmoid, Tanh - Funções exponenciais
- [ ✅ ] Sin, Cos, Tan, Asin, Acos, Atan - Trigonométricas
- [ ✅ ] Equal, NotEqual, Greater, Less - Comparações
- [ ✅ ] LogicalAnd, LogicalOr, LogicalNot - Lógicas
- [ ✅ ] Ceil, Floor, Round, Clip - Arredondamento
- [ ✅ ] Cast - Conversão de tipos

**OPERAÇÕES ALEATÓRIAS (Random Ops):**

- [ ✅ ] RandomUniform, RandomNormal
- [ ✅ ] TruncatedNormal, RandomGamma
- [ ✅ ] RandomShuffle, Multinomial

**OTIMIZADORES (Training Ops):**

- [ ✅ ] GradientDescent (SGD)
- [ ✅ ] Adam, Adagrad, RMSProp
- [ ✅ ] Momentum, Adadelta, Ftrl

**SPARSE TENSORS (Sparse Ops):**

- [ ✅ ] SparseAdd, SparseTensorDenseMatMul
- [ ✅ ] SparseConcat, SparseSlice
- [ ✅ ] SparseReshape

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

Apenas f32, apenas 2D para transpose, sem Clone nativo, sem operadores matemáticos (+, -, *, /) implementados diretamente no FlowTensors.

#### 🚀 A fazer

**OPERAÇÕES DE ARRAY (Array Ops):**

- [ ] Concat - Concatenar tensors
- [ ] Stack/Unstack - Empilhar/desempilhar
- [ ] Split - Dividir tensor
- [ ] Slice - Fatiar tensor
- [ ] Gather/GatherNd - Coletar elementos
- [ ] Transpose N-dimensional - Transpor qualquer dimensão
- [ ] Fill - Preencher com valor
- [ ] Pad/PadV2 - Padding
- [ ] Reverse - Reverter tensor
- [ ] OneHot - Codificação one-hot
- [ ] Where - Selecionar por condição

**REDES NEURAIS (NN Ops):**

- [ ] Relu, Relu6, Elu, Selu - Ativações
- [ ] Softmax, LogSoftmax - Normalização
- [ ] Conv2D/Conv3D - Convolução
- [ ] MaxPool/AvgPool - Pooling
- [ ] BatchNormalization - Normalização
- [ ] SoftmaxCrossEntropy - Loss functions
- [ ] BiasAdd - Adicionar bias
- [ ] Dropout - Regularização

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
