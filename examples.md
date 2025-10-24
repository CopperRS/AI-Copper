
## 🎓 Exemplos de Uso

### 1. Usando LibTorch

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

### 3. API Unificada

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
