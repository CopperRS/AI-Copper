# Exemplo de Uso Simplificado - AI Copper

## 🚀 Início Rápido

### 1. Crie um novo projeto

```bash
cargo new meu_projeto_ia
cd meu_projeto_ia
```

### 2. Adicione AI Copper ao Cargo.toml

```toml
[package]
name = "meu_projeto_ia"
version = "0.1.0"
edition = "2021"

[dependencies]
ai_copper = "0.1.3"
```

### 3. Use no seu código

Edite `src/main.rs`:

```rust
use ai_copper::Tensor;

fn main() {
    println!("🎯 Testando AI Copper com download automático!");
    
    // Cria tensor com valores aleatórios
    let tensor = Tensor::rand(3, 3);
    println!("Tensor aleatório:");
    tensor.print();
    
    // Operações matemáticas
    let result = tensor.relu().pow(2.0);
    println!("\nApós ReLU e elevado ao quadrado:");
    result.print();
    
    // Estatísticas
    println!("\nEstatísticas:");
    println!("Média: {}", result.mean());
    println!("Desvio Padrão: {}", result.std());
    println!("Máximo: {}", result.max());
    println!("Mínimo: {}", result.min());
}
```

### 4. Compile e execute

```bash
cargo build
cargo run
```

**Na primeira vez:**
- O sistema irá baixar LibTorch e TensorFlow automaticamente
- Isso pode levar alguns minutos (download de ~260 MB)
- As bibliotecas ficam em cache para uso futuro

**Nas próximas vezes:**
- A compilação será rápida
- As bibliotecas já estarão disponíveis

## 📊 Exemplo: Rede Neural Simples

```rust
use ai_copper::{Tensor, Linear};

fn main() {
    // Dados de entrada (4 amostras, 2 features cada)
    let x = Tensor::from_values(&[
        1.0, 2.0,  // amostra 1
        2.0, 3.0,  // amostra 2
        3.0, 4.0,  // amostra 3
        4.0, 5.0,  // amostra 4
    ], 4, 2);
    
    // Labels (4 amostras)
    let y = Tensor::from_values(&[0.0, 1.0, 1.0, 0.0], 4, 1);
    
    // Cria modelo linear (2 inputs -> 1 output)
    let model = Linear::new(2, 1);
    
    // Forward pass
    let predictions = model.forward(&x).sigmoid();
    
    println!("Predições:");
    predictions.print();
    
    println!("\nLabels verdadeiros:");
    y.print();
    
    // Calcula perda MSE
    let loss = predictions.mse_loss(&y);
    println!("\nPerda (MSE): {}", loss.sum());
}
```

## 🎲 Exemplo: Operações com Tensores

```rust
use ai_copper::Tensor;

fn main() {
    // Criação de tensores
    let zeros = Tensor::zeros(2, 3);
    let ones = Tensor::ones(2, 3);
    let identity = Tensor::eye(4);
    let random = Tensor::rand(3, 3);
    let normal = Tensor::randn(3, 3);
    
    println!("Matriz identidade 4x4:");
    identity.print();
    
    println!("\nDistribuição normal:");
    normal.print();
    
    // Operações aritméticas
    let a = Tensor::from_values(&[1.0, 2.0, 3.0], 1, 3);
    let b = Tensor::from_values(&[4.0, 5.0, 6.0], 1, 3);
    
    let soma = &a + &b;
    let produto = &a * &b;
    
    println!("\nA:");
    a.print();
    println!("B:");
    b.print();
    println!("A + B:");
    soma.print();
    println!("A * B:");
    produto.print();
    
    // Funções matemáticas
    let valores = Tensor::from_values(&[0.0, 1.57, 3.14], 1, 3);
    let senos = valores.sin();
    
    println!("\nSeno de [0, π/2, π]:");
    senos.print();
}
```

## 🧠 Exemplo: Treinamento de Rede Neural

```rust
use ai_copper::{Tensor, Linear, Optimizer};

fn main() {
    // Dados XOR
    let x_train = Tensor::from_values(&[
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ], 4, 2);
    
    let y_train = Tensor::from_values(&[
        0.0,
        1.0,
        1.0,
        0.0,
    ], 4, 1);
    
    // Modelo
    let layer1 = Linear::new(2, 4);  // 2 -> 4
    let layer2 = Linear::new(4, 1);  // 4 -> 1
    
    // Otimizador
    let mut optimizer = Optimizer::adam(&layer1, 0.01);
    
    // Treinamento
    println!("Iniciando treinamento...\n");
    
    for epoch in 0..1000 {
        // Forward
        let hidden = layer1.forward(&x_train).relu();
        let output = layer2.forward(&hidden).sigmoid();
        
        // Loss
        let loss = output.mse_loss(&y_train);
        
        // Backward
        loss.backward();
        
        // Update
        optimizer.step();
        
        if epoch % 100 == 0 {
            println!("Época {}: Loss = {}", epoch, loss.sum());
        }
    }
    
    println!("\nTreinamento concluído!");
    
    // Teste
    let hidden = layer1.forward(&x_train).relu();
    let predictions = layer2.forward(&hidden).sigmoid();
    
    println!("\nPredições finais:");
    predictions.print();
    println!("\nLabels verdadeiros:");
    y_train.print();
}
```

## 🔄 Conversão entre Backends

```rust
use ai_copper::{Tensor, TensorBackend};

fn main() {
    // Cria tensor no LibTorch
    let tensor_torch = Tensor::rand(3, 3);
    
    println!("Tensor LibTorch:");
    tensor_torch.print();
    
    // Converte para TensorFlow
    let tensor_tf = tensor_torch.to_tensorflow();
    
    println!("\nMesmo tensor no TensorFlow:");
    tensor_tf.print();
    
    // Converte de volta
    let tensor_torch2 = tensor_tf.to_libtorch();
    
    println!("\nDe volta ao LibTorch:");
    tensor_torch2.print();
}
```

## 📝 Dicas

### Apenas LibTorch (reduz download)

Se você só precisa do LibTorch:

```toml
[dependencies]
ai_copper = { version = "0.1.3", default-features = false, features = ["libtorch"] }
```

### Apenas TensorFlow (reduz download)

Se você só precisa do TensorFlow:

```toml
[dependencies]
ai_copper = { version = "0.1.3", default-features = false, features = ["tensorflow"] }
```

### Cache de Bibliotecas

As bibliotecas baixadas ficam em `deps/` e são reutilizadas. Para forçar novo download:

```bash
# Windows
Remove-Item -Recurse -Force deps

# Linux/Mac
rm -rf deps
```

## 🆘 Problemas?

Consulte [INSTALLATION.md](INSTALLATION.md) para troubleshooting detalhado.
