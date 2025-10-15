// Exemplo demonstrando as novas funcionalidades do ai_copper
use ai_copper::tensor_libortch::tensor::{Tensor, Linear, Optimizer};

fn main() {
    println!("=== AI Copper - Advanced Features Demo ===\n");

    // ==================== CRIAÇÃO DE TENSORES ====================
    println!("1. CRIAÇÃO DE TENSORES");
    println!("{}", "-".repeat(50));
    
    let zeros = Tensor::zeros(2, 3);
    println!("Zeros (2x3):");
    zeros.print();
    
    let randn = Tensor::randn(2, 3);
    println!("\nRandn - Distribuição Normal (2x3):");
    randn.print();
    
    let eye = Tensor::eye(3);
    println!("\nEye - Matriz Identidade (3x3):");
    eye.print();
    
    let ones = Tensor::ones(2, 2);
    let zeros_like = ones.zeros_like();
    println!("\nZeros Like (mesma forma que ones):");
    zeros_like.print();
    
    // ==================== FUNÇÕES MATEMÁTICAS ====================
    println!("\n2. FUNÇÕES MATEMÁTICAS");
    println!("{}", "-".repeat(50));
    
    let x = Tensor::from_values(&[0.0, 1.0, 2.0, 3.0], 2, 2);
    println!("Tensor original:");
    x.print();
    
    let sin_x = x.sin();
    println!("\nSin(x):");
    sin_x.print();
    
    let exp_x = x.exp();
    println!("\nExp(x):");
    exp_x.print();
    
    let sqrt_abs = x.abs().sqrt();
    println!("\nSqrt(Abs(x)):");
    sqrt_abs.print();
    
    let pow_x = x.pow(2.0);
    println!("\nPow(x, 2):");
    pow_x.print();
    
    // ==================== FUNÇÕES DE ATIVAÇÃO ====================
    println!("\n3. FUNÇÕES DE ATIVAÇÃO");
    println!("{}", "-".repeat(50));
    
    let activations = Tensor::from_values(&[-2.0, -1.0, 0.0, 1.0, 2.0], 1, 5);
    println!("Valores originais:");
    activations.print();
    
    let relu = activations.relu();
    println!("\nReLU:");
    relu.print();
    
    let sigmoid = activations.sigmoid();
    println!("\nSigmoid:");
    sigmoid.print();
    
    let tanh = activations.tanh();
    println!("\nTanh:");
    tanh.print();
    
    // ==================== ESTATÍSTICAS ====================
    println!("\n4. ESTATÍSTICAS AVANÇADAS");
    println!("{}", "-".repeat(50));
    
    let data = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    println!("Dataset:");
    data.print();
    
    println!("\nEstatísticas:");
    println!("  Soma: {:.4}", data.sum());
    println!("  Média: {:.4}", data.mean());
    println!("  Desvio Padrão: {:.4}", data.std());
    println!("  Variância: {:.4}", data.var());
    println!("  Máximo: {:.4}", data.max());
    println!("  Mínimo: {:.4}", data.min());
    println!("  Argmax (índice): {}", data.argmax());
    println!("  Argmin (índice): {}", data.argmin());
    
    // ==================== NEURAL NETWORK COM NOVAS FEATURES ====================
    println!("\n5. REDE NEURAL COM NOVAS FEATURES");
    println!("{}", "-".repeat(50));
    
    // Criar uma rede neural simples
    let layer1 = Linear::new(2, 3);
    let layer2 = Linear::new(3, 1);
    
    // Otimizador Adam (NOVO!)
    let optimizer = Optimizer::adam(&layer1, 0.01);
    
    // Dados de treinamento
    let x_train = Tensor::from_values(&[0.5, 0.3, 0.8, 0.2, 0.1, 0.9], 3, 2);
    let y_train = Tensor::from_values(&[1.0, 0.5, 0.2], 3, 1);
    
    println!("Treinando com Adam optimizer...");
    
    for epoch in 0..5 {
        // Forward pass com ativação ReLU
        let h1 = layer1.forward(&x_train);
        let h1_activated = h1.relu(); // ReLU activation (NOVO!)
        
        let output = layer2.forward(&h1_activated);
        
        // Loss (pode usar MSE ou CrossEntropy)
        let loss = output.mse_loss(&y_train);
        
        println!("Epoch {}: Loss = {:.6}", epoch, loss.as_slice()[0]);
        
        // Backward e otimização
        loss.backward();
        optimizer.step();
    }
    
    println!("\n=== Demo Completo! ===");
    println!("\nNovas funcionalidades demonstradas:");
    println!("  ✓ randn, eye, zeros_like, ones_like");
    println!("  ✓ sin, cos, exp, log, sqrt, abs, pow");
    println!("  ✓ relu, sigmoid, tanh");
    println!("  ✓ std, var, argmax, argmin");
    println!("  ✓ Adam optimizer");
}
