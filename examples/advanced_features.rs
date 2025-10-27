// Exemplo demonstrando as novas funcionalidades do ai_copper
use ai_copper::tensor_libortch::tensor::{Tensor, Linear, Optimizer};
use ai_copper::FlowTensors;

fn main() {
    // Demo principal: mostra várias funcionalidades oferecidas pela biblioteca
    // Comentários estão em português (pt-BR) explicando cada bloco.
    println!("=== AI Copper - Demo de Funcionalidades Avançadas ===\n");

    // ==================== CRIAÇÃO DE TENSORES ====================
    // Nesta seção criamos tensores básicos (zeros, randn, eye, ones, zeros_like)
    // Esses exemplos mostram as funções utilitárias para construir tensores.
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
    // Aplicamos funções elementares (sin, exp, sqrt, pow) sobre tensores.
    // Cada operação retorna um novo tensor com o resultado elemento-a-elemento.
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

    // Exemplos adicionais de LibTorch (operações úteis extras)
    println!("\nLibTorch extras:");
    // rand (uniform)
    let r = Tensor::rand(2, 3);
    println!("Rand (2x3):");
    r.print();

    // log - usar valores positivos para evitar -inf
    let pos = Tensor::from_values(&[0.1, 1.0, 2.0, 3.0], 2, 2);
    println!("\nLog(pos):");
    pos.log().print();

    // MatMul e Transpose: demonstra multiplicação matricial e transposição
    let a = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = Tensor::from_values(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    println!("\nMatMul a(2x3) x b(3x2):");
    let c = a.matmul(&b);
    c.print();

    println!("\nTranspose of a:");
    a.transpose().print();

    // CrossEntropyLoss: exemplo simples de cálculo de perda (cross-entropy)
    let logits = Tensor::from_values(&[2.0, 0.5, 0.1, 0.2, 1.5, 0.3], 3, 2); // 3 samples, 2 classes
    let targets = Tensor::from_values(&[1.0, 0.0, 1.0], 3, 1); // class indices or one-hot depending on impl
    println!("\nCrossEntropyLoss(example):");
    let cel = logits.cross_entropy_loss(&targets);
    cel.print();

    // ==================== TENSORFLOW - FLOW TENSORS ====================
    // Demonstração das operações disponíveis no wrapper do TensorFlow
    // `FlowTensors` é a representação leve de tensores do backend TensorFlow.
    println!("\n6. TENSORFLOW - FLOW TENSORS (Exemplos de ops)");
    println!("{}", "-".repeat(50));

    // Criar um FlowTensor manualmente: passa dados (f32) e as dimensões (i64)
    // Usamos `expect` para deixar claro falhas em exemplos (mensagem legível).
    let ft = FlowTensors::new(&[0.0_f32, 1.0, 2.0, 3.0], &[2, 2])
        .expect("falha ao criar FlowTensors");
    // `data()` retorna Option<&[f32]> — aqui usamos `expect` para extrair o slice
    // e mostrar os valores. Em código de produção prefira tratar Option/Result.
    println!(
        "FlowTensor original (como slice): {:?}",
        ft.data().expect("falha ao obter dados do FlowTensors")
    );

    let ft_exp = ft.exp().expect("exp failed");
    println!(
        "Exp(ft): {:?}",
        ft_exp.data().expect("failed to get FlowTensors data")
    );

    let ft_ln = ft_exp.ln().expect("ln failed");
    println!(
        "Ln(Exp(ft)) (should approx equal original): {:?}",
        ft_ln.data().expect("failed to get FlowTensors data")
    );

    // log1p: calcula log(1 + x) elemento-a-elemento
    let ft_log1p = ft.log1p().expect("log1p falhou");
    println!("Log1p(ft): {:?}", ft_log1p.data().expect("falha ao obter dados"));

    // Sigmoid e Tanh: funções de ativação comuns
    let ft_sig = ft.sigmoid().expect("sigmoid falhou");
    println!("Sigmoid(ft): {:?}", ft_sig.data().expect("falha ao obter dados"));

    let ft_tanh = ft.tanh().expect("tanh falhou");
    println!("Tanh(ft): {:?}", ft_tanh.data().expect("falha ao obter dados"));

    // Funções trigonométricas (seno e cosseno)
    println!(
        "Sin(ft): {:?}",
        ft.sin()
            .expect("sin falhou")
            .data()
            .expect("falha ao obter dados")
    );
    println!(
        "Cos(ft): {:?}",
        ft.cos()
            .expect("cos falhou")
            .data()
            .expect("falha ao obter dados")
    );

    // Comparações elemento-a-elemento
    // As funções retornam um tensor f32 contendo 1.0 para true e 0.0 para false
    let ft2 = FlowTensors::new(&[0.0_f32, 0.5, 2.0, 4.0], &[2, 2]).unwrap();
    println!(
        "ft > ft2 : {:?}",
        ft.greater(&ft2)
            .expect("greater failed")
            .data()
            .expect("failed to get FlowTensors data")
    );
    println!(
        "ft < ft2 : {:?}",
        ft.less(&ft2)
            .expect("less failed")
            .data()
            .expect("failed to get FlowTensors data")
    );
    println!(
        "ft == ft2: {:?}",
        ft.equal(&ft2)
            .expect("equal failed")
            .data()
            .expect("failed to get FlowTensors data")
    );
    println!(
        "ft != ft2: {:?}",
        ft.not_equal(&ft2)
            .expect("not_equal failed")
            .data()
            .expect("failed to get FlowTensors data")
    );


    // Mais funções trigonométricas e inversas
    println!(
        "Tan(ft): {:?}",
        ft.tan()
            .expect("tan falhou")
            .data()
            .expect("falha ao obter dados")
    );
    println!(
        "Asin(ft): {:?}",
        ft.asin()
            .expect("asin falhou")
            .data()
            .expect("falha ao obter dados")
    );
    println!(
        "Acos(ft): {:?}",
        ft.acos()
            .expect("acos falhou")
            .data()
            .expect("falha ao obter dados")
    );
    println!(
        "Atan(ft): {:?}",
        ft.atan()
            .expect("atan falhou")
            .data()
            .expect("falha ao obter dados")
    );

    // Operações aritméticas entre FlowTensors
    let ft3 = FlowTensors::new(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    println!(
        "ft + ft3: {:?}",
        ft.add(&ft3)
            .expect("add failed")
            .data()
            .expect("failed to get FlowTensors data")
    );
    println!(
        "ft - ft3: {:?}",
        ft.sub(&ft3)
            .expect("sub failed")
            .data()
            .expect("failed to get FlowTensors data")
    );
    println!(
        "ft * ft3: {:?}",
        ft.mul(&ft3)
            .expect("mul failed")
            .data()
            .expect("failed to get FlowTensors data")
    );
    println!(
        "ft / ft3: {:?}",
        ft.div(&ft3)
            .expect("div failed")
            .data()
            .expect("failed to get FlowTensors data")
    );


    // Exemplo grande para acionar paralelismo interno (CPU-only)
    // (útil para demonstrar o uso de rayon em operações element-wise)
    let large_vals = vec![0.1_f32; 5000];
    let large_ft = FlowTensors::new(&large_vals, &[5000]).unwrap();
    let large_exp = large_ft.exp().unwrap();
    println!(
        "Large exp first element: {:?}",
        large_exp.data().expect("failed to get FlowTensors data")[0]
    );
    
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
