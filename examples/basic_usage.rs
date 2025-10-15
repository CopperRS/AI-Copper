//! Exemplos de uso da biblioteca AI Copper
//! 
//! Esta biblioteca unifica funcionalidades do LibTorch e TensorFlow
//! em uma API Rust única e conveniente.

use ai_copper::{
    // Módulos do LibTorch
    Tensor, Linear, Optimizer,
    
    // Módulos do TensorFlow
    FlowTensors, TensorFlowModel,
    
    // API Unificada
    UnifiedTensor, Backend, Device,
};

fn main() {
    println!("=== AI Copper - Exemplos ===\n");
    
    // Exemplo 1: Usando LibTorch diretamente
    exemplo_libtorch();
    
    // Exemplo 2: Usando TensorFlow diretamente
    exemplo_tensorflow();
    
    // Exemplo 3: Usando a API Unificada
    exemplo_unificado();
    
    // Exemplo 4: Treinamento com LibTorch
    exemplo_treinamento();
}

fn exemplo_libtorch() {
    println!("--- Exemplo 1: LibTorch ---");
    
    // Criar tensores
    let t1 = Tensor::ones(2, 3);
    let t2 = Tensor::rand(2, 3);
    let t3 = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    
    println!("Tensor de uns:");
    t1.print();
    
    println!("\nTensor aleatório:");
    t2.print();
    
    println!("\nTensor de valores:");
    t3.print();
    
    // Operações
    println!("\nSoma: {}", t1.sum());
    println!("Média: {}", t1.mean());
    println!("Máximo: {}", t2.max());
    println!("Mínimo: {}", t2.min());
    
    // Operações aritméticas
    let t4 = t1.clone() + t2.clone();
    println!("\nSoma de tensores:");
    t4.print();
    
    // Transposta
    println!("\nTransposta de t3:");
    let t3_t = t3.transpose();
    t3_t.print();
    
    // Map (aplicar função)
    let t5 = t3.map(|x| x * 2.0);
    println!("\nTensor multiplicado por 2:");
    t5.print();
    
    // Multiplicação de matrizes
    let a = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Tensor::from_values(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = a.matmul(&b);
    println!("\nMultiplicação de matrizes:");
    c.print();
    
    println!();
}

fn exemplo_tensorflow() {
    println!("--- Exemplo 2: TensorFlow ---");
    
    // Versão do TensorFlow
    println!("Versão do TensorFlow: {}", FlowTensors::version_tf());
    
    // Criar tensores
    let t1 = FlowTensors::ones(&[2, 3]).expect("Failed to create tensor");
    let t2 = FlowTensors::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("Failed to create tensor");
    
    println!("\nTensor de uns: {:?}", t1.dims());
    println!("Dados: {:?}", t1.data());
    
    println!("\nTensor de valores: {:?}", t2.dims());
    println!("Dados: {:?}", t2.data());
    
    // Operações
    println!("\nSoma: {}", t1.sum());
    println!("Média: {}", t1.mean());
    println!("Máximo: {}", t2.max());
    println!("Mínimo: {}", t2.min());
    
    // Transposta
    if let Some(t2_t) = t2.transpose() {
        println!("\nTransposta: {:?}", t2_t.dims());
        println!("Dados: {:?}", t2_t.data());
    }
    
    // Map
    if let Some(t3) = t2.map(|x| x * 2.0) {
        println!("\nTensor multiplicado por 2:");
        println!("Dados: {:?}", t3.data());
    }
    
    // Reshape
    if let Some(reshaped) = t2.reshape(&[3, 2]) {
        println!("\nReshape de [2, 3] para [3, 2]:");
        println!("Dimensões: {:?}", reshaped.dims());
    }
    
    println!();
}

fn exemplo_unificado() {
    println!("--- Exemplo 3: API Unificada ---");
    
    // Criar tensores com diferentes backends
    let backend_torch = Backend::LibTorch;
    let backend_tf = Backend::TensorFlow;
    let device = Device::CPU;
    
    println!("Usando backend: {}", backend_torch);
    let t1 = UnifiedTensor::ones(2, 3, backend_torch, device);
    t1.print();
    
    println!("\nUsando backend: {}", backend_tf);
    let t2 = UnifiedTensor::rand(2, 3, backend_tf, device);
    t2.print();
    
    // Conversão entre backends
    println!("\nConvertendo TensorFlow para LibTorch:");
    let t3 = t2.to_backend(Backend::LibTorch);
    t3.print();
    println!("Backend atual: {}", t3.backend());
    
    // Operações unificadas
    let t4 = UnifiedTensor::from_values(&[1.0, 2.0, 3.0, 4.0], 2, 2, backend_torch, device);
    let t5 = UnifiedTensor::from_values(&[5.0, 6.0, 7.0, 8.0], 2, 2, backend_torch, device);
    
    println!("\nSoma unificada:");
    let t6 = t4 + t5;
    t6.print();
    
    // Operações matemáticas
    println!("\nEstatísticas:");
    println!("Soma: {}", t6.sum());
    println!("Média: {}", t6.mean());
    println!("Máximo: {}", t6.max());
    println!("Mínimo: {}", t6.min());
    
    // Transposta
    println!("\nTransposta:");
    let t7 = t6.transpose();
    t7.print();
    
    println!();
}

fn exemplo_treinamento() {
    println!("--- Exemplo 4: Treinamento com LibTorch ---");
    
    // Dados de treinamento: y = 2*x + 1
    let x = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0], 4, 1);
    let y = Tensor::from_values(&[3.0, 5.0, 7.0, 9.0], 4, 1);
    
    // Criar modelo linear
    let linear = Linear::new(1, 1);
    let optimizer = Optimizer::sgd(&linear, 0.01);
    
    // Treinamento
    println!("Treinando modelo linear...");
    for epoch in 0..100 {
        // Forward pass
        let pred = linear.forward(&x);
        
        // Calcular loss
        let loss = pred.mse_loss(&y);
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss.as_slice()[0]);
        }
        
        // Backward pass
        loss.backward();
        
        // Atualizar pesos
        optimizer.step();
    }
    
    // Teste
    println!("\nTestando o modelo:");
    let test_x = Tensor::from_values(&[5.0, 6.0], 2, 1);
    let pred = linear.forward(&test_x);
    println!("Predições:");
    pred.print();
    println!("Esperado: ~11.0 e ~13.0");
    
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::ones(2, 3);
        assert_eq!(t.rows, 2);
        assert_eq!(t.cols, 3);
        assert_eq!(t.sum(), 6.0);
    }

    #[test]
    fn test_tensor_operations() {
        let t1 = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let t2 = Tensor::from_values(&[1.0, 1.0, 1.0, 1.0], 2, 2);
        let t3 = t1 + t2;
        
        assert_eq!(t3.as_slice()[0], 2.0);
        assert_eq!(t3.sum(), 14.0);
    }

    #[test]
    fn test_unified_tensor() {
        let t1 = UnifiedTensor::ones(2, 2, Backend::LibTorch, Device::CPU);
        let (rows, cols) = t1.shape();
        
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
        assert_eq!(t1.sum(), 4.0);
    }

    #[test]
    fn test_backend_conversion() {
        let t1 = UnifiedTensor::ones(2, 2, Backend::LibTorch, Device::CPU);
        let t2 = t1.to_backend(Backend::TensorFlow);
        
        assert_eq!(t2.backend(), Backend::TensorFlow);
        assert_eq!(t2.sum(), 4.0);
    }
}
