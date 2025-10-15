use super::{Backend, Device};
use crate::tensor_libortch::tensor::Tensor as LibTorchTensor;
use crate::tensor_tensorflow::tensors_flow::FlowTensors;
use std::ops::{Add, Sub, Mul, Div};

/// Tensor unificado que pode usar LibTorch ou TensorFlow como backend
pub enum UnifiedTensor {
    LibTorch(LibTorchTensor),
    TensorFlow(FlowTensors),
}

impl UnifiedTensor {
    /// Cria um tensor de zeros
    pub fn zeros(rows: i32, cols: i32, backend: Backend, _device: Device) -> Self {
        match backend {
            Backend::LibTorch => {
                let values = vec![0.0f32; (rows * cols) as usize];
                UnifiedTensor::LibTorch(LibTorchTensor::from_values(&values, rows, cols))
            }
            Backend::TensorFlow => {
                let values = vec![0.0f32; (rows * cols) as usize];
                let dims = vec![rows as i64, cols as i64];
                UnifiedTensor::TensorFlow(
                    FlowTensors::new(&values, &dims)
                        .expect("Failed to create TensorFlow tensor")
                )
            }
        }
    }

    /// Cria um tensor de uns
    pub fn ones(rows: i32, cols: i32, backend: Backend, _device: Device) -> Self {
        match backend {
            Backend::LibTorch => {
                UnifiedTensor::LibTorch(LibTorchTensor::ones(rows, cols))
            }
            Backend::TensorFlow => {
                let values = vec![1.0f32; (rows * cols) as usize];
                let dims = vec![rows as i64, cols as i64];
                UnifiedTensor::TensorFlow(
                    FlowTensors::new(&values, &dims)
                        .expect("Failed to create TensorFlow tensor")
                )
            }
        }
    }

    /// Cria um tensor com valores aleatórios
    pub fn rand(rows: i32, cols: i32, backend: Backend, _device: Device) -> Self {
        match backend {
            Backend::LibTorch => {
                UnifiedTensor::LibTorch(LibTorchTensor::rand(rows, cols))
            }
            Backend::TensorFlow => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let values: Vec<f32> = (0..(rows * cols))
                    .map(|_| rng.gen::<f32>())
                    .collect();
                let dims = vec![rows as i64, cols as i64];
                UnifiedTensor::TensorFlow(
                    FlowTensors::new(&values, &dims)
                        .expect("Failed to create TensorFlow tensor")
                )
            }
        }
    }

    /// Cria um tensor a partir de valores
    pub fn from_values(values: &[f32], rows: i32, cols: i32, backend: Backend, _device: Device) -> Self {
        match backend {
            Backend::LibTorch => {
                UnifiedTensor::LibTorch(LibTorchTensor::from_values(values, rows, cols))
            }
            Backend::TensorFlow => {
                let dims = vec![rows as i64, cols as i64];
                UnifiedTensor::TensorFlow(
                    FlowTensors::new(values, &dims)
                        .expect("Failed to create TensorFlow tensor")
                )
            }
        }
    }

    /// Retorna os dados do tensor como slice
    pub fn as_slice(&self) -> &[f32] {
        match self {
            UnifiedTensor::LibTorch(t) => t.as_slice(),
            UnifiedTensor::TensorFlow(t) => {
                t.data().expect("Failed to get tensor data")
            }
        }
    }

    /// Imprime o tensor
    pub fn print(&self) {
        match self {
            UnifiedTensor::LibTorch(t) => t.print(),
            UnifiedTensor::TensorFlow(t) => {
                println!("TensorFlow Tensor {:?}", t.dims());
                let data = t.data().expect("Failed to get tensor data");
                let dims = t.dims();
                if dims.len() == 2 {
                    for r in 0..dims[0] {
                        for c in 0..dims[1] {
                            let idx = (r * dims[1] + c) as usize;
                            print!("{:.4} ", data[idx]);
                        }
                        println!();
                    }
                } else {
                    println!("{:?}", data);
                }
            }
        }
    }

    /// Retorna o backend usado
    pub fn backend(&self) -> Backend {
        match self {
            UnifiedTensor::LibTorch(_) => Backend::LibTorch,
            UnifiedTensor::TensorFlow(_) => Backend::TensorFlow,
        }
    }

    /// Retorna as dimensões do tensor (rows, cols)
    pub fn shape(&self) -> (i32, i32) {
        match self {
            UnifiedTensor::LibTorch(t) => (t.rows, t.cols),
            UnifiedTensor::TensorFlow(t) => {
                let dims = t.dims();
                if dims.len() == 2 {
                    (dims[0] as i32, dims[1] as i32)
                } else {
                    (1, dims.iter().product::<i64>() as i32)
                }
            }
        }
    }

    /// Converte o tensor para outro backend
    pub fn to_backend(&self, target_backend: Backend) -> Self {
        if self.backend() == target_backend {
            return match self {
                UnifiedTensor::LibTorch(t) => UnifiedTensor::LibTorch(t.clone()),
                UnifiedTensor::TensorFlow(_) => {
                    // TensorFlow tensors não implementam Clone facilmente, então recriamos
                    let data = self.as_slice().to_vec();
                    let (rows, cols) = self.shape();
                    UnifiedTensor::from_values(&data, rows, cols, target_backend, Device::CPU)
                }
            };
        }

        let data = self.as_slice().to_vec();
        let (rows, cols) = self.shape();
        UnifiedTensor::from_values(&data, rows, cols, target_backend, Device::CPU)
    }

    /// Calcula a soma de todos os elementos
    pub fn sum(&self) -> f32 {
        self.as_slice().iter().sum()
    }

    /// Calcula a média de todos os elementos
    pub fn mean(&self) -> f32 {
        let data = self.as_slice();
        let sum: f32 = data.iter().sum();
        sum / data.len() as f32
    }

    /// Calcula o valor máximo
    pub fn max(&self) -> f32 {
        self.as_slice().iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Calcula o valor mínimo
    pub fn min(&self) -> f32 {
        self.as_slice().iter()
            .cloned()
            .fold(f32::INFINITY, f32::min)
    }

    /// Transpõe o tensor (apenas para matrizes 2D)
    pub fn transpose(&self) -> Self {
        let (rows, cols) = self.shape();
        let data = self.as_slice();
        let mut transposed = vec![0.0f32; (rows * cols) as usize];
        
        for r in 0..rows {
            for c in 0..cols {
                let src_idx = (r * cols + c) as usize;
                let dst_idx = (c * rows + r) as usize;
                transposed[dst_idx] = data[src_idx];
            }
        }

        UnifiedTensor::from_values(&transposed, cols, rows, self.backend(), Device::CPU)
    }

    /// Aplica uma função a cada elemento
    pub fn map<F>(&self, f: F) -> Self 
    where
        F: Fn(f32) -> f32,
    {
        let data: Vec<f32> = self.as_slice().iter().map(|&x| f(x)).collect();
        let (rows, cols) = self.shape();
        UnifiedTensor::from_values(&data, rows, cols, self.backend(), Device::CPU)
    }

    /// Retorna uma cópia do tensor
    pub fn clone_tensor(&self) -> Self {
        let data = self.as_slice().to_vec();
        let (rows, cols) = self.shape();
        UnifiedTensor::from_values(&data, rows, cols, self.backend(), Device::CPU)
    }

    // ==================== ACTIVATION FUNCTIONS ====================
    
    /// Aplica ReLU (apenas LibTorch)
    pub fn relu(&self) -> Self {
        match self {
            UnifiedTensor::LibTorch(t) => UnifiedTensor::LibTorch(t.relu()),
            UnifiedTensor::TensorFlow(_) => {
                // Implementação manual para TensorFlow
                self.map(|x| if x > 0.0 { x } else { 0.0 })
            }
        }
    }

    /// Aplica Sigmoid (apenas LibTorch)
    pub fn sigmoid(&self) -> Self {
        match self {
            UnifiedTensor::LibTorch(t) => UnifiedTensor::LibTorch(t.sigmoid()),
            UnifiedTensor::TensorFlow(_) => {
                // Implementação manual para TensorFlow
                self.map(|x| 1.0 / (1.0 + (-x).exp()))
            }
        }
    }

    /// Aplica Tanh (apenas LibTorch)
    pub fn tanh(&self) -> Self {
        match self {
            UnifiedTensor::LibTorch(t) => UnifiedTensor::LibTorch(t.tanh()),
            UnifiedTensor::TensorFlow(_) => {
                // Implementação manual para TensorFlow
                self.map(|x| x.tanh())
            }
        }
    }

    // ==================== MATHEMATICAL FUNCTIONS ====================
    
    /// Aplica seno elemento a elemento
    pub fn sin(&self) -> Self {
        match self {
            UnifiedTensor::LibTorch(t) => UnifiedTensor::LibTorch(t.sin()),
            UnifiedTensor::TensorFlow(_) => self.map(|x| x.sin()),
        }
    }

    /// Aplica cosseno elemento a elemento
    pub fn cos(&self) -> Self {
        match self {
            UnifiedTensor::LibTorch(t) => UnifiedTensor::LibTorch(t.cos()),
            UnifiedTensor::TensorFlow(_) => self.map(|x| x.cos()),
        }
    }

    /// Aplica exponencial elemento a elemento
    pub fn exp(&self) -> Self {
        match self {
            UnifiedTensor::LibTorch(t) => UnifiedTensor::LibTorch(t.exp()),
            UnifiedTensor::TensorFlow(_) => self.map(|x| x.exp()),
        }
    }

    /// Aplica logaritmo natural elemento a elemento
    pub fn log(&self) -> Self {
        match self {
            UnifiedTensor::LibTorch(t) => UnifiedTensor::LibTorch(t.log()),
            UnifiedTensor::TensorFlow(_) => self.map(|x| x.ln()),
        }
    }

    /// Aplica raiz quadrada elemento a elemento
    pub fn sqrt(&self) -> Self {
        match self {
            UnifiedTensor::LibTorch(t) => UnifiedTensor::LibTorch(t.sqrt()),
            UnifiedTensor::TensorFlow(_) => self.map(|x| x.sqrt()),
        }
    }

    /// Aplica valor absoluto elemento a elemento
    pub fn abs(&self) -> Self {
        match self {
            UnifiedTensor::LibTorch(t) => UnifiedTensor::LibTorch(t.abs()),
            UnifiedTensor::TensorFlow(_) => self.map(|x| x.abs()),
        }
    }

    /// Eleva cada elemento a uma potência
    pub fn pow(&self, exponent: f32) -> Self {
        match self {
            UnifiedTensor::LibTorch(t) => UnifiedTensor::LibTorch(t.pow(exponent)),
            UnifiedTensor::TensorFlow(_) => self.map(|x| x.powf(exponent)),
        }
    }

    // ==================== TENSOR CREATION ====================
    
    /// Cria tensor com distribuição normal
    pub fn randn(rows: i32, cols: i32, backend: Backend, _device: Device) -> Self {
        match backend {
            Backend::LibTorch => UnifiedTensor::LibTorch(LibTorchTensor::randn(rows, cols)),
            Backend::TensorFlow => {
                use rand::Rng;
                use rand_distr::{Normal, Distribution};
                let normal = Normal::new(0.0, 1.0).unwrap();
                let mut rng = rand::thread_rng();
                let values: Vec<f32> = (0..(rows * cols))
                    .map(|_| normal.sample(&mut rng))
                    .collect();
                let dims = vec![rows as i64, cols as i64];
                UnifiedTensor::TensorFlow(
                    FlowTensors::new(&values, &dims).expect("Failed to create TensorFlow tensor")
                )
            }
        }
    }

    /// Cria matriz identidade
    pub fn eye(size: i32, backend: Backend, _device: Device) -> Self {
        match backend {
            Backend::LibTorch => UnifiedTensor::LibTorch(LibTorchTensor::eye(size)),
            Backend::TensorFlow => {
                let mut values = vec![0.0f32; (size * size) as usize];
                for i in 0..size {
                    values[(i * size + i) as usize] = 1.0;
                }
                let dims = vec![size as i64, size as i64];
                UnifiedTensor::TensorFlow(
                    FlowTensors::new(&values, &dims).expect("Failed to create TensorFlow tensor")
                )
            }
        }
    }

    /// Cria tensor de zeros com a mesma forma
    pub fn zeros_like(&self) -> Self {
        match self {
            UnifiedTensor::LibTorch(t) => UnifiedTensor::LibTorch(t.zeros_like()),
            UnifiedTensor::TensorFlow(_) => {
                let (rows, cols) = self.shape();
                UnifiedTensor::zeros(rows, cols, Backend::TensorFlow, Device::CPU)
            }
        }
    }

    /// Cria tensor de ones com a mesma forma
    pub fn ones_like(&self) -> Self {
        match self {
            UnifiedTensor::LibTorch(t) => UnifiedTensor::LibTorch(t.ones_like()),
            UnifiedTensor::TensorFlow(_) => {
                let (rows, cols) = self.shape();
                UnifiedTensor::ones(rows, cols, Backend::TensorFlow, Device::CPU)
            }
        }
    }

    // ==================== STATISTICAL FUNCTIONS ====================
    
    /// Calcula o desvio padrão
    pub fn std(&self) -> f32 {
        match self {
            UnifiedTensor::LibTorch(t) => t.std(),
            UnifiedTensor::TensorFlow(_) => {
                let mean = self.mean();
                let variance = self.as_slice()
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>() / self.as_slice().len() as f32;
                variance.sqrt()
            }
        }
    }

    /// Calcula a variância
    pub fn var(&self) -> f32 {
        match self {
            UnifiedTensor::LibTorch(t) => t.var(),
            UnifiedTensor::TensorFlow(_) => {
                let mean = self.mean();
                self.as_slice()
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>() / self.as_slice().len() as f32
            }
        }
    }

    /// Retorna o índice do valor máximo
    pub fn argmax(&self) -> i32 {
        match self {
            UnifiedTensor::LibTorch(t) => t.argmax(),
            UnifiedTensor::TensorFlow(_) => {
                let data = self.as_slice();
                let (max_idx, _) = data.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                max_idx as i32
            }
        }
    }

    /// Retorna o índice do valor mínimo
    pub fn argmin(&self) -> i32 {
        match self {
            UnifiedTensor::LibTorch(t) => t.argmin(),
            UnifiedTensor::TensorFlow(_) => {
                let data = self.as_slice();
                let (min_idx, _) = data.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                min_idx as i32
            }
        }
    }
}

// Implementação de operadores aritméticos
impl Add for UnifiedTensor {
    type Output = UnifiedTensor;

    fn add(self, other: UnifiedTensor) -> UnifiedTensor {
        let (rows1, cols1) = self.shape();
        let (rows2, cols2) = other.shape();

        if rows1 != rows2 || cols1 != cols2 {
            panic!("Cannot add tensors with different dimensions!");
        }

        let result: Vec<f32> = self.as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(a, b)| a + b)
            .collect();

        UnifiedTensor::from_values(&result, rows1, cols1, self.backend(), Device::CPU)
    }
}

impl Sub for UnifiedTensor {
    type Output = UnifiedTensor;

    fn sub(self, other: UnifiedTensor) -> UnifiedTensor {
        let (rows1, cols1) = self.shape();
        let (rows2, cols2) = other.shape();

        if rows1 != rows2 || cols1 != cols2 {
            panic!("Cannot subtract tensors with different dimensions!");
        }

        let result: Vec<f32> = self.as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(a, b)| a - b)
            .collect();

        UnifiedTensor::from_values(&result, rows1, cols1, self.backend(), Device::CPU)
    }
}

impl Mul for UnifiedTensor {
    type Output = UnifiedTensor;

    fn mul(self, other: UnifiedTensor) -> UnifiedTensor {
        let (rows1, cols1) = self.shape();
        let (rows2, cols2) = other.shape();

        if rows1 != rows2 || cols1 != cols2 {
            panic!("Cannot multiply tensors with different dimensions!");
        }

        let result: Vec<f32> = self.as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(a, b)| a * b)
            .collect();

        UnifiedTensor::from_values(&result, rows1, cols1, self.backend(), Device::CPU)
    }
}

impl Div for UnifiedTensor {
    type Output = UnifiedTensor;

    fn div(self, other: UnifiedTensor) -> UnifiedTensor {
        let (rows1, cols1) = self.shape();
        let (rows2, cols2) = other.shape();

        if rows1 != rows2 || cols1 != cols2 {
            panic!("Cannot divide tensors with different dimensions!");
        }

        let result: Vec<f32> = self.as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(a, b)| {
                if *b == 0.0 {
                    panic!("Division by zero!");
                }
                a / b
            })
            .collect();

        UnifiedTensor::from_values(&result, rows1, cols1, self.backend(), Device::CPU)
    }
}
