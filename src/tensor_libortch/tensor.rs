use crate::tensor_libortch::ffi::{
    CreateMatrixTensor, 
    CreateTensorOnes, 
    CreateTensorRand,
    CreateTensorZeros,
    CreateTensorRandn,
    CreateTensorEye,
    TensorZerosLike,
    TensorOnesLike,
    TensorData, 
    TensorRows, 
    TensorCols, 
    FreeTensor,
    TensorReLU,
    TensorSigmoid,
    TensorTanh,
    TensorSin,
    TensorCos,
    TensorExp,
    TensorLog,
    TensorSqrt,
    TensorAbs,
    TensorPow,
    TensorStd,
    TensorVar,
    TensorArgmax,
    TensorArgmin,
    CreateLinear,
    LinearForward,
    MSELoss,
    CrossEntropyLoss,
    CreateSGD,
    CreateAdam,
    Backward,
    OptimizerStep,
    OptimizerZeroGrad,
    FreeOptimizer,
};

pub struct Tensor {
    pub ptr: *mut libc::c_void,
    pub rows: i32,
    pub cols: i32,
}

impl Tensor {
    pub fn ones(rows: i32, cols: i32) -> Self {
        let ptr: *mut libc::c_void = unsafe { CreateTensorOnes(rows, cols) };
        if ptr.is_null() {
            panic!("Error creating tensor");
        }

        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };

        Tensor { ptr, rows, cols }
    }

    pub fn from_values(values: &[f32], rows: i32, cols: i32) -> Self {
        let ptr: *mut libc::c_void = unsafe { CreateMatrixTensor(values.as_ptr(), rows, cols) };
        if ptr.is_null() {
            panic!("Error creating tensor from values");
        }

        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };

        Tensor { ptr, rows, cols }
    }

    pub fn rand(rows: i32, cols: i32) -> Self {
        let ptr = unsafe { CreateTensorRand(rows, cols) };
        if ptr.is_null() {
            panic!("Error creating tensor");
        }

        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };

        Tensor { ptr, rows, cols }
    }

    pub fn as_slice(&self) -> &[f32] {
        let total = (self.rows * self.cols) as usize;
        unsafe {
            let data_ptr = TensorData(self.ptr);
            std::slice::from_raw_parts(data_ptr, total)
        }
    }

    pub fn print(&self) {
        println!("Variable[CPUFloatType {{{}, {}}}]", self.rows, self.cols);
        let slice = self.as_slice();
        for r in 0..self.rows {
            for c in 0..self.cols {
                let val = slice[(r * self.cols + c) as usize];
                print!("{:.4} ", val);
            }
            println!();
        }
    }

    pub fn mse_loss(&self, target: &Tensor) -> Tensor {
        let loss_ptr = unsafe { MSELoss(self.ptr, target.ptr) };
        if loss_ptr.is_null() {
            panic!("Error calculating MSELoss");
        }
        let rows = unsafe { TensorRows(loss_ptr) };
        let cols = unsafe { TensorCols(loss_ptr) };
        Tensor { ptr: loss_ptr, rows, cols }
    }

    pub fn backward(&self) {
        unsafe { Backward(self.ptr) };
    }

    /// Calcula a soma de todos os elementos
    pub fn sum(&self) -> f32 {
        self.as_slice().iter().sum()
    }

    /// Calcula a média de todos os elementos
    pub fn mean(&self) -> f32 {
        let sum: f32 = self.as_slice().iter().sum();
        sum / (self.rows * self.cols) as f32
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

    /// Transpõe o tensor
    pub fn transpose(&self) -> Tensor {
        let data = self.as_slice();
        let mut transposed = vec![0.0f32; (self.rows * self.cols) as usize];
        
        for r in 0..self.rows {
            for c in 0..self.cols {
                let src_idx = (r * self.cols + c) as usize;
                let dst_idx = (c * self.rows + r) as usize;
                transposed[dst_idx] = data[src_idx];
            }
        }

        Tensor::from_values(&transposed, self.cols, self.rows)
    }

    /// Aplica uma função a cada elemento
    pub fn map<F>(&self, f: F) -> Tensor 
    where
        F: Fn(f32) -> f32,
    {
        let data: Vec<f32> = self.as_slice().iter().map(|&x| f(x)).collect();
        Tensor::from_values(&data, self.rows, self.cols)
    }

    /// Cria tensor de zeros
    pub fn zeros(rows: i32, cols: i32) -> Self {
        let values = vec![0.0f32; (rows * cols) as usize];
        Tensor::from_values(&values, rows, cols)
    }

    /// Reshape do tensor
    pub fn reshape(&self, new_rows: i32, new_cols: i32) -> Tensor {
        if self.rows * self.cols != new_rows * new_cols {
            panic!("Cannot reshape: total elements must remain the same");
        }
        let data = self.as_slice().to_vec();
        Tensor::from_values(&data, new_rows, new_cols)
    }

    /// Multiplicação de matrizes
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        if self.cols != other.rows {
            panic!("Invalid dimensions for matrix multiplication: ({}, {}) x ({}, {})", 
                   self.rows, self.cols, other.rows, other.cols);
        }

        let mut result = vec![0.0f32; (self.rows * other.cols) as usize];
        let a = self.as_slice();
        let b = other.as_slice();

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    let a_idx = (i * self.cols + k) as usize;
                    let b_idx = (k * other.cols + j) as usize;
                    sum += a[a_idx] * b[b_idx];
                }
                let result_idx = (i * other.cols + j) as usize;
                result[result_idx] = sum;
            }
        }

        Tensor::from_values(&result, self.rows, other.cols)
    }

    // ==================== ACTIVATION FUNCTIONS ====================
    
    /// Aplica a função de ativação ReLU (Rectified Linear Unit)
    /// ReLU(x) = max(0, x)
    pub fn relu(&self) -> Tensor {
        let ptr = unsafe { TensorReLU(self.ptr) };
        if ptr.is_null() {
            panic!("Error applying ReLU");
        }
        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };
        Tensor { ptr, rows, cols }
    }

    /// Aplica a função de ativação Sigmoid
    /// Sigmoid(x) = 1 / (1 + e^(-x))
    pub fn sigmoid(&self) -> Tensor {
        let ptr = unsafe { TensorSigmoid(self.ptr) };
        if ptr.is_null() {
            panic!("Error applying Sigmoid");
        }
        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };
        Tensor { ptr, rows, cols }
    }

    /// Aplica a função de ativação Tanh (Tangente Hiperbólica)
    /// Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    pub fn tanh(&self) -> Tensor {
        let ptr = unsafe { TensorTanh(self.ptr) };
        if ptr.is_null() {
            panic!("Error applying Tanh");
        }
        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };
        Tensor { ptr, rows, cols }
    }

    // ==================== MATHEMATICAL FUNCTIONS ====================
    
    /// Aplica a função seno elemento a elemento
    pub fn sin(&self) -> Tensor {
        let ptr = unsafe { TensorSin(self.ptr) };
        if ptr.is_null() {
            panic!("Error applying sin");
        }
        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };
        Tensor { ptr, rows, cols }
    }

    /// Aplica a função cosseno elemento a elemento
    pub fn cos(&self) -> Tensor {
        let ptr = unsafe { TensorCos(self.ptr) };
        if ptr.is_null() {
            panic!("Error applying cos");
        }
        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };
        Tensor { ptr, rows, cols }
    }

    /// Aplica a função exponencial (e^x) elemento a elemento
    pub fn exp(&self) -> Tensor {
        let ptr = unsafe { TensorExp(self.ptr) };
        if ptr.is_null() {
            panic!("Error applying exp");
        }
        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };
        Tensor { ptr, rows, cols }
    }

    /// Aplica a função logaritmo natural elemento a elemento
    pub fn log(&self) -> Tensor {
        let ptr = unsafe { TensorLog(self.ptr) };
        if ptr.is_null() {
            panic!("Error applying log");
        }
        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };
        Tensor { ptr, rows, cols }
    }

    /// Aplica a função raiz quadrada elemento a elemento
    pub fn sqrt(&self) -> Tensor {
        let ptr = unsafe { TensorSqrt(self.ptr) };
        if ptr.is_null() {
            panic!("Error applying sqrt");
        }
        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };
        Tensor { ptr, rows, cols }
    }

    /// Aplica a função valor absoluto elemento a elemento
    pub fn abs(&self) -> Tensor {
        let ptr = unsafe { TensorAbs(self.ptr) };
        if ptr.is_null() {
            panic!("Error applying abs");
        }
        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };
        Tensor { ptr, rows, cols }
    }

    /// Eleva cada elemento a uma potência
    pub fn pow(&self, exponent: f32) -> Tensor {
        let ptr = unsafe { TensorPow(self.ptr, exponent) };
        if ptr.is_null() {
            panic!("Error applying pow");
        }
        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };
        Tensor { ptr, rows, cols }
    }

    // ==================== TENSOR CREATION (STATIC METHODS) ====================
    
    /// Cria tensor com distribuição normal (média 0, desvio padrão 1)
    pub fn randn(rows: i32, cols: i32) -> Self {
        let ptr = unsafe { CreateTensorRandn(rows, cols) };
        if ptr.is_null() {
            panic!("Error creating randn tensor");
        }
        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };
        Tensor { ptr, rows, cols }
    }

    /// Cria matriz identidade (diagonal com 1s)
    pub fn eye(size: i32) -> Self {
        let ptr = unsafe { CreateTensorEye(size) };
        if ptr.is_null() {
            panic!("Error creating eye tensor");
        }
        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };
        Tensor { ptr, rows, cols }
    }

    /// Cria tensor de zeros com a mesma forma que outro tensor
    pub fn zeros_like(&self) -> Tensor {
        let ptr = unsafe { TensorZerosLike(self.ptr) };
        if ptr.is_null() {
            panic!("Error creating zeros_like tensor");
        }
        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };
        Tensor { ptr, rows, cols }
    }

    /// Cria tensor de ones com a mesma forma que outro tensor
    pub fn ones_like(&self) -> Tensor {
        let ptr = unsafe { TensorOnesLike(self.ptr) };
        if ptr.is_null() {
            panic!("Error creating ones_like tensor");
        }
        let rows = unsafe { TensorRows(ptr) };
        let cols = unsafe { TensorCols(ptr) };
        Tensor { ptr, rows, cols }
    }

    // ==================== STATISTICAL FUNCTIONS ====================
    
    /// Calcula o desvio padrão de todos os elementos
    pub fn std(&self) -> f32 {
        unsafe { TensorStd(self.ptr) }
    }

    /// Calcula a variância de todos os elementos
    pub fn var(&self) -> f32 {
        unsafe { TensorVar(self.ptr) }
    }

    /// Retorna o índice do valor máximo (flatten)
    pub fn argmax(&self) -> i32 {
        unsafe { TensorArgmax(self.ptr) }
    }

    /// Retorna o índice do valor mínimo (flatten)
    pub fn argmin(&self) -> i32 {
        unsafe { TensorArgmin(self.ptr) }
    }

    // ==================== LOSS FUNCTIONS ====================
    
    /// Calcula Cross Entropy Loss para classificação
    pub fn cross_entropy_loss(&self, target: &Tensor) -> Tensor {
        let loss_ptr = unsafe { CrossEntropyLoss(self.ptr, target.ptr) };
        if loss_ptr.is_null() {
            panic!("Error calculating CrossEntropyLoss");
        }
        let rows = unsafe { TensorRows(loss_ptr) };
        let cols = unsafe { TensorCols(loss_ptr) };
        Tensor { ptr: loss_ptr, rows, cols }
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        let values = self.as_slice().to_vec();
        Tensor::from_values(&values, self.rows, self.cols)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe {
            FreeTensor(self.ptr);
        }
    }
}

pub struct Linear {
    pub ptr: *mut libc::c_void,
    pub in_features: i32,
    pub out_features: i32,
}

impl Linear {
    pub fn new(in_features: i32, out_features: i32) -> Self {
        let ptr = unsafe { CreateLinear(in_features, out_features) };
        if ptr.is_null() {
            panic!("Error creating Linear layer");
        }
        Linear { ptr, in_features, out_features }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let out_ptr = unsafe { LinearForward(self.ptr, input.ptr) };
        if out_ptr.is_null() {
            panic!("Error in Linear forward");
        }
        let rows = unsafe { TensorRows(out_ptr) };
        let cols = unsafe { TensorCols(out_ptr) };
        Tensor { ptr: out_ptr, rows, cols }
    }
}

impl Drop for Linear {
    fn drop(&mut self) {
        // Optionally implement a FreeLinear if you add it to the C++ side
    }
}

pub struct Optimizer {
    pub ptr: *mut libc::c_void,
}

impl Optimizer {
    pub fn sgd(linear: &Linear, lr: f32) -> Self {
        let ptr = unsafe { CreateSGD(linear.ptr, lr) };
        if ptr.is_null() {
            panic!("Error creating SGD optimizer");
        }
        Optimizer { ptr }
    }

    pub fn adam(linear: &Linear, lr: f32) -> Self {
        let ptr = unsafe { CreateAdam(linear.ptr, lr) };
        if ptr.is_null() {
            panic!("Error creating Adam optimizer");
        }
        Optimizer { ptr }
    }

    pub fn step(&self) {
        unsafe { OptimizerStep(self.ptr) };
    }

    pub fn zero_grad(&self) {
        unsafe { OptimizerZeroGrad(self.ptr) };
    }
}

impl Drop for Optimizer {
    fn drop(&mut self) {
        unsafe { FreeOptimizer(self.ptr) };
    }
}