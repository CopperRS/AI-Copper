use crate::tensor::ffi::{
    CreateMatrixTensor, 
    CreateTensorOnes, 
    CreateTensorRand, 
    TensorData, 
    TensorRows, 
    TensorCols, 
    FreeTensor,
    CreateLinear,
    LinearForward,
    MSELoss,
    CreateSGD,
    Backward,
    OptimizerStep,
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

    pub fn step(&self) {
        unsafe { OptimizerStep(self.ptr) };
    }
}

impl Drop for Optimizer {
    fn drop(&mut self) {
        unsafe { FreeOptimizer(self.ptr) };
    }
}