use crate::tensor::ffi::{
    CreateMatrixTensor, 
    CreateTensorOnes, 
    CreateTensorRand, 
    TensorData, 
    TensorRows, 
    TensorCols, 
    FreeTensor
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