use libc::c_void;

#[link(name = "ai_copper")]
extern "C" {
    fn CreateTensor(rows: i32, cols: i32) -> *mut c_void;
    fn FreeTensor(ptr: *mut c_void);
    fn TensorData(ptr: *mut c_void) -> *const f32;
    fn TensorRows(ptr: *mut c_void) -> i32;
    fn TensorCols(ptr: *mut c_void) -> i32;
}

pub struct Tensor {
    ptr: *mut c_void,
    rows: i32,
    cols: i32,
}

impl Tensor {
    pub fn new(rows: i32, cols: i32) -> Self {
        let ptr = unsafe { CreateTensor(rows, cols) };
        if ptr.is_null() {
            panic!("Falha ao criar tensor");
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

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe {
            FreeTensor(self.ptr);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::new(4, 4);
        assert!(tensor.is_valid(), "O tensor deveria ser válido");
    }

    #[test]
    #[should_panic]
    fn test_tensor_creation_invalid() {
        let _ = Tensor::new(-1, 3);
    }
}