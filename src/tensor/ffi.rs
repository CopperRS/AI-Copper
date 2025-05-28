use libc::c_void;

#[link(name = "ai_copper", kind = "dylib")]
extern "C" {
    pub fn CreateMatrixTensor(values: *const f32, rows: i32, cols: i32) -> *mut c_void;
    pub fn CreateTensorOnes(rows: i32, cols: i32) -> *mut c_void;
    pub fn CreateTensorRand(rows: i32, cols: i32) -> *mut c_void;
    pub fn FreeTensor(ptr: *mut c_void);
    pub fn TensorData(ptr: *mut c_void) -> *const f32;
    pub fn TensorRows(ptr: *mut c_void) -> i32;
    pub fn TensorCols(ptr: *mut c_void) -> i32;
}