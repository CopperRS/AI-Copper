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
    pub fn CreateLinear(in_features: i32, out_features: i32) -> *mut c_void;
    pub fn LinearForward(linear_ptr: *mut c_void, input_tensor_ptr: *mut c_void) -> *mut c_void;
    pub fn MSELoss(prediction_ptr: *mut c_void, target_ptr: *mut c_void) -> *mut c_void;
    pub fn CreateSGD(linear_ptr: *mut c_void, lr: f32) -> *mut c_void;
    pub fn Backward(loss_ptr: *mut c_void);
    pub fn OptimizerStep(optimizer_ptr: *mut c_void);
    pub fn FreeOptimizer(ptr: *mut c_void);
}