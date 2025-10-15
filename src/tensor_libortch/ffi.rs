use libc::c_void;

#[link(name = "ai_copper_cpp", kind = "dylib")]
extern "C" {
    // Tensor Creation
    pub fn CreateMatrixTensor(values: *const f32, rows: i32, cols: i32) -> *mut c_void;
    pub fn CreateTensorOnes(rows: i32, cols: i32) -> *mut c_void;
    pub fn CreateTensorRand(rows: i32, cols: i32) -> *mut c_void;
    pub fn CreateTensorZeros(rows: i32, cols: i32) -> *mut c_void;
    pub fn CreateTensorRandn(rows: i32, cols: i32) -> *mut c_void;
    pub fn CreateTensorEye(size: i32) -> *mut c_void;
    pub fn TensorZerosLike(tensor_ptr: *mut c_void) -> *mut c_void;
    pub fn TensorOnesLike(tensor_ptr: *mut c_void) -> *mut c_void;
    
    // Tensor Operations
    pub fn FreeTensor(ptr: *mut c_void);
    pub fn TensorData(ptr: *mut c_void) -> *const f32;
    pub fn TensorRows(ptr: *mut c_void) -> i32;
    pub fn TensorCols(ptr: *mut c_void) -> i32;
    
    // Activation Functions
    pub fn TensorReLU(tensor_ptr: *mut c_void) -> *mut c_void;
    pub fn TensorSigmoid(tensor_ptr: *mut c_void) -> *mut c_void;
    pub fn TensorTanh(tensor_ptr: *mut c_void) -> *mut c_void;
    
    // Mathematical Functions
    pub fn TensorSin(tensor_ptr: *mut c_void) -> *mut c_void;
    pub fn TensorCos(tensor_ptr: *mut c_void) -> *mut c_void;
    pub fn TensorExp(tensor_ptr: *mut c_void) -> *mut c_void;
    pub fn TensorLog(tensor_ptr: *mut c_void) -> *mut c_void;
    pub fn TensorSqrt(tensor_ptr: *mut c_void) -> *mut c_void;
    pub fn TensorAbs(tensor_ptr: *mut c_void) -> *mut c_void;
    pub fn TensorPow(tensor_ptr: *mut c_void, exponent: f32) -> *mut c_void;
    
    // Statistical Functions
    pub fn TensorStd(tensor_ptr: *mut c_void) -> f32;
    pub fn TensorVar(tensor_ptr: *mut c_void) -> f32;
    pub fn TensorArgmax(tensor_ptr: *mut c_void) -> i32;
    pub fn TensorArgmin(tensor_ptr: *mut c_void) -> i32;
    
    // Neural Network
    pub fn CreateLinear(in_features: i32, out_features: i32) -> *mut c_void;
    pub fn LinearForward(linear_ptr: *mut c_void, input_tensor_ptr: *mut c_void) -> *mut c_void;
    
    // Loss Functions
    pub fn MSELoss(prediction_ptr: *mut c_void, target_ptr: *mut c_void) -> *mut c_void;
    pub fn CrossEntropyLoss(prediction_ptr: *mut c_void, target_ptr: *mut c_void) -> *mut c_void;
    
    // Optimizers
    pub fn CreateSGD(linear_ptr: *mut c_void, lr: f32) -> *mut c_void;
    pub fn CreateAdam(linear_ptr: *mut c_void, lr: f32) -> *mut c_void;
    pub fn Backward(loss_ptr: *mut c_void);
    pub fn OptimizerStep(optimizer_ptr: *mut c_void);
    pub fn OptimizerZeroGrad(optimizer_ptr: *mut c_void);
    pub fn FreeOptimizer(ptr: *mut c_void);
}