use std::os::raw::{c_char, c_void, c_int};


#[link(name = "ai_copper", kind = "dylib")]
unsafe extern "C" { 
    pub unsafe fn VersionTF() -> *const c_char;
    pub fn LoadSavedModel(model_path: *const c_char, tags: *const c_char) -> *mut c_void;
    pub fn RunSession(
        model_handle: *mut c_void,
        input_names: *const *const c_char,
        input_tensors: *const *mut c_void,
        num_inputs: c_int,
        output_names: *const *const c_char,
        output_tensors: *mut *mut c_void,
        num_outputs: c_int,
    ) -> *mut c_void;
    pub fn CreateTFTensor(values: *const f32, dims: *const i64, num_dims: c_int) -> *mut c_void;
    pub fn CreateTFTensor_double(values: *const f64, dims: *const i64, num_dims: c_int) -> *mut c_void;
    pub fn CreateTFTensor_int32(values: *const i32, dims: *const i64, num_dims: c_int) -> *mut c_void;
    pub fn CreateTFTensor_int64(values: *const i64, dims: *const i64, num_dims: c_int) -> *mut c_void;
    pub fn CreateTFTensor_int8(values: *const i8, dims: *const i64, num_dims: c_int) -> *mut c_void;
    pub fn CreateTFTensor_int16(values: *const i16, dims: *const i64, num_dims: c_int) -> *mut c_void;
    pub fn CreateTFTensor_uint8(values: *const u8, dims: *const i64, num_dims: c_int) -> *mut c_void;
    pub fn CreateTFTensor_uint16(values: *const u16, dims: *const i64, num_dims: c_int) -> *mut c_void;
    pub fn CreateTFTensor_bool(values: *const u8, dims: *const i64, num_dims: c_int) -> *mut c_void;
    pub fn CreateTFTensor_complex64(values: *const f32, dims: *const i64, num_dims: c_int) -> *mut c_void;
    pub fn CreateTFTensor_complex128(values: *const f64, dims: *const i64, num_dims: c_int) -> *mut c_void;
    pub fn CreateTFTensor_string(values: *const *const c_char, dims: *const i64, num_dims: c_int) -> *mut c_void;
    pub fn GetTensorData_string(tensor_ptr: *mut c_void, out_count: *mut i64) -> *mut *mut c_char;
    pub fn FreeStringArray(arr: *mut *mut c_char, count: i64);
    pub fn GetTensorData(tensor_ptr: *mut c_void) -> *mut f32;
    pub fn GetTensorData_double(tensor_ptr: *mut c_void) -> *mut f64;
    pub fn GetTensorData_int32(tensor_ptr: *mut c_void) -> *mut i32;
    pub fn GetTensorData_int64(tensor_ptr: *mut c_void) -> *mut i64;
    pub fn GetTensorData_int8(tensor_ptr: *mut c_void) -> *mut i8;
    pub fn GetTensorData_int16(tensor_ptr: *mut c_void) -> *mut i16;
    pub fn GetTensorData_uint8(tensor_ptr: *mut c_void) -> *mut u8;
    pub fn GetTensorData_uint16(tensor_ptr: *mut c_void) -> *mut u16;
    pub fn GetTensorData_bool(tensor_ptr: *mut c_void) -> *mut u8;
    pub fn GetTensorData_complex64(tensor_ptr: *mut c_void) -> *mut f32;
    pub fn GetTensorData_complex128(tensor_ptr: *mut c_void) -> *mut f64;
    pub fn GetTensorData_string_placeholder(tensor_ptr: *mut c_void) -> *mut u8;
    pub fn FreeTFTensor(tensor_ptr: *mut c_void);
    pub fn FreeModel(model_handle: *mut c_void);
    // Native op stubs (to be implemented in C++ using TensorFlow internals)
    pub fn TF_Softmax_Native(tensor_ptr: *mut c_void, axis: c_int) -> *mut c_void;
    pub fn TF_LogSoftmax_Native(tensor_ptr: *mut c_void, axis: c_int) -> *mut c_void;
    pub fn TF_Conv2D_Native(
        input: *mut c_void,
        filter: *mut c_void,
        stride_h: c_int,
        stride_w: c_int,
        dilation_h: c_int,
        dilation_w: c_int,
        padding: *const c_char,
        layout: *const c_char,
    ) -> *mut c_void;
    pub fn TF_Conv3D_Native(
        input: *mut c_void,
        filter: *mut c_void,
        stride_d: c_int,
        stride_h: c_int,
        stride_w: c_int,
        dilation_d: c_int,
        dilation_h: c_int,
        dilation_w: c_int,
        padding: *const c_char,
        layout: *const c_char,
    ) -> *mut c_void;
    pub fn TF_BiasAdd_Native(input: *mut c_void, bias: *mut c_void, axis: c_int) -> *mut c_void;
    pub fn TF_BatchNorm_Native(
        input: *mut c_void,
        mean: *mut c_void,
        variance: *mut c_void,
        scale: *mut c_void,
        offset: *mut c_void,
        epsilon: f32,
        axis: c_int,
    ) -> *mut c_void;
    pub fn TF_SoftmaxCrossEntropy_Native(logits: *mut c_void, labels: *mut c_void, axis: c_int) -> *mut c_void;
    pub fn TF_SoftmaxCrossEntropy_Sparse_Native(logits: *mut c_void, labels_idx: *mut c_void, axis: c_int) -> *mut c_void;
    pub fn TF_Dropout_Native(input: *mut c_void, keep_prob: f32, seed: u64) -> *mut c_void;
    pub fn TF_MaxPool_Native(
        input: *mut c_void,
        k_h: c_int,
        k_w: c_int,
        stride_h: c_int,
        stride_w: c_int,
        padding: *const c_char,
        layout: *const c_char,
    ) -> *mut c_void;
    pub fn TF_AvgPool_Native(
        input: *mut c_void,
        k_h: c_int,
        k_w: c_int,
        stride_h: c_int,
        stride_w: c_int,
        padding: *const c_char,
        layout: *const c_char,
    ) -> *mut c_void;
    // Helpers to inspect TF_Tensor returned from native functions
    // Returns a heap-allocated int64 array of dims; out_len is set to number of dims. Caller must free via FreeInt64Array
    pub fn GetTFTensorDims(tensor_ptr: *mut c_void, out_len: *mut c_int) -> *mut i64;
    // Free int64 array returned by GetTFTensorDims
    pub fn FreeInt64Array(arr: *mut i64);
    // Returns dtype code mapped to Rust DType enum ordering (0=F32,1=F64,2=I32,3=I64,4=I8,5=I16,6=U8,7=U16,8=Bool,9=Complex64,10=Complex128,11=StringPlaceholder,12=Unknown)
    pub fn GetTFTensorDType(tensor_ptr: *mut c_void) -> c_int;
}
