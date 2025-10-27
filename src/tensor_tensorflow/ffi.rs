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
}
