use std::os::raw::{c_char, c_void, c_int};


#[link(name = "ai_copper_cpp", kind = "dylib")]
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
    pub fn GetTensorData(tensor_ptr: *mut c_void) -> *mut f32;
    pub fn FreeTFTensor(tensor_ptr: *mut c_void);
    pub fn FreeModel(model_handle: *mut c_void);
}
