use std::ffi::CStr;
use std::os::raw::c_void;

use crate::tensor_tensorflow::ffi::{
    VersionTF,
};

pub struct FlowTensors { 
    pub ptr: *mut c_void,
    pub rows: i64,
    pub cols: i64,
}



impl FlowTensors{
  pub fn version_tf() -> String {
        unsafe {
            let version_ptr = VersionTF();
            if version_ptr.is_null() {
                panic!("Falha ao obter a versão do TensorFlow");
            }
            CStr::from_ptr(version_ptr as *const i8)
                .to_string_lossy()
                .into_owned()
        }
    }
}
