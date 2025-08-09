use libc::{c_char};

#[link(name = "ai_copper", kind = "dylib")]
unsafe extern "C" { 
    pub unsafe fn VersionTF() -> *const c_char;



}
