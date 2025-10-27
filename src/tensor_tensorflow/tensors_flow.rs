use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void, c_int};
use std::ptr;

// Random utilities
use rand::{thread_rng, Rng};
use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::distributions::WeightedIndex;
use rand_distr::{Normal, Gamma};

pub struct FlowTensors {
    ptr: *mut c_void, // Ponteiro para TF_Tensor*
    dims: Vec<i64>,   // Dimensões do tensor (suporta qualquer número de dimensões)
    dtype: DType,
}

/// Supported data types for FlowTensors (Rust-side representation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    I8,
    I16,
    U8,
    U16,
    Bool,
    Complex64,
    Complex128,
    StringPlaceholder,
    Unknown,
}

// Binary operator enum at module scope so impls can be referenced from methods below.
#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOp {
    pub fn apply_f32(&self, a: f32, b: f32) -> f32 {
        match self {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
        }
    }

    pub fn apply_f64(&self, a: f64, b: f64) -> f64 {
        match self {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
        }
    }

    pub fn apply_i32(&self, a: i32, b: i32) -> i32 {
        match self {
            BinaryOp::Add => a.wrapping_add(b),
            BinaryOp::Sub => a.wrapping_sub(b),
            BinaryOp::Mul => a.wrapping_mul(b),
            BinaryOp::Div => a / b,
        }
    }

    pub fn apply_i64(&self, a: i64, b: i64) -> i64 {
        match self {
            BinaryOp::Add => a.wrapping_add(b),
            BinaryOp::Sub => a.wrapping_sub(b),
            BinaryOp::Mul => a.wrapping_mul(b),
            BinaryOp::Div => a / b,
        }
    }
}

// We'll keep the FlowTensors struct backward-compatible but it will carry dtype info.
impl FlowTensors {
    fn with_ptr(ptr: *mut c_void, dims: Vec<i64>, dtype: DType) -> Self {
        FlowTensors { ptr, dims, dtype }
    }
    /// Get the recorded dtype
    pub fn dtype(&self) -> DType { self.dtype }
}

pub struct TensorFlowModel {
    handle: *mut c_void, // Ponteiro para ModelHandle
}

impl TensorFlowModel {
    /// Carrega um modelo SavedModel
    pub fn load(model_path: &str, tags: &str) -> Option<Self> {
        unsafe {
            let model_path_c = CString::new(model_path).ok()?;
            let tags_c = CString::new(tags).ok()?;
            let handle = crate::tensor_tensorflow::ffi::LoadSavedModel(
                model_path_c.as_ptr(),
                tags_c.as_ptr(),
            );
            if handle.is_null() {
                return None;
            }
            Some(TensorFlowModel { handle })
        }
    }

    /// Executa inferência no modelo
    pub fn run(
        &self,
        input_names: &[&str],
        input_tensors: &[&FlowTensors],
        output_names: &[&str],
    ) -> Option<Vec<FlowTensors>> {
        unsafe {
            // Converter nomes de entrada e saída para C strings
            let input_names_c: Vec<CString> = input_names
                .iter()
                .map(|&name| CString::new(name).unwrap())
                .collect();
            let input_names_ptr: Vec<*const c_char> =
                input_names_c.iter().map(|cstr| cstr.as_ptr()).collect();

            let output_names_c: Vec<CString> = output_names
                .iter()
                .map(|&name| CString::new(name).unwrap())
                .collect();
            let output_names_ptr: Vec<*const c_char> =
                output_names_c.iter().map(|cstr| cstr.as_ptr()).collect();

            // Obter ponteiros dos tensores de entrada
            let input_tensors_ptr: Vec<*mut c_void> =
                input_tensors.iter().map(|tensor| tensor.ptr).collect();

            // Preparar espaço para tensores de saída
            let mut output_tensors_ptr: Vec<*mut c_void> =
                vec![ptr::null_mut(); output_names.len()];

            // Chamar RunSession
            let result = crate::tensor_tensorflow::ffi::RunSession(
                self.handle,
                input_names_ptr.as_ptr(),
                input_tensors_ptr.as_ptr(),
                input_tensors.len() as c_int,
                output_names_ptr.as_ptr(),
                output_tensors_ptr.as_mut_ptr(),
                output_names.len() as c_int,
            );

            if result.is_null() {
                return None;
            }

            // Criar FlowTensors para as saídas
            let output_tensors: Vec<FlowTensors> = output_tensors_ptr
                .into_iter()
                .enumerate()
                .filter_map(|(_i, ptr)| {
                    if ptr.is_null() {
                        return None;
                    }
                    // Supor dimensões de saída (você deve obter as dimensões reais do modelo)
                    let dims = vec![1, 1]; // Exemplo: ajustar conforme o modelo
                            Some(FlowTensors { ptr, dims, dtype: DType::Unknown })
                })
                .collect();

            Some(output_tensors)
        }
    }
}

impl Drop for TensorFlowModel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                crate::tensor_tensorflow::ffi::FreeModel(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

unsafe impl Send for TensorFlowModel {}
unsafe impl Sync for TensorFlowModel {}

impl FlowTensors {
    /// Cria um tensor a partir de um array de valores e dimensões
    pub fn new(values: &[f32], dims: &[i64]) -> Option<Self> {
        unsafe {
            let tensor_ptr = crate::tensor_tensorflow::ffi::CreateTFTensor(
                values.as_ptr(),
                dims.as_ptr(),
                dims.len() as c_int,
            );
            if tensor_ptr.is_null() {
                return None;
            }
            Some(FlowTensors { ptr: tensor_ptr, dims: dims.to_vec(), dtype: DType::F32 })
        }
    }

    /// Obtém os dados do tensor como um slice de f32
    pub fn data(&self) -> Option<&[f32]> {
        unsafe {
            let data_ptr = crate::tensor_tensorflow::ffi::GetTensorData(self.ptr);
            if data_ptr.is_null() {
                eprintln!("Erro: GetTensorData retornou ponteiro nulo");
                return None;
            }
            let size = self.dims.iter().product::<i64>() as usize;
            if size == 0 {
                eprintln!("Aviso: Tensor com tamanho zero");
                return Some(&[]);
            }
            Some(std::slice::from_raw_parts(data_ptr, size))
        }
    }

    /// Obtém as dimensões do tensor
    pub fn dims(&self) -> &[i64] {
        &self.dims
    }

    // --- Typed constructors (true dtype support via native FFI) ---
    pub fn new_f64(values: &[f64], dims: &[i64]) -> Option<Self> {
        unsafe {
            let tensor_ptr = crate::tensor_tensorflow::ffi::CreateTFTensor_double(
                values.as_ptr(),
                dims.as_ptr(),
                dims.len() as c_int,
            );
            if tensor_ptr.is_null() { return None; }
            Some(FlowTensors { ptr: tensor_ptr, dims: dims.to_vec(), dtype: DType::F64 })
        }
    }

    pub fn new_i32(values: &[i32], dims: &[i64]) -> Option<Self> {
        unsafe {
            let tensor_ptr = crate::tensor_tensorflow::ffi::CreateTFTensor_int32(
                values.as_ptr(), dims.as_ptr(), dims.len() as c_int);
            if tensor_ptr.is_null() { return None; }
            Some(FlowTensors { ptr: tensor_ptr, dims: dims.to_vec(), dtype: DType::I32 })
        }
    }

    pub fn new_i64(values: &[i64], dims: &[i64]) -> Option<Self> {
        unsafe {
            let tensor_ptr = crate::tensor_tensorflow::ffi::CreateTFTensor_int64(
                values.as_ptr(), dims.as_ptr(), dims.len() as c_int);
            if tensor_ptr.is_null() { return None; }
            Some(FlowTensors { ptr: tensor_ptr, dims: dims.to_vec(), dtype: DType::I64 })
        }
    }

    pub fn new_i8(values: &[i8], dims: &[i64]) -> Option<Self> {
        unsafe {
            let tensor_ptr = crate::tensor_tensorflow::ffi::CreateTFTensor_int8(
                values.as_ptr(), dims.as_ptr(), dims.len() as c_int);
            if tensor_ptr.is_null() { return None; }
            Some(FlowTensors { ptr: tensor_ptr, dims: dims.to_vec(), dtype: DType::I8 })
        }
    }

    pub fn new_i16(values: &[i16], dims: &[i64]) -> Option<Self> {
        unsafe {
            let tensor_ptr = crate::tensor_tensorflow::ffi::CreateTFTensor_int16(
                values.as_ptr(), dims.as_ptr(), dims.len() as c_int);
            if tensor_ptr.is_null() { return None; }
            Some(FlowTensors { ptr: tensor_ptr, dims: dims.to_vec(), dtype: DType::I16 })
        }
    }

    pub fn new_u8(values: &[u8], dims: &[i64]) -> Option<Self> {
        unsafe {
            let tensor_ptr = crate::tensor_tensorflow::ffi::CreateTFTensor_uint8(
                values.as_ptr(), dims.as_ptr(), dims.len() as c_int);
            if tensor_ptr.is_null() { return None; }
            Some(FlowTensors { ptr: tensor_ptr, dims: dims.to_vec(), dtype: DType::U8 })
        }
    }

    pub fn new_u16(values: &[u16], dims: &[i64]) -> Option<Self> {
        unsafe {
            let tensor_ptr = crate::tensor_tensorflow::ffi::CreateTFTensor_uint16(
                values.as_ptr(), dims.as_ptr(), dims.len() as c_int);
            if tensor_ptr.is_null() { return None; }
            Some(FlowTensors { ptr: tensor_ptr, dims: dims.to_vec(), dtype: DType::U16 })
        }
    }

    pub fn new_bool(values: &[u8], dims: &[i64]) -> Option<Self> {
        unsafe {
            let tensor_ptr = crate::tensor_tensorflow::ffi::CreateTFTensor_bool(
                values.as_ptr(), dims.as_ptr(), dims.len() as c_int);
            if tensor_ptr.is_null() { return None; }
            Some(FlowTensors { ptr: tensor_ptr, dims: dims.to_vec(), dtype: DType::Bool })
        }
    }

    pub fn new_complex64(values: &[f32], dims: &[i64]) -> Option<Self> {
        unsafe {
            let tensor_ptr = crate::tensor_tensorflow::ffi::CreateTFTensor_complex64(
                values.as_ptr(), dims.as_ptr(), dims.len() as c_int);
            if tensor_ptr.is_null() { return None; }
            Some(FlowTensors { ptr: tensor_ptr, dims: dims.to_vec(), dtype: DType::Complex64 })
        }
    }

    pub fn new_complex128(values: &[f64], dims: &[i64]) -> Option<Self> {
        unsafe {
            let tensor_ptr = crate::tensor_tensorflow::ffi::CreateTFTensor_complex128(
                values.as_ptr(), dims.as_ptr(), dims.len() as c_int);
            if tensor_ptr.is_null() { return None; }
            Some(FlowTensors { ptr: tensor_ptr, dims: dims.to_vec(), dtype: DType::Complex128 })
        }
    }

    /// Placeholder: creates a byte-packed string tensor (see C++ note)
    pub fn new_string(values: &[&str], dims: &[i64]) -> Option<Self> {
        unsafe {
            let cstrings: Vec<CString> = values.iter().map(|s| CString::new(*s).unwrap()).collect();
            let ptrs: Vec<*const c_char> = cstrings.iter().map(|c| c.as_ptr()).collect();
            let tensor_ptr = crate::tensor_tensorflow::ffi::CreateTFTensor_string(ptrs.as_ptr(), dims.as_ptr(), dims.len() as c_int);
            if tensor_ptr.is_null() { return None; }
            Some(FlowTensors { ptr: tensor_ptr, dims: dims.to_vec(), dtype: DType::StringPlaceholder })
        }
    }

    // --- Typed accessors ---
    pub fn data_f64(&self) -> Option<&[f64]> {
        unsafe {
            let data_ptr = crate::tensor_tensorflow::ffi::GetTensorData_double(self.ptr);
            if data_ptr.is_null() { return None; }
            let size = self.dims.iter().product::<i64>() as usize;
            Some(std::slice::from_raw_parts(data_ptr, size))
        }
    }

    pub fn data_i32(&self) -> Option<&[i32]> {
        unsafe {
            let data_ptr = crate::tensor_tensorflow::ffi::GetTensorData_int32(self.ptr);
            if data_ptr.is_null() { return None; }
            let size = self.dims.iter().product::<i64>() as usize;
            Some(std::slice::from_raw_parts(data_ptr, size))
        }
    }

    pub fn data_i64(&self) -> Option<&[i64]> {
        unsafe {
            let data_ptr = crate::tensor_tensorflow::ffi::GetTensorData_int64(self.ptr);
            if data_ptr.is_null() { return None; }
            let size = self.dims.iter().product::<i64>() as usize;
            Some(std::slice::from_raw_parts(data_ptr, size))
        }
    }

    /// Update the contents of this TF_Tensor in-place using a slice of f32 values.
    /// Returns true on success.
    // in-place helpers were removed per user request (undo last change).

    pub fn data_i8(&self) -> Option<&[i8]> {
        unsafe {
            let data_ptr = crate::tensor_tensorflow::ffi::GetTensorData_int8(self.ptr);
            if data_ptr.is_null() { return None; }
            let size = self.dims.iter().product::<i64>() as usize;
            Some(std::slice::from_raw_parts(data_ptr, size))
        }
    }

    pub fn data_i16(&self) -> Option<&[i16]> {
        unsafe {
            let data_ptr = crate::tensor_tensorflow::ffi::GetTensorData_int16(self.ptr);
            if data_ptr.is_null() { return None; }
            let size = self.dims.iter().product::<i64>() as usize;
            Some(std::slice::from_raw_parts(data_ptr, size))
        }
    }

    pub fn data_u8(&self) -> Option<&[u8]> {
        unsafe {
            let data_ptr = crate::tensor_tensorflow::ffi::GetTensorData_uint8(self.ptr);
            if data_ptr.is_null() { return None; }
            let size = self.dims.iter().product::<i64>() as usize;
            Some(std::slice::from_raw_parts(data_ptr, size))
        }
    }

    pub fn data_u16(&self) -> Option<&[u16]> {
        unsafe {
            let data_ptr = crate::tensor_tensorflow::ffi::GetTensorData_uint16(self.ptr);
            if data_ptr.is_null() { return None; }
            let size = self.dims.iter().product::<i64>() as usize;
            Some(std::slice::from_raw_parts(data_ptr, size))
        }
    }

    pub fn data_bool(&self) -> Option<&[u8]> {
        unsafe {
            let data_ptr = crate::tensor_tensorflow::ffi::GetTensorData_bool(self.ptr);
            if data_ptr.is_null() { return None; }
            let size = self.dims.iter().product::<i64>() as usize;
            Some(std::slice::from_raw_parts(data_ptr, size))
        }
    }

    /// For complex64 returns interleaved float pairs [re,im, re,im, ...]
    pub fn data_complex64(&self) -> Option<&[f32]> {
        unsafe {
            let data_ptr = crate::tensor_tensorflow::ffi::GetTensorData_complex64(self.ptr);
            if data_ptr.is_null() { return None; }
            let size = self.dims.iter().product::<i64>() as usize * 2; // two floats per complex
            Some(std::slice::from_raw_parts(data_ptr, size))
        }
    }

    pub fn data_complex128(&self) -> Option<&[f64]> {
        unsafe {
            let data_ptr = crate::tensor_tensorflow::ffi::GetTensorData_complex128(self.ptr);
            if data_ptr.is_null() { return None; }
            let size = self.dims.iter().product::<i64>() as usize * 2; // two doubles per complex
            Some(std::slice::from_raw_parts(data_ptr, size))
        }
    }

    /// Placeholder accessor for string-packed tensors
    pub fn data_string_placeholder(&self) -> Option<&[u8]> {
        unsafe {
            let data_ptr = crate::tensor_tensorflow::ffi::GetTensorData_string_placeholder(self.ptr);
            if data_ptr.is_null() { return None; }
            // We do not know exact packed size here; best-effort: compute byte size from dims as upper bound
            let size = self.dims.iter().product::<i64>() as usize * 8; // heuristic
            Some(std::slice::from_raw_parts(data_ptr, size))
        }
    }

    /// Decode TF_STRING tensor and return Vec<String>
    pub fn data_strings(&self) -> Option<Vec<String>> {
        unsafe {
            let mut count: i64 = 0;
            let raw = crate::tensor_tensorflow::ffi::GetTensorData_string(self.ptr, &mut count as *mut i64);
            if raw.is_null() { return None; }
            let mut out = Vec::with_capacity(count as usize);
            for i in 0..(count as isize) {
                let ptr = *raw.offset(i);
                if ptr.is_null() {
                    out.push(String::new());
                } else {
                    let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
                    out.push(s);
                }
            }
            crate::tensor_tensorflow::ffi::FreeStringArray(raw, count);
            Some(out)
        }
    }

    /// Obtém a versão do TensorFlow
    pub fn version_tf() -> String {
        unsafe {
            let version_ptr = crate::tensor_tensorflow::ffi::VersionTF();
            if version_ptr.is_null() {
                panic!("Falha ao obter a versão do TensorFlow");
            }
            CStr::from_ptr(version_ptr)
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Calcula a soma de todos os elementos
    pub fn sum(&self) -> f32 {
        self.data()
            .expect("Failed to get tensor data")
            .iter()
            .sum()
    }

    /// Calcula a média de todos os elementos
    pub fn mean(&self) -> f32 {
        let data = self.data().expect("Failed to get tensor data");
        let sum: f32 = data.iter().sum();
        sum / data.len() as f32
    }

    /// Calcula o valor máximo
    pub fn max(&self) -> f32 {
        self.data()
            .expect("Failed to get tensor data")
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Calcula o valor mínimo
    pub fn min(&self) -> f32 {
        self.data()
            .expect("Failed to get tensor data")
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min)
    }

    /// Transpõe o tensor (apenas para matrizes 2D)
    pub fn transpose(&self) -> Option<FlowTensors> {
        if self.dims.len() != 2 {
            return None;
        }

        let rows = self.dims[0] as usize;
        let cols = self.dims[1] as usize;
        let data = self.data()?;
        let mut transposed = vec![0.0f32; rows * cols];

        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = data[r * cols + c];
            }
        }

        FlowTensors::new(&transposed, &[cols as i64, rows as i64])
    }

    /// Aplica uma função a cada elemento
    pub fn map<F>(&self, f: F) -> Option<FlowTensors>
    where
        F: Fn(f32) -> f32 + Sync + Send,
    {
        let data = self.data()?;
        let size = data.len();
        // Threshold to decide between sequential and parallel
        const PAR_THRESHOLD: usize = 1 << 12; // 4096

        let mapped: Vec<f32> = if size >= PAR_THRESHOLD {
            // Use rayon parallel iterator for large tensors (CPU-only parallelism)
            use rayon::prelude::*;
            data.par_iter().map(|&x| f(x)).collect()
        } else {
            data.iter().map(|&x| f(x)).collect()
        };

        FlowTensors::new(&mapped, &self.dims)
    }

    /// Operadores aritméticos elemento a elemento
    pub fn add(&self, other: &FlowTensors) -> Option<FlowTensors> {
        self.elementwise_binary_op(other, BinaryOp::Add)
    }

    pub fn sub(&self, other: &FlowTensors) -> Option<FlowTensors> {
        self.elementwise_binary_op(other, BinaryOp::Sub)
    }

    pub fn mul(&self, other: &FlowTensors) -> Option<FlowTensors> {
        self.elementwise_binary_op(other, BinaryOp::Mul)
    }

    pub fn div(&self, other: &FlowTensors) -> Option<FlowTensors> {
        self.elementwise_binary_op(other, BinaryOp::Div)
    }

    // DType promotion for binary ops. For Div we prefer floating target types.
    fn promote_dtype(a: DType, b: DType, is_div: bool) -> DType {
        use DType::*;
        // Rank list from highest to lowest
        let rank = |d: DType| match d {
            Complex128 => 12,
            Complex64 => 11,
            F64 => 10,
            F32 => 9,
            I64 => 8,
            I32 => 7,
            I16 => 6,
            I8 => 5,
            U16 => 4,
            U8 => 3,
            Bool => 1,
            StringPlaceholder => 0,
            Unknown => 0,
        };
        let mut target = if rank(a) >= rank(b) { a } else { b };
        if is_div {
            // promote integers to float for division
            target = match target {
                I64 | I32 | I16 | I8 | U16 | U8 => F64,
                Bool => F32,
                other => other,
            };
        }
        target
    }

    // Unified element-wise binary operator dispatcher
    fn elementwise_binary_op(&self, other: &FlowTensors, op: BinaryOp) -> Option<FlowTensors> {
        if self.dims != other.dims {
            eprintln!("Erro: dimensões incompatíveis em FlowTensors::{:?}", op);
            return None;
        }
        let is_div = matches!(op, BinaryOp::Div);
        let target = FlowTensors::promote_dtype(self.dtype, other.dtype, is_div);
        let size = self.dims.iter().product::<i64>() as usize;
        match target {
            DType::F32 => {
                // read both as f32 (fallback casting)
                let a: Vec<f32> = self.to_f32_vec()?;
                let b: Vec<f32> = other.to_f32_vec()?;
                let res: Vec<f32> = if size >= (1<<12) {
                    use rayon::prelude::*;
                    a.par_iter().zip(b.par_iter()).map(|(x,y)| op.apply_f32(*x,*y)).collect()
                } else {
                    a.iter().zip(b.iter()).map(|(x,y)| op.apply_f32(*x,*y)).collect()
                };
                FlowTensors::new(&res, &self.dims)
            }
            DType::F64 => {
                let a: Vec<f64> = self.to_f64_vec()?;
                let b: Vec<f64> = other.to_f64_vec()?;
                let res: Vec<f64> = if size >= (1<<12) {
                    use rayon::prelude::*;
                    a.par_iter().zip(b.par_iter()).map(|(x,y)| op.apply_f64(*x,*y)).collect()
                } else {
                    a.iter().zip(b.iter()).map(|(x,y)| op.apply_f64(*x,*y)).collect()
                };
                FlowTensors::new_f64(&res, &self.dims)
            }
            DType::I32 => {
                let a: Vec<i32> = self.to_i32_vec()?;
                let b: Vec<i32> = other.to_i32_vec()?;
                let res: Vec<i32> = a.iter().zip(b.iter()).map(|(x,y)| op.apply_i32(*x,*y)).collect();
                FlowTensors::new_i32(&res, &self.dims)
            }
            DType::I64 => {
                let a: Vec<i64> = self.to_i64_vec()?;
                let b: Vec<i64> = other.to_i64_vec()?;
                let res: Vec<i64> = a.iter().zip(b.iter()).map(|(x,y)| op.apply_i64(*x,*y)).collect();
                FlowTensors::new_i64(&res, &self.dims)
            }
            DType::Complex64 => {
                // operate on interleaved (re,im) pairs
                let a = self.to_complex64_vec()?;
                let b = other.to_complex64_vec()?;
                if a.len() != b.len() {
                    eprintln!("Erro: tamanhos diferentes para complex arithmetic");
                    return None;
                }
                let mut out_interleaved: Vec<f32> = Vec::with_capacity(a.len() * 2);
                for (i, (ar, ai)) in a.iter().enumerate() {
                    let (br, bi) = b[i];
                    let (rr, ri) = match op {
                        BinaryOp::Add => (ar + br, ai + bi),
                        BinaryOp::Sub => (ar - br, ai - bi),
                        BinaryOp::Mul => (ar * br - ai * bi, ar * bi + ai * br),
                        BinaryOp::Div => {
                            let denom = br * br + bi * bi;
                            if denom == 0.0 {
                                eprintln!("Erro: divisão por zero em complex Division");
                                return None;
                            }
                            ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
                        }
                    };
                    out_interleaved.push(rr);
                    out_interleaved.push(ri);
                }
                FlowTensors::new_complex64(&out_interleaved, &self.dims)
            }
            _ => {
                eprintln!("Op for dtype {:?} not implemented yet", target);
                None
            }
        }
    }

    // Conversion helpers: convert any supported dtype to Vec<T>
    fn to_f32_vec(&self) -> Option<Vec<f32>> {
        match self.dtype {
            DType::F32 => Some(self.data()?.to_vec()),
            DType::F64 => Some(self.data_f64()?.iter().map(|&v| v as f32).collect()),
            DType::I32 => Some(self.data_i32()?.iter().map(|&v| v as f32).collect()),
            DType::I64 => Some(self.data_i64()?.iter().map(|&v| v as f32).collect()),
            DType::I8 => Some(self.data_i8()?.iter().map(|&v| v as f32).collect()),
            DType::I16 => Some(self.data_i16()?.iter().map(|&v| v as f32).collect()),
            DType::U8 => Some(self.data_u8()?.iter().map(|&v| v as f32).collect()),
            DType::U16 => Some(self.data_u16()?.iter().map(|&v| v as f32).collect()),
            DType::Bool => Some(self.data_bool()?.iter().map(|&v| if v==0 {0.0} else {1.0}).collect()),
            _ => None,
        }
    }

    fn to_f64_vec(&self) -> Option<Vec<f64>> {
        match self.dtype {
            DType::F64 => Some(self.data_f64()?.to_vec()),
            DType::F32 => Some(self.data()?.iter().map(|&v| v as f64).collect()),
            DType::I32 => Some(self.data_i32()?.iter().map(|&v| v as f64).collect()),
            DType::I64 => Some(self.data_i64()?.iter().map(|&v| v as f64).collect()),
            DType::I8 => Some(self.data_i8()?.iter().map(|&v| v as f64).collect()),
            DType::I16 => Some(self.data_i16()?.iter().map(|&v| v as f64).collect()),
            DType::U8 => Some(self.data_u8()?.iter().map(|&v| v as f64).collect()),
            DType::U16 => Some(self.data_u16()?.iter().map(|&v| v as f64).collect()),
            DType::Bool => Some(self.data_bool()?.iter().map(|&v| if v==0 {0.0} else {1.0}).collect()),
            _ => None,
        }
    }

    fn to_i32_vec(&self) -> Option<Vec<i32>> {
        match self.dtype {
            DType::I32 => Some(self.data_i32()?.to_vec()),
            DType::F32 => Some(self.data()?.iter().map(|&v| v as i32).collect()),
            DType::F64 => Some(self.data_f64()?.iter().map(|&v| v as i32).collect()),
            DType::I64 => Some(self.data_i64()?.iter().map(|&v| v as i32).collect()),
            DType::I8 => Some(self.data_i8()?.iter().map(|&v| v as i32).collect()),
            DType::I16 => Some(self.data_i16()?.iter().map(|&v| v as i32).collect()),
            DType::U8 => Some(self.data_u8()?.iter().map(|&v| v as i32).collect()),
            DType::U16 => Some(self.data_u16()?.iter().map(|&v| v as i32).collect()),
            DType::Bool => Some(self.data_bool()?.iter().map(|&v| if v==0 {0} else {1}).collect()),
            _ => None,
        }
    }

    fn to_i64_vec(&self) -> Option<Vec<i64>> {
        match self.dtype {
            DType::I64 => Some(self.data_i64()?.to_vec()),
            DType::I32 => Some(self.data_i32()?.iter().map(|&v| v as i64).collect()),
            DType::F32 => Some(self.data()?.iter().map(|&v| v as i64).collect()),
            DType::F64 => Some(self.data_f64()?.iter().map(|&v| v as i64).collect()),
            DType::I8 => Some(self.data_i8()?.iter().map(|&v| v as i64).collect()),
            DType::I16 => Some(self.data_i16()?.iter().map(|&v| v as i64).collect()),
            DType::U8 => Some(self.data_u8()?.iter().map(|&v| v as i64).collect()),
            DType::U16 => Some(self.data_u16()?.iter().map(|&v| v as i64).collect()),
            DType::Bool => Some(self.data_bool()?.iter().map(|&v| if v==0 {0} else {1}).collect()),
            _ => None,
        }
    }

    fn to_complex64_vec(&self) -> Option<Vec<(f32,f32)>> {
        // returns Vec of (real, imag)
        match self.dtype {
            DType::Complex64 => {
                let raw = self.data_complex64()?;
                let mut out = Vec::with_capacity(raw.len()/2);
                for i in 0..(raw.len()/2) { out.push((raw[2*i], raw[2*i+1])); }
                Some(out)
            }
            _ => None,
        }
    }

    /// Multiplicação de matrizes (2D)
    pub fn matmul(&self, other: &FlowTensors) -> Option<FlowTensors> {
        if self.dims.len() != 2 || other.dims.len() != 2 {
            eprintln!("Erro: matmul requer tensores 2D");
            return None;
        }
        let rows = self.dims[0] as usize;
        let cols = self.dims[1] as usize;
        let other_rows = other.dims[0] as usize;
        let other_cols = other.dims[1] as usize;
        if cols != other_rows {
            eprintln!("Erro: dimensões inválidas para matmul");
            return None;
        }

        let a = self.data()?;
        let b = other.data()?;
        let mut result = vec![0.0f32; rows * other_cols];

        for i in 0..rows {
            for j in 0..other_cols {
                let mut sum = 0.0f32;
                for k in 0..cols {
                    let a_idx = i * cols + k;
                    let b_idx = k * other_cols + j;
                    sum += a[a_idx] * b[b_idx];
                }
                result[i * other_cols + j] = sum;
            }
        }

        let dims = vec![rows as i64, other_cols as i64];
        FlowTensors::new(&result, &dims)
    }

    /// BatchMatMul: suporta tensores 3D [batch, M, K] x [batch, K, N] -> [batch, M, N]
    pub fn batch_matmul(&self, other: &FlowTensors) -> Option<FlowTensors> {
        if self.dims.len() != 3 || other.dims.len() != 3 {
            eprintln!("Erro: batch_matmul requer tensores 3D");
            return None;
        }
        let batch = self.dims[0] as usize;
        let m = self.dims[1] as usize;
        let k = self.dims[2] as usize;
        let other_batch = other.dims[0] as usize;
        let other_k = other.dims[1] as usize;
        let n = other.dims[2] as usize;
        if batch != other_batch || k != other_k {
            eprintln!("Erro: dimensões incompatíveis para batch_matmul");
            return None;
        }

        let a = self.data()?;
        let b = other.data()?;
        let mut result = vec![0.0f32; batch * m * n];

        for b_idx in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        let a_idx = b_idx * (m * k) + i * k + kk;
                        let b_index = b_idx * (k * n) + kk * n + j;
                        sum += a[a_idx] * b[b_index];
                    }
                    let out_idx = b_idx * (m * n) + i * n + j;
                    result[out_idx] = sum;
                }
            }
        }

        let dims = vec![batch as i64, m as i64, n as i64];
        FlowTensors::new(&result, &dims)
    }

    /// Operações de comparação elemento a elemento
    /// Retornam um tensor f32 com 1.0 para true e 0.0 para false
    pub fn equal(&self, other: &FlowTensors) -> Option<FlowTensors> {
        if self.dims != other.dims {
            eprintln!("Erro: dimensões incompatíveis em FlowTensors::equal");
            return None;
        }
        let a = self.data()?;
        let b = other.data()?;
        let size = a.len();
        const PAR_THRESHOLD: usize = 1 << 12;
        let result: Vec<f32> = if size >= PAR_THRESHOLD {
            use rayon::prelude::*;
            a.par_iter()
                .zip(b.par_iter())
                .map(|(x, y)| if x == y { 1.0f32 } else { 0.0f32 })
                .collect()
        } else {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| if x == y { 1.0f32 } else { 0.0f32 })
                .collect()
        };
        FlowTensors::new(&result, &self.dims)
    }

    pub fn not_equal(&self, other: &FlowTensors) -> Option<FlowTensors> {
        if self.dims != other.dims {
            eprintln!("Erro: dimensões incompatíveis em FlowTensors::not_equal");
            return None;
        }
        let a = self.data()?;
        let b = other.data()?;
        let size = a.len();
        const PAR_THRESHOLD: usize = 1 << 12;
        let result: Vec<f32> = if size >= PAR_THRESHOLD {
            use rayon::prelude::*;
            a.par_iter()
                .zip(b.par_iter())
                .map(|(x, y)| if x != y { 1.0f32 } else { 0.0f32 })
                .collect()
        } else {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| if x != y { 1.0f32 } else { 0.0f32 })
                .collect()
        };
        FlowTensors::new(&result, &self.dims)
    }

    /// Operadores lógicos (tratando qualquer valor != 0.0 como true)
    /// Retornam tensor f32 com 1.0 para true e 0.0 para false
    pub fn logical_and(&self, other: &FlowTensors) -> Option<FlowTensors> {
        if self.dims != other.dims {
            eprintln!("Erro: dimensões incompatíveis em FlowTensors::logical_and");
            return None;
        }
        let a = self.data()?;
        let b = other.data()?;
        let size = a.len();
        const PAR_THRESHOLD: usize = 1 << 12;
        let result: Vec<f32> = if size >= PAR_THRESHOLD {
            use rayon::prelude::*;
            a.par_iter()
                .zip(b.par_iter())
                .map(|(x, y)| if *x != 0.0 && *y != 0.0 { 1.0f32 } else { 0.0f32 })
                .collect()
        } else {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| if *x != 0.0 && *y != 0.0 { 1.0f32 } else { 0.0f32 })
                .collect()
        };
        FlowTensors::new(&result, &self.dims)
    }

    pub fn logical_or(&self, other: &FlowTensors) -> Option<FlowTensors> {
        if self.dims != other.dims {
            eprintln!("Erro: dimensões incompatíveis em FlowTensors::logical_or");
            return None;
        }
        let a = self.data()?;
        let b = other.data()?;
        let size = a.len();
        const PAR_THRESHOLD: usize = 1 << 12;
        let result: Vec<f32> = if size >= PAR_THRESHOLD {
            use rayon::prelude::*;
            a.par_iter()
                .zip(b.par_iter())
                .map(|(x, y)| if *x != 0.0 || *y != 0.0 { 1.0f32 } else { 0.0f32 })
                .collect()
        } else {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| if *x != 0.0 || *y != 0.0 { 1.0f32 } else { 0.0f32 })
                .collect()
        };
        FlowTensors::new(&result, &self.dims)
    }

    pub fn logical_not(&self) -> Option<FlowTensors> {
        let a = self.data()?;
        let size = a.len();
        const PAR_THRESHOLD: usize = 1 << 12;
        let result: Vec<f32> = if size >= PAR_THRESHOLD {
            use rayon::prelude::*;
            a.par_iter().map(|x| if *x == 0.0 { 1.0f32 } else { 0.0f32 }).collect()
        } else {
            a.iter().map(|x| if *x == 0.0 { 1.0f32 } else { 0.0f32 }).collect()
        };
        FlowTensors::new(&result, &self.dims)
    }

    pub fn greater(&self, other: &FlowTensors) -> Option<FlowTensors> {
        if self.dims != other.dims {
            eprintln!("Erro: dimensões incompatíveis em FlowTensors::greater");
            return None;
        }
        let a = self.data()?;
        let b = other.data()?;
        let size = a.len();
        const PAR_THRESHOLD: usize = 1 << 12;
        let result: Vec<f32> = if size >= PAR_THRESHOLD {
            use rayon::prelude::*;
            a.par_iter()
                .zip(b.par_iter())
                .map(|(x, y)| if x > y { 1.0f32 } else { 0.0f32 })
                .collect()
        } else {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| if x > y { 1.0f32 } else { 0.0f32 })
                .collect()
        };
        FlowTensors::new(&result, &self.dims)
    }

    pub fn less(&self, other: &FlowTensors) -> Option<FlowTensors> {
        if self.dims != other.dims {
            eprintln!("Erro: dimensões incompatíveis em FlowTensors::less");
            return None;
        }
        let a = self.data()?;
        let b = other.data()?;
        let size = a.len();
        const PAR_THRESHOLD: usize = 1 << 12;
        let result: Vec<f32> = if size >= PAR_THRESHOLD {
            use rayon::prelude::*;
            a.par_iter()
                .zip(b.par_iter())
                .map(|(x, y)| if x < y { 1.0f32 } else { 0.0f32 })
                .collect()
        } else {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| if x < y { 1.0f32 } else { 0.0f32 })
                .collect()
        };
        FlowTensors::new(&result, &self.dims)
    }

    /// Funções matemáticas
    pub fn pow(&self, exponent: f32) -> Option<FlowTensors> {
        self.map(|x| x.powf(exponent))
    }

    pub fn sqrt(&self) -> Option<FlowTensors> {
        self.map(|x| x.sqrt())
    }

    /// Arredondamentos e clip
    pub fn ceil(&self) -> Option<FlowTensors> {
        self.map(|x| x.ceil())
    }

    pub fn floor(&self) -> Option<FlowTensors> {
        self.map(|x| x.floor())
    }

    pub fn round(&self) -> Option<FlowTensors> {
        self.map(|x| x.round())
    }

    /// Clip - limita valores entre min e max
    pub fn clip(&self, min: f32, max: f32) -> Option<FlowTensors> {
        self.map(|x| {
            if x.is_nan() {
                x
            } else {
                if x < min { min } else if x > max { max } else { x }
            }
        })
    }

    /// Cast simples: converte para inteiros mantendo armazenado como f32 (truncamento)
    pub fn cast_to_int(&self) -> Option<FlowTensors> {
        self.map(|x| x.trunc())
    }

    /// Cast para booleano (1.0/0.0)
    pub fn cast_to_bool(&self) -> Option<FlowTensors> {
        self.map(|x| if x != 0.0 { 1.0f32 } else { 0.0f32 })
    }

    pub fn square(&self) -> Option<FlowTensors> {
        self.map(|x| x * x)
    }

    pub fn abs(&self) -> Option<FlowTensors> {
        self.map(|x| x.abs())
    }

    /// Exponenciais e funções relacionadas
    pub fn exp(&self) -> Option<FlowTensors> {
        self.map(|x| x.exp())
    }

    /// Logaritmo natural
    pub fn ln(&self) -> Option<FlowTensors> {
        self.map(|x| x.ln())
    }

    /// log(1 + x) — implementado como (1+x).ln()
    pub fn log1p(&self) -> Option<FlowTensors> {
        // Use a numerically stable implementation for log1p via libm
        self.map(|x| libm::log1pf(x))
    }

    /// Sigmoid: 1 / (1 + e^{-x})
    pub fn sigmoid(&self) -> Option<FlowTensors> {
        self.map(|x| 1.0f32 / (1.0f32 + (-x).exp()))
    }

    /// Tangente hiperbólica
    pub fn tanh(&self) -> Option<FlowTensors> {
        self.map(|x| x.tanh())
    }

    /// Funções trigonométricas
    pub fn sin(&self) -> Option<FlowTensors> {
        self.map(|x| x.sin())
    }

    pub fn cos(&self) -> Option<FlowTensors> {
        self.map(|x| x.cos())
    }

    pub fn tan(&self) -> Option<FlowTensors> {
        self.map(|x| x.tan())
    }

    pub fn asin(&self) -> Option<FlowTensors> {
        self.map(|x| x.asin())
    }

    pub fn acos(&self) -> Option<FlowTensors> {
        self.map(|x| x.acos())
    }

    pub fn atan(&self) -> Option<FlowTensors> {
        self.map(|x| x.atan())
    }

    /// Cria tensor de zeros
    pub fn zeros(dims: &[i64]) -> Option<Self> {
        let size = dims.iter().product::<i64>() as usize;
        let values = vec![0.0f32; size];
        FlowTensors::new(&values, dims)
    }

    /// Cria tensor de uns
    pub fn ones(dims: &[i64]) -> Option<Self> {
        let size = dims.iter().product::<i64>() as usize;
        let values = vec![1.0f32; size];
        FlowTensors::new(&values, dims)
    }

    /// Fill - cria tensor preenchido com um valor arbitrário
    pub fn fill(dims: &[i64], value: f32) -> Option<Self> {
        let size = dims.iter().product::<i64>() as usize;
        let values = vec![value; size];
        FlowTensors::new(&values, dims)
    }

    /// RandomUniform - valores uniformes no intervalo [low, high)
    pub fn random_uniform(dims: &[i64], low: f32, high: f32) -> Option<Self> {
        if low >= high {
            eprintln!("Erro: random_uniform requer low < high");
            return None;
        }
        let size = dims.iter().product::<i64>() as usize;
        let mut rng = thread_rng();
        let mut values = Vec::with_capacity(size);
        for _ in 0..size {
            values.push(rng.gen_range(low..high));
        }
        FlowTensors::new(&values, dims)
    }

    /// RandomNormal - distribuição normal com média e desvio padrão fornecidos
    pub fn random_normal(dims: &[i64], mean: f32, std: f32) -> Option<Self> {
        if std <= 0.0 {
            eprintln!("Erro: random_normal requer std > 0");
            return None;
        }
        let size = dims.iter().product::<i64>() as usize;
        let mut rng = thread_rng();
        let normal = Normal::new(mean as f64, std as f64).ok()?;
        let mut values = Vec::with_capacity(size);
        for _ in 0..size {
            values.push(normal.sample(&mut rng) as f32);
        }
        FlowTensors::new(&values, dims)
    }

    /// TruncatedNormal - amostras da normal truncada dentro de +/- trunc_std * std
    pub fn truncated_normal(dims: &[i64], mean: f32, std: f32, trunc_std: f32) -> Option<Self> {
        if std <= 0.0 || trunc_std <= 0.0 {
            eprintln!("Erro: truncated_normal requer std > 0 e trunc_std > 0");
            return None;
        }
        let size = dims.iter().product::<i64>() as usize;
        let mut rng = thread_rng();
        let normal = Normal::new(mean as f64, std as f64).ok()?;
        let mut values = Vec::with_capacity(size);
        let lower = mean as f32 - trunc_std * std;
        let upper = mean as f32 + trunc_std * std;
        for _ in 0..size {
            // rejeição simples com limite de tentativas
            let mut val = normal.sample(&mut rng) as f32;
            let mut tries = 0;
            while (val < lower || val > upper) && tries < 10 {
                val = normal.sample(&mut rng) as f32;
                tries += 1;
            }
            // se ainda inválido após tentativas, clamp como fallback
            if val < lower { val = lower; }
            if val > upper { val = upper; }
            values.push(val);
        }
        FlowTensors::new(&values, dims)
    }

    /// RandomGamma - distribuição Gamma com shape (alpha) e scale (theta)
    pub fn random_gamma(dims: &[i64], shape: f32, scale: f32) -> Option<Self> {
        if shape <= 0.0 || scale <= 0.0 {
            eprintln!("Erro: random_gamma requer shape>0 e scale>0");
            return None;
        }
        let size = dims.iter().product::<i64>() as usize;
        let mut rng = thread_rng();
        let gamma = Gamma::new(shape as f64, scale as f64).ok()?;
        let mut values = Vec::with_capacity(size);
        for _ in 0..size {
            values.push(gamma.sample(&mut rng) as f32);
        }
        FlowTensors::new(&values, dims)
    }

    /// Reshape do tensor
    pub fn reshape(&self, new_dims: &[i64]) -> Option<FlowTensors> {
        let old_size: i64 = self.dims.iter().product();
        let new_size: i64 = new_dims.iter().product();
        
        if old_size != new_size {
            return None;
        }

        let data = self.data()?.to_vec();
        FlowTensors::new(&data, new_dims)
    }

    /// Pad - aplica padding ao tensor. `paddings` deve ter comprimento igual ao rank
    /// e cada elemento é um par (before, after) em usize. Padding usa valor constante
    /// 0.0 por padrão. Para usar outro valor, chame `pad_v2`.
    pub fn pad(&self, paddings: &[(usize, usize)]) -> Option<FlowTensors> {
        FlowTensors::pad_v2(self, paddings, 0.0f32)
    }

    /// PadV2 - equivalente a Pad mas permite escolher o valor de preenchimento
    pub fn pad_v2(&self, paddings: &[(usize, usize)], constant: f32) -> Option<FlowTensors> {
        let rank = self.dims.len();
        if paddings.len() != rank { return None; }
        // Compute new dims
        let mut new_dims: Vec<i64> = Vec::with_capacity(rank);
        for (i, &d) in self.dims.iter().enumerate() {
            let (before, after) = paddings[i];
            new_dims.push((before + (d as usize) + after) as i64);
        }

        let new_size = new_dims.iter().product::<i64>() as usize;
        let mut out = vec![constant; new_size];

        // Helper to compute strides
        let old_strides: Vec<usize> = {
            let mut s = vec![1usize; rank];
            for i in (0..rank - 1).rev() {
                s[i] = s[i + 1] * (self.dims[i + 1] as usize);
            }
            s
        };
        let new_strides: Vec<usize> = {
            let mut s = vec![1usize; rank];
            for i in (0..rank - 1).rev() {
                s[i] = s[i + 1] * (new_dims[i + 1] as usize);
            }
            s
        };

        // Iterate old tensor and copy into new with offsets
        let old_size = self.dims.iter().product::<i64>() as usize;
        let old_data = self.data()?;
        for idx in 0..old_size {
            // compute multi-index
            let mut rem = idx;
            let mut multi = vec![0usize; rank];
            for i in 0..rank {
                multi[i] = rem / old_strides[i];
                rem = rem % old_strides[i];
            }
            // compute new flat index with paddings
            let mut new_flat = 0usize;
            for i in 0..rank {
                let before = paddings[i].0;
                let coord = before + multi[i];
                new_flat += coord * new_strides[i];
            }
            out[new_flat] = old_data[idx];
        }

        FlowTensors::new(&out, &new_dims)
    }

    /// Reverse - inverte elementos ao longo de eixos especificados.
    /// `axes` é um slice com índices de eixos a reverter. Se vazio, reverte todos os eixos.
    pub fn reverse(&self, axes: &[usize]) -> Option<FlowTensors> {
        let rank = self.dims.len();
        let mut axes_set = vec![false; rank];
        if axes.is_empty() {
            for a in 0..rank { axes_set[a] = true; }
        } else {
            for &a in axes { if a >= rank { return None; } axes_set[a] = true; }
        }

        let size = self.dims.iter().product::<i64>() as usize;
        let data = self.data()?;
        let mut out = vec![0.0f32; size];

        // Precompute strides
        let strides: Vec<usize> = {
            let mut s = vec![1usize; rank];
            for i in (0..rank - 1).rev() { s[i] = s[i + 1] * (self.dims[i + 1] as usize); }
            s
        };

        for flat in 0..size {
            // compute multi-index
            let mut rem = flat;
            let mut multi = vec![0usize; rank];
            for i in 0..rank {
                multi[i] = rem / strides[i];
                rem = rem % strides[i];
            }
            // compute reversed multi
            let mut rev_multi = multi.clone();
            for i in 0..rank {
                if axes_set[i] {
                    rev_multi[i] = (self.dims[i] as usize) - 1 - multi[i];
                }
            }
            // compute new flat index
            let mut new_flat = 0usize;
            for i in 0..rank { new_flat += rev_multi[i] * strides[i]; }
            out[new_flat] = data[flat];
        }

        FlowTensors::new(&out, &self.dims)
    }

    /// OneHot: cria um tensor one-hot a partir de índices 1D.
    /// Retorna tensor shape [indices.len(), depth]
    pub fn one_hot(indices: &[i64], depth: i64, on_value: f32, off_value: f32) -> Option<FlowTensors> {
        if depth <= 0 { return None; }
        let n = indices.len();
        let mut out = vec![off_value; n * (depth as usize)];
        for (i, &idx) in indices.iter().enumerate() {
            if idx < 0 || idx >= depth { continue; }
            let pos = i * (depth as usize) + (idx as usize);
            out[pos] = on_value;
        }
        FlowTensors::new(&out, &[n as i64, depth])
    }

    /// Where - seleção condicional: para cada elemento, se condition != 0 -> x else y
    /// Todos os tensores devem ter as mesmas dimensões.
    pub fn where_cond(condition: &FlowTensors, x: &FlowTensors, y: &FlowTensors) -> Option<FlowTensors> {
        if condition.dims() != x.dims() || x.dims() != y.dims() { return None; }
        let cond = condition.data()?;
        let a = x.data()?;
        let b = y.data()?;
        let size = cond.len();
        let mut out = vec![0.0f32; size];
        for i in 0..size {
            out[i] = if cond[i] != 0.0 { a[i] } else { b[i] };
        }
        FlowTensors::new(&out, x.dims())
    }

    // --- Array ops: Concat, Stack/Unstack, Split, Slice, Gather, Transpose ND ---

    // Helper: compute row-major strides (in elements) for dims
    fn compute_strides(dims: &[i64]) -> Vec<usize> {
        let rank = dims.len();
        let mut strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * (dims[i + 1] as usize);
        }
        strides
    }

    // Helper: convert flat index to multi-index using strides
    fn flat_to_multi(mut flat: usize, strides: &[usize], dims: &[i64]) -> Vec<usize> {
        let rank = dims.len();
        let mut multi = vec![0usize; rank];
        for i in 0..rank {
            multi[i] = flat / strides[i];
            flat = flat % strides[i];
        }
        multi
    }

    // Helper: convert multi-index to flat index
    fn multi_to_flat(multi: &[usize], strides: &[usize]) -> usize {
        multi.iter().zip(strides.iter()).map(|(m,s)| m * s).sum()
    }

    /// Transpose with an arbitrary permutation `perm` of axes.
    pub fn transpose_nd(&self, perm: &[usize]) -> Option<FlowTensors> {
        let rank = self.dims.len();
        if perm.len() != rank { return None; }
        // validate perm is a permutation
        let mut seen = vec![false; rank];
        for &p in perm { if p >= rank { return None; } seen[p] = true; }
        if seen.iter().any(|&b| !b) { return None; }

        let new_dims: Vec<i64> = perm.iter().map(|&p| self.dims[p]).collect();
        let old_strides = FlowTensors::compute_strides(&self.dims);
        let new_strides = FlowTensors::compute_strides(&new_dims);
        let size = self.dims.iter().product::<i64>() as usize;
        let data = self.data()?;
        let mut out = vec![0.0f32; size];

        for flat in 0..size {
            let multi_new = FlowTensors::flat_to_multi(flat, &new_strides, &new_dims);
            // build corresponding old multi by inverting perm
            let mut multi_old = vec![0usize; rank];
            for (i, &v) in perm.iter().enumerate() { multi_old[v] = multi_new[i]; }
            let old_flat = FlowTensors::multi_to_flat(&multi_old, &old_strides);
            out[flat] = data[old_flat];
        }

        FlowTensors::new(&out, &new_dims)
    }

    /// Concatenate a list of tensors along `axis`.
    /// All tensors must have the same rank and matching dims except on `axis`.
    pub fn concat(tensors: &[FlowTensors], axis: usize) -> Option<FlowTensors> {
        if tensors.is_empty() { return None; }
        let rank = tensors[0].dims.len();
        if axis >= rank { return None; }
        // verify compatibility and compute output dims
        let mut out_dims = tensors[0].dims.clone();
        out_dims[axis] = 0;
        for t in tensors {
            if t.dims.len() != rank { return None; }
            for i in 0..rank { if i != axis && t.dims[i] != tensors[0].dims[i] { return None; } }
            out_dims[axis] += t.dims[axis];
        }
        let out_size = out_dims.iter().product::<i64>() as usize;
        let out_strides = FlowTensors::compute_strides(&out_dims);
        let mut out = vec![0.0f32; out_size];

        // prepare cumulative sizes for axis
        let mut cum_sizes: Vec<i64> = Vec::with_capacity(tensors.len());
        let mut acc = 0i64;
        for t in tensors { cum_sizes.push(acc); acc += t.dims[axis]; }

        for flat in 0..out_size {
            let multi = FlowTensors::flat_to_multi(flat, &out_strides, &out_dims);
            let coord_axis = multi[axis] as i64;
            // find which tensor owns this axis coordinate
            let mut owner = 0usize;
            while owner + 1 < cum_sizes.len() && coord_axis >= cum_sizes[owner + 1] { owner += 1; }
            let local_axis = coord_axis - cum_sizes[owner];
            // build source multi index
            let mut src_multi: Vec<usize> = multi.clone();
            src_multi[axis] = local_axis as usize;
            let src_strides = FlowTensors::compute_strides(&tensors[owner].dims);
            let src_flat = FlowTensors::multi_to_flat(&src_multi, &src_strides);
            let src_data = tensors[owner].data()?;
            out[flat] = src_data[src_flat];
        }

        FlowTensors::new(&out, &out_dims)
    }

    /// Stack tensors by inserting a new axis at `axis` with length == tensors.len().
    /// All tensors must have identical shapes.
    pub fn stack(tensors: &[FlowTensors], axis: usize) -> Option<FlowTensors> {
        if tensors.is_empty() { return None; }
        let rank = tensors[0].dims.len();
        if axis > rank { return None; }
        for t in tensors { if t.dims != tensors[0].dims { return None; } }
        // build new dims
        let mut new_dims = tensors[0].dims.clone();
        new_dims.insert(axis, tensors.len() as i64);
        let new_size = new_dims.iter().product::<i64>() as usize;
        let new_strides = FlowTensors::compute_strides(&new_dims);
        let mut out = vec![0.0f32; new_size];

        // iterate output flat, map to source tensor and index
        let src_strides = FlowTensors::compute_strides(&tensors[0].dims);
        for flat in 0..new_size {
            let multi = FlowTensors::flat_to_multi(flat, &new_strides, &new_dims);
            let which = multi[axis];
            // build src multi by removing axis
            let mut src_multi = Vec::with_capacity(rank);
            for (i, &v) in multi.iter().enumerate() {
                if i == axis { continue; }
                src_multi.push(v);
            }
            let src_flat = FlowTensors::multi_to_flat(&src_multi, &src_strides);
            let src_data = tensors[which].data()?;
            out[flat] = src_data[src_flat];
        }
        FlowTensors::new(&out, &new_dims)
    }

    /// Unstack: split along axis producing tensors without that axis.
    pub fn unstack(&self, axis: usize) -> Option<Vec<FlowTensors>> {
        let rank = self.dims.len();
        if axis >= rank { return None; }
        let n = self.dims[axis] as usize;
        let mut results: Vec<Vec<f32>> = vec![Vec::new(); n];
        let mut out_dims: Vec<i64> = self.dims.clone();
        out_dims.remove(axis);
        let out_size = out_dims.iter().product::<i64>() as usize;
        for v in &mut results { v.resize(out_size, 0.0f32); }

        let strides = FlowTensors::compute_strides(&self.dims);
        let data = self.data()?;
        let size = data.len();
        for flat in 0..size {
            let multi = FlowTensors::flat_to_multi(flat, &strides, &self.dims);
            let which = multi[axis];
            // build out multi by removing axis
            let mut out_multi = Vec::with_capacity(rank - 1);
            for (i, &v) in multi.iter().enumerate() { if i != axis { out_multi.push(v); } }
            let out_strides = FlowTensors::compute_strides(&out_dims);
            let out_flat = FlowTensors::multi_to_flat(&out_multi, &out_strides);
            results[which].push(data[flat]);
        }

        let mut out_tensors = Vec::with_capacity(n);
        for v in results { out_tensors.push(FlowTensors::new(&v, &out_dims)?); }
        Some(out_tensors)
    }

    /// Split tensor into `num_splits` equal parts along `axis`.
    pub fn split(&self, num_splits: usize, axis: usize) -> Option<Vec<FlowTensors>> {
        if num_splits == 0 { return None; }
        let rank = self.dims.len();
        if axis >= rank { return None; }
        let dim = self.dims[axis] as usize;
        if dim % num_splits != 0 { return None; }
        let part = dim / num_splits;
        let mut parts = Vec::with_capacity(num_splits);
        for i in 0..num_splits {
            let mut new_dims = self.dims.clone();
            new_dims[axis] = part as i64;
            parts.push(new_dims);
        }
        // reuse concat-like mapping: for each output flat compute source
        let mut results: Vec<Vec<f32>> = parts.iter().map(|d| vec![0.0f32; d.iter().product::<i64>() as usize]).collect();
        let out_strides = FlowTensors::compute_strides(&self.dims);
        let data = self.data()?;
        let total = data.len();
        for flat in 0..total {
            let multi = FlowTensors::flat_to_multi(flat, &out_strides, &self.dims);
            let axis_coord = multi[axis];
            let which = axis_coord / part;
            let local_axis = axis_coord % part;
            // build dest multi
            let mut dest_multi = multi.clone(); dest_multi[axis] = local_axis;
            let dest_strides = FlowTensors::compute_strides(&parts[which]);
            let dest_flat = FlowTensors::multi_to_flat(&dest_multi, &dest_strides);
            results[which][dest_flat] = data[flat];
        }

        let mut out = Vec::with_capacity(num_splits);
        for (i, v) in results.into_iter().enumerate() { out.push(FlowTensors::new(&v, &parts[i])?); }
        Some(out)
    }

    /// Slice: begin and size must have length == rank. Size may contain -1 to mean 'till end'.
    pub fn slice(&self, begin: &[i64], size: &[i64]) -> Option<FlowTensors> {
        let rank = self.dims.len();
        if begin.len() != rank || size.len() != rank { return None; }
        let mut new_dims = vec![0i64; rank];
        for i in 0..rank {
            let b = begin[i];
            let s = if size[i] < 0 { self.dims[i] - b } else { size[i] };
            if b < 0 || b + s > self.dims[i] { return None; }
            new_dims[i] = s;
        }
        let new_size = new_dims.iter().product::<i64>() as usize;
        let new_strides = FlowTensors::compute_strides(&new_dims);
        let old_strides = FlowTensors::compute_strides(&self.dims);
        let mut out = vec![0.0f32; new_size];
        let data = self.data()?;
        for flat in 0..new_size {
            let multi = FlowTensors::flat_to_multi(flat, &new_strides, &new_dims);
            let mut old_multi = vec![0usize; rank];
            for i in 0..rank { old_multi[i] = (multi[i] as i64 + begin[i]) as usize; }
            let old_flat = FlowTensors::multi_to_flat(&old_multi, &old_strides);
            out[flat] = data[old_flat];
        }
        FlowTensors::new(&out, &new_dims)
    }

    /// Gather along `axis` using 1D indices.
    pub fn gather(params: &FlowTensors, indices: &[i64], axis: usize) -> Option<FlowTensors> {
        let rank = params.dims.len();
        if axis >= rank { return None; }
        let mut out_dims = params.dims.clone();
        out_dims[axis] = indices.len() as i64;
        let out_size = out_dims.iter().product::<i64>() as usize;
        let out_strides = FlowTensors::compute_strides(&out_dims);
        let mut out = vec![0.0f32; out_size];
        let params_strides = FlowTensors::compute_strides(&params.dims);
        let data = params.data()?;
        for flat in 0..out_size {
            let multi = FlowTensors::flat_to_multi(flat, &out_strides, &out_dims);
            let idx = indices[multi[axis]];
            if idx < 0 || idx >= params.dims[axis] { return None; }
            let mut src_multi = multi.clone();
            src_multi[axis] = idx as usize;
            let src_flat = FlowTensors::multi_to_flat(&src_multi, &params_strides);
            out[flat] = data[src_flat];
        }
        FlowTensors::new(&out, &out_dims)
    }

    /// GatherNd - very small convenience wrapper for common 2D gather_nd (list of indices vectors)
    pub fn gather_nd(params: &FlowTensors, indices_nd: &[Vec<i64>]) -> Option<FlowTensors> {
        // Build an output of shape [indices_nd.len(), params.dims[rank-1]] for simple cases
        let rank = params.dims.len();
        if rank == 0 { return None; }
        if indices_nd.is_empty() { return None; }
        // Each index should index into first `k` dims; we will support indexing full-rank-1 prefix
        let k = indices_nd[0].len();
        if k == 0 || k > rank { return None; }
        // output dims = [indices_nd.len(), params.dims[k..].product]
        let tail_dims = &params.dims[k..];
        let mut out_dims = vec![indices_nd.len() as i64];
        out_dims.extend_from_slice(tail_dims);
        let out_size = out_dims.iter().product::<i64>() as usize;
        let out_strides = FlowTensors::compute_strides(&out_dims);
        let mut out = vec![0.0f32; out_size];

        let params_strides = FlowTensors::compute_strides(&params.dims);
        let params_data = params.data()?;

        for i in 0..indices_nd.len() {
            let idx_vec = &indices_nd[i];
            if idx_vec.len() != k { return None; }
            // compute base offset in params
            let mut base = 0usize;
            for (d, &v) in idx_vec.iter().enumerate() {
                if v < 0 || v >= params.dims[d] { return None; }
                base += (v as usize) * params_strides[d];
            }
            // copy the remaining tail slice
            let tail_size = tail_dims.iter().product::<i64>() as usize;
            for j in 0..tail_size {
                out[i * tail_size + j] = params_data[base + j];
            }
        }

        FlowTensors::new(&out, &out_dims)
    }

    /// RandomShuffle - embaralha elementos.
    /// Para tensores 1D: embaralha todos os elementos.
    /// Para tensores com >=2 dims: embaralha ao longo da primeira dimensão (linhas).
    pub fn random_shuffle(&self) -> Option<FlowTensors> {
        let mut rng = thread_rng();
        let data = self.data()?.to_vec();
        if self.dims.len() == 1 {
            let mut v = data;
            v.as_mut_slice().shuffle(&mut rng);
            return FlowTensors::new(&v, &self.dims);
        }

        // Shuffle along first axis
        let first = self.dims[0] as usize;
        let inner_size: usize = self.dims[1..].iter().product::<i64>() as usize;
        if inner_size == 0 || first == 0 {
            return FlowTensors::new(&data, &self.dims);
        }
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(first);
        for i in 0..first {
            let start = i * inner_size;
            let end = start + inner_size;
            rows.push(data[start..end].to_vec());
        }
        rows.shuffle(&mut rng);
        let mut out: Vec<f32> = Vec::with_capacity(data.len());
        for row in rows.iter() {
            out.extend_from_slice(row);
        }
        FlowTensors::new(&out, &self.dims)
    }

    /// Multinomial: amostra índices a partir de um vetor de probabilidades (1D)
    /// Retorna um tensor 1D de inteiros (armazenados como f32) com o índice amostrado.
    pub fn multinomial(probs: &FlowTensors, num_samples: usize) -> Option<FlowTensors> {
        // probs deve ser 1D e somar > 0
        if probs.dims.len() != 1 {
            eprintln!("Erro: multinomial espera probs 1D");
            return None;
        }
        let p = probs.data()?;
        if p.is_empty() {
            eprintln!("Erro: multinomial probs vazio");
            return None;
        }
        // WeightedIndex requer f64 slice
        let weights: Vec<f64> = p.iter().map(|&v| v as f64).collect();
        let dist = WeightedIndex::new(weights.as_slice()).ok()?;
        let mut rng = thread_rng();
        let mut samples: Vec<f32> = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            let idx = dist.sample(&mut rng);
            samples.push(idx as f32);
        }
        FlowTensors::new(&samples, &[num_samples as i64])
    }
}

impl Drop for FlowTensors {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                crate::tensor_tensorflow::ffi::FreeTFTensor(self.ptr);
            }
            self.ptr = ptr::null_mut();
        }
    }
}

// Garante que FlowTensors é seguro para Send e Sync (necessário para FFI)
unsafe impl Send for FlowTensors {}
unsafe impl Sync for FlowTensors {}

/// Lightweight SparseTensor representation (CPU-only helpers)
/// This is a convenience helper that materializes a dense tensor under the hood
/// and returns a `FlowTensors` via the existing CreateTFTensor FFI. It is not
/// a zero-copy sparse implementation but provides the sparse ops listed in the
/// project TODO as usable examples.
#[derive(Clone)]
pub struct SparseTensor {
    /// Each entry in `indices` is a coordinate vector with length == rank
    pub indices: Vec<Vec<i64>>,
    /// Non-zero values corresponding to `indices`
    pub values: Vec<f32>,
    /// Dense shape of the sparse tensor
    pub shape: Vec<i64>,
}

impl SparseTensor {
    /// Create a new SparseTensor, returns None if inputs are inconsistent
    pub fn new(indices: Vec<Vec<i64>>, values: Vec<f32>, shape: Vec<i64>) -> Option<Self> {
        if indices.len() != values.len() { return None; }
        if indices.iter().any(|idx| idx.len() != shape.len()) { return None; }
        Some(SparseTensor { indices, values, shape })
    }

    /// Materialize as a dense row-major flattened Vec<f32>
    fn to_dense_flat(&self) -> Option<Vec<f32>> {
        let size: usize = self.shape.iter().product::<i64>() as usize;
        if size == 0 { return Some(vec![]); }
        let mut out = vec![0f32; size];
        let rank = self.shape.len();
        for (coord, &val) in self.indices.iter().zip(self.values.iter()) {
            // compute flat index
            let mut flat: usize = 0;
            for (i, &c) in coord.iter().enumerate() {
                let mut mul: usize = 1;
                for &s in &self.shape[i+1..] { mul *= s as usize; }
                flat += (c as usize) * mul;
            }
            if flat >= out.len() { return None; }
            out[flat] = val;
        }
        Some(out)
    }

    /// Convert to FlowTensors (dense) by materializing the sparse tensor
    pub fn to_flow(&self) -> Option<FlowTensors> {
        let data = self.to_dense_flat()?;
        FlowTensors::new(&data, &self.shape)
    }

    /// SparseAdd: adds two sparse tensors (shape must match) and returns a dense FlowTensors
    pub fn sparse_add(&self, other: &SparseTensor) -> Option<FlowTensors> {
        if self.shape != other.shape { eprintln!("sparse_add: shapes differ"); return None; }
        let a = self.to_dense_flat()?;
        let b = other.to_dense_flat()?;
        let mut out = a;
        for (i, v) in b.into_iter().enumerate() { out[i] += v; }
        FlowTensors::new(&out, &self.shape)
    }

    /// SparseTensorDenseMatMul: assumes `self` is 2D (MxK) and `dense` is 2D (KxN).
    /// Returns dense FlowTensors with shape [M, N].
    pub fn sparse_tensor_dense_matmul(&self, dense: &FlowTensors) -> Option<FlowTensors> {
        if self.shape.len() != 2 || dense.dims().len() != 2 { eprintln!("matmul expects 2D tensors"); return None; }
        let m = self.shape[0] as usize;
        let k = self.shape[1] as usize;
        let k2 = dense.dims()[0] as usize;
        let n = dense.dims()[1] as usize;
        if k != k2 { eprintln!("matmul inner dims mismatch"); return None; }
        let a = self.to_dense_flat()?; // length m*k
        let b = dense.data()?; // length k*n
        let mut out = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0f32;
                for p in 0..k {
                    sum += a[i*k + p] * b[p*n + j];
                }
                out[i*n + j] = sum;
            }
        }
        FlowTensors::new(&out, &[m as i64, n as i64])
    }

    /// SparseConcat along the first axis (axis=0). Returns a dense FlowTensors result.
    pub fn sparse_concat_axis0(tensors: &[SparseTensor]) -> Option<FlowTensors> {
        if tensors.is_empty() { return None; }
        let rank = tensors[0].shape.len();
        if rank == 0 { return None; }
        // ensure shapes compatible on axes 1..rank
        for t in tensors.iter() {
            if t.shape.len() != rank { return None; }
            if t.shape[1..] != tensors[0].shape[1..] { return None; }
        }
        let mut concatenated_indices: Vec<Vec<i64>> = Vec::new();
        let mut concatenated_values: Vec<f32> = Vec::new();
        let mut offset: i64 = 0;
        for t in tensors.iter() {
            for (idx, &val) in t.indices.iter().zip(t.values.iter()) {
                let mut new_idx = idx.clone();
                new_idx[0] += offset;
                concatenated_indices.push(new_idx);
                concatenated_values.push(val);
            }
            offset += t.shape[0];
        }
        let mut new_shape = tensors[0].shape.clone();
        new_shape[0] = offset;
        let st = SparseTensor::new(concatenated_indices, concatenated_values, new_shape)?;
        st.to_flow()
    }

    /// SparseSlice: returns a dense FlowTensors corresponding to the slice defined by `begin` and `size`
    /// `begin` and `size` must have the same rank as the tensor
    pub fn sparse_slice(&self, begin: &[i64], size: &[i64]) -> Option<FlowTensors> {
        if begin.len() != self.shape.len() || size.len() != self.shape.len() { return None; }
        // compute new indices/values inside slice
        let mut new_indices: Vec<Vec<i64>> = Vec::new();
        let mut new_values: Vec<f32> = Vec::new();
        for (idx, &val) in self.indices.iter().zip(self.values.iter()) {
            let mut inside = true;
            for d in 0..idx.len() {
                if idx[d] < begin[d] || idx[d] >= begin[d] + size[d] { inside = false; break; }
            }
            if inside {
                let new_idx: Vec<i64> = idx.iter().enumerate().map(|(d,&c)| c - begin[d]).collect();
                new_indices.push(new_idx);
                new_values.push(val);
            }
        }
        let st = SparseTensor::new(new_indices, new_values, size.to_vec())?;
        st.to_flow()
    }

    /// SparseReshape: reshape sparse tensor to new_shape (same number of elements)
    pub fn sparse_reshape(&self, new_shape: &[i64]) -> Option<FlowTensors> {
        let old_count: i64 = self.shape.iter().product();
        let new_count: i64 = new_shape.iter().product();
        if old_count != new_count { return None; }
        // map each indexed element by flat index -> new multi-index
        let mut new_indices: Vec<Vec<i64>> = Vec::with_capacity(self.indices.len());
        for idx in self.indices.iter() {
            // compute flat
            let mut flat: i64 = 0;
            for (i, &c) in idx.iter().enumerate() {
                let mut mul: i64 = 1;
                for &s in &self.shape[i+1..] { mul *= s; }
                flat += c * mul;
            }
            // compute new multi-index
            let mut rem = flat;
            let mut new_idx = vec![0i64; new_shape.len()];
            for i in 0..new_shape.len() {
                let mut mul: i64 = 1;
                for &s in &new_shape[i+1..] { mul *= s; }
                new_idx[i] = rem / mul;
                rem = rem % mul;
            }
            new_indices.push(new_idx);
        }
        let st = SparseTensor::new(new_indices, self.values.clone(), new_shape.to_vec())?;
        st.to_flow()
    }
}