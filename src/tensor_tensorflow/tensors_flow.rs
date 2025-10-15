use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void, c_int};
use std::ptr;

pub struct FlowTensors {
    ptr: *mut c_void, // Ponteiro para TF_Tensor*
    dims: Vec<i64>,   // Dimensões do tensor (suporta qualquer número de dimensões)
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
                    Some(FlowTensors {
                        ptr,
                        dims,
                    })
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
            Some(FlowTensors {
                ptr: tensor_ptr,
                dims: dims.to_vec(),
            })
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
        F: Fn(f32) -> f32,
    {
        let data = self.data()?;
        let mapped: Vec<f32> = data.iter().map(|&x| f(x)).collect();
        FlowTensors::new(&mapped, &self.dims)
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