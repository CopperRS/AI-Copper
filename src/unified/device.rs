/// Representa o dispositivo onde o tensor será executado
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// CPU
    CPU,
    /// GPU/CUDA
    CUDA,
    /// GPU/ROCm (AMD)
    ROCm,
    /// TPU (TensorFlow)
    TPU,
}

impl Default for Device {
    fn default() -> Self {
        Device::CPU
    }
}

impl Device {
    /// Verifica se o dispositivo está disponível
    pub fn is_available(&self) -> bool {
        match self {
            Device::CPU => true,
            Device::CUDA => {
                // TODO: Implementar verificação real de CUDA
                false
            }
            Device::ROCm => {
                // TODO: Implementar verificação real de ROCm
                false
            }
            Device::TPU => {
                // TODO: Implementar verificação real de TPU
                false
            }
        }
    }

    /// Retorna uma string representando o dispositivo
    pub fn as_str(&self) -> &str {
        match self {
            Device::CPU => "cpu",
            Device::CUDA => "cuda",
            Device::ROCm => "rocm",
            Device::TPU => "tpu",
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
