/// Representa o backend de computação usado
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// LibTorch (PyTorch C++)
    LibTorch,
    /// TensorFlow C API
    TensorFlow,
}

impl Default for Backend {
    fn default() -> Self {
        Backend::LibTorch
    }
}

impl Backend {
    /// Retorna o nome do backend como string
    pub fn name(&self) -> &str {
        match self {
            Backend::LibTorch => "LibTorch",
            Backend::TensorFlow => "TensorFlow",
        }
    }

    /// Verifica se o backend está disponível
    pub fn is_available(&self) -> bool {
        match self {
            Backend::LibTorch => true, // Assumindo que LibTorch está sempre disponível
            Backend::TensorFlow => true, // Assumindo que TensorFlow está sempre disponível
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
