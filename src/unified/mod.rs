// Unified API Module
// Provides a unified interface for both LibTorch and TensorFlow

mod tensor;
mod device;
mod backend;

pub use tensor::UnifiedTensor;
pub use device::Device;
pub use backend::Backend;
