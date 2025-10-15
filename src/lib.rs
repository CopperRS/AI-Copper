// AI Copper - Unified AI Library
// Combines LibTorch and TensorFlow capabilities

//==========================================
// TensorFlow Module
//==========================================
pub mod tensor_tensorflow;
pub use tensor_tensorflow::tensors_flow::FlowTensors;
pub use tensor_tensorflow::tensors_flow::TensorFlowModel;

//==========================================
// LibTorch Module
//==========================================
pub mod tensor_libortch;
pub use tensor_libortch::operators;
pub use tensor_libortch::tensor::{Tensor, Linear, Optimizer};

//==========================================
// Unified API Module
//==========================================
pub mod unified;
pub use unified::{Device, Backend, UnifiedTensor};