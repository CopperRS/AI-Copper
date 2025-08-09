
//TensorFlow
pub mod tensor_tensorflow;
pub use tensor_tensorflow::tensors_flow::FlowTensors;


// LibTorch
pub mod tensor_libortch;
pub use tensor_libortch::operators;
pub use tensor_libortch::tensor::Tensor;
pub use tensor_libortch::tensor::Linear;
pub use tensor_libortch::tensor::Optimizer;