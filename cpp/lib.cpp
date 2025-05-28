#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>

extern "C" {

//This method allows the creation of custom tensors    
void* CreateMatrixTensor(float* values, int rows, int cols) {
    at::Tensor* tensor = new at::Tensor(torch::from_blob(values, {rows, cols}, torch::kFloat32).clone());
    return static_cast<void*>(tensor);
}

//This method creates tensors with values ​​1
void* CreateTensorOnes(int rows, int cols) {
    at::Tensor* tensor = new at::Tensor(torch::ones({rows, cols}, torch::kFloat32));
    return static_cast<void*>(tensor);
}

//This method creates tensors with values ​​randomly generated
void* CreateTensorRand(int rows, int cols) {
    at::Tensor* tensor = new at::Tensor(torch::rand({rows, cols}, torch::kFloat32));
    return static_cast<void*>(tensor);
}

//This method clear the memory of the tensor
void FreeTensor(void* ptr) {
    delete static_cast<at::Tensor*>(ptr);
}



/*This method returns a pointer to the data stored in the tensor.
Used to directly access the numeric values ​​stored in the tensor*/
float* TensorData(void* ptr) {
    at::Tensor* tensor = static_cast<at::Tensor*>(ptr);
    return tensor->data_ptr<float>();
}

//This method returns the number of rows in the tensor.
int TensorRows(void* ptr) {
    at::Tensor* tensor = static_cast<at::Tensor*>(ptr);
    return tensor->size(0);
}

//This method returns the number of columns in the tensor.
int TensorCols(void* ptr) {
    at::Tensor* tensor = static_cast<at::Tensor*>(ptr);
    return tensor->size(1);

}


}
