#include <torch/torch.h>

extern "C" {

void* CreateTensor(int rows, int cols) {
    at::Tensor* tensor = new at::Tensor(torch::rand({rows, cols}, torch::kFloat32));
    return static_cast<void*>(tensor);
}

void FreeTensor(void* ptr) {
    delete static_cast<at::Tensor*>(ptr);
}


float* TensorData(void* ptr) {
    at::Tensor* tensor = static_cast<at::Tensor*>(ptr);
    return tensor->data_ptr<float>();
}


int TensorRows(void* ptr) {
    at::Tensor* tensor = static_cast<at::Tensor*>(ptr);
    return tensor->size(0);
}

int TensorCols(void* ptr) {
    at::Tensor* tensor = static_cast<at::Tensor*>(ptr);
    return tensor->size(1);

}


}
