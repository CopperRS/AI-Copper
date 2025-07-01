#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <ATen/Tensor.h>


// Exemplo para todas as funções FFI
extern "C" {

void* CreateLinear(int in_features, int out_features) {
    try {
        auto* linear = new torch::nn::LinearImpl(in_features, out_features);
        return static_cast<void*>(linear);
    } catch (...) {
        return nullptr;
    }
}

void* LinearForward(void* linear_ptr, void* input_tensor_ptr) {
    try {
        auto* linear = static_cast<torch::nn::LinearImpl*>(linear_ptr);
        auto* input = static_cast<at::Tensor*>(input_tensor_ptr);
        at::Tensor* output = new at::Tensor(linear->forward(*input));
        return static_cast<void*>(output);
    } catch (...) {
        return nullptr;
    }
}

void* MSELoss(void* prediction_ptr, void* target_ptr) {
    try {
        auto* prediction = static_cast<at::Tensor*>(prediction_ptr);
        auto* target = static_cast<at::Tensor*>(target_ptr);
        at::Tensor* loss = new at::Tensor(torch::mse_loss(*prediction, *target));
        return static_cast<void*>(loss);
    } catch (...) {
        return nullptr;
    }
}

void* CreateSGD(void* linear_ptr, float lr) {
    try {
        auto* linear = static_cast<torch::nn::LinearImpl*>(linear_ptr);
        auto* optimizer = new torch::optim::SGD(linear->parameters(), lr);
        return static_cast<void*>(optimizer);
    } catch (...) {
        return nullptr;
    }
}

void Backward(void* loss_ptr) {
    try {
        auto* loss = static_cast<at::Tensor*>(loss_ptr);
        loss->backward();
    } catch (...) {
        // Do nothing
    }
}

void OptimizerStep(void* optimizer_ptr) {
    try {
        auto* optimizer = static_cast<torch::optim::Optimizer*>(optimizer_ptr);
        optimizer->step();
    } catch (...) {
        // Do nothing
    }
}

void FreeOptimizer(void* ptr) {
    try {
        delete static_cast<torch::optim::Optimizer*>(ptr);
    } catch (...) {
        // Do nothing
    }
}

void* CreateMatrixTensor(float* values, int rows, int cols) {
    try {
        at::Tensor* tensor = new at::Tensor(torch::from_blob(values, {rows, cols}, torch::kFloat32).clone());
        return static_cast<void*>(tensor);
    } catch (...) {
        return nullptr;
    }
}

void* CreateTensorOnes(int rows, int cols) {
    try {
        at::Tensor* tensor = new at::Tensor(torch::ones({rows, cols}, torch::kFloat32));
        return static_cast<void*>(tensor);
    } catch (...) {
        return nullptr;
    }
}

void* CreateTensorRand(int rows, int cols) {
    try {
        at::Tensor* tensor = new at::Tensor(torch::rand({rows, cols}, torch::kFloat32));
        return static_cast<void*>(tensor);
    } catch (...) {
        return nullptr;
    }
}

void FreeTensor(void* ptr) {
    try {
        delete static_cast<at::Tensor*>(ptr);
    } catch (...) {
        // Do nothing
    }
}

float* TensorData(void* ptr) {
    try {
        at::Tensor* tensor = static_cast<at::Tensor*>(ptr);
        return tensor->data_ptr<float>();
    } catch (...) {
        return nullptr;
    }
}

int TensorRows(void* ptr) {
    try {
        at::Tensor* tensor = static_cast<at::Tensor*>(ptr);
        return tensor->size(0);
    } catch (...) {
        return -1;
    }
}

int TensorCols(void* ptr) {
    try {
        at::Tensor* tensor = static_cast<at::Tensor*>(ptr);
        return tensor->size(1);
    } catch (...) {
        return -1;
    }
}

}
