#define _CRT_SECURE_NO_WARNINGS
#include <tensorflow/c/c_api.h>
#include <string>
#include <cstring>
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <vector>
#include <numeric>

extern "C" {

#if defined(_WIN32)
  #define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
  #define EXPORT __attribute__((visibility("default")))
#else
  #define EXPORT
#endif


struct ModelHandle {
    TF_Session* session;
    TF_Graph* graph;
};

// TensorFlow C API functions
EXPORT const char* VersionTF() { 
    return TF_Version();
}

EXPORT void* LoadSavedModel(const char* model_path, const char* tags) {
    try {
        TF_Status* status = TF_NewStatus();
        TF_SessionOptions* session_opts = TF_NewSessionOptions();
        TF_Graph* graph = TF_NewGraph();

        const char* tag_array[] = {tags};
        TF_Session* session = TF_LoadSessionFromSavedModel(
            session_opts, nullptr, model_path, tag_array, 1, graph, nullptr, status);

        if (TF_GetCode(status) != TF_OK) {
            fprintf(stderr, "Erro ao carregar SavedModel: %s\n", TF_Message(status));
            TF_DeleteStatus(status);
            TF_DeleteSessionOptions(session_opts);
            TF_DeleteGraph(graph);
            return nullptr;
        }

        TF_DeleteStatus(status);
        TF_DeleteSessionOptions(session_opts);

        ModelHandle* handle = new ModelHandle{session, graph};
        return static_cast<void*>(handle);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em LoadSavedModel: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* RunSession(void* model_handle, const char** input_names, void** input_tensors, int num_inputs,
                        const char** output_names, void** output_tensors, int num_outputs) {
    try {
        auto* handle = static_cast<ModelHandle*>(model_handle);
        if (!handle || !handle->session || !handle->graph) {
            fprintf(stderr, "Erro: ModelHandle inválido\n");
            return nullptr;
        }
        TF_Session* session = handle->session;
        TF_Graph* graph = handle->graph;
        TF_Status* status = TF_NewStatus();

        std::vector<TF_Tensor*> inputs(num_inputs);
        for (int i = 0; i < num_inputs; ++i) {
            inputs[i] = static_cast<TF_Tensor*>(input_tensors[i]);
            if (!inputs[i]) {
                fprintf(stderr, "Erro: Tensor de entrada %d inválido\n", i);
                TF_DeleteStatus(status);
                return nullptr;
            }
            // Depuração da entrada
            int num_dims = TF_NumDims(inputs[i]);
            printf("Entrada %d: %d dimensões\n", i, num_dims);
            for (int j = 0; j < num_dims; ++j) {
                printf("Dimensão %d: %lld\n", j, TF_Dim(inputs[i], j));
            }
        }

        std::vector<TF_Output> input_ops(num_inputs);
        for (int i = 0; i < num_inputs; ++i) {
            TF_Operation* op = TF_GraphOperationByName(graph, input_names[i]);
            if (!op) {
                fprintf(stderr, "Erro: Operação de entrada %s não encontrada\n", input_names[i]);
                TF_DeleteStatus(status);
                return nullptr;
            }
            input_ops[i] = {op, 0};
        }

        std::vector<TF_Output> output_ops(num_outputs);
        for (int i = 0; i < num_outputs; ++i) {
            TF_Operation* op = TF_GraphOperationByName(graph, output_names[i]);
            if (!op) {
                fprintf(stderr, "Erro: Operação de saída %s não encontrada\n", output_names[i]);
                TF_DeleteStatus(status);
                return nullptr;
            }
            output_ops[i] = {op, 0};
        }

        std::vector<TF_Tensor*> tf_output_tensors(num_outputs, nullptr);
        for (int i = 0; i < num_outputs; ++i) {
            output_tensors[i] = nullptr;
        }

        TF_SessionRun(
            session, nullptr,
            input_ops.data(), inputs.data(), num_inputs,
            output_ops.data(), tf_output_tensors.data(), num_outputs,
            nullptr, 0, nullptr, status);

        if (TF_GetCode(status) != TF_OK) {
            fprintf(stderr, "Erro ao executar sessão: %s\n", TF_Message(status));
            TF_DeleteStatus(status);
            return nullptr;
        }

        // Depuração da saída
        for (int i = 0; i < num_outputs; ++i) {
            if (tf_output_tensors[i]) {
                int num_dims = TF_NumDims(tf_output_tensors[i]);
                printf("Saída %d: %d dimensões\n", i, num_dims);
                for (int j = 0; j < num_dims; ++j) {
                    printf("Dimensão %d: %lld\n", j, TF_Dim(tf_output_tensors[i], j));
                }
                size_t num_elements = 1;
                for (int j = 0; j < num_dims; ++j) {
                    num_elements *= TF_Dim(tf_output_tensors[i], j);
                }
                float* data = static_cast<float*>(TF_TensorData(tf_output_tensors[i]));
                printf("Primeiros 10 valores (máx %zu): [", num_elements);
                for (size_t k = 0; k < std::min(num_elements, size_t(10)); ++k) {
                    printf("%f", data[k]);
                    if (k < std::min(num_elements, size_t(10)) - 1) printf(", ");
                }
                printf("]\n");
            } else {
                printf("Saída %d: Tensor nulo\n", i);
            }
        }

        for (int i = 0; i < num_outputs; ++i) {
            output_tensors[i] = static_cast<void*>(tf_output_tensors[i]);
        }

        TF_DeleteStatus(status);
        return static_cast<void*>(tf_output_tensors.data());
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em RunSession: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* CreateTFTensor(float* values, int64_t* dims, int num_dims) {
    try {
        if (!values || !dims || num_dims <= 0) {
            fprintf(stderr, "Erro: Parâmetros inválidos em CreateTFTensor\n");
            return nullptr;
        }
        
        // Calcula o número total de elementos
        size_t num_elements = std::accumulate(dims, dims + num_dims, size_t(1), std::multiplies<size_t>());
        size_t data_size = sizeof(float) * num_elements;
        
        // Aloca memória que será gerenciada pelo TensorFlow
        float* tensor_data = static_cast<float*>(malloc(data_size));
        if (!tensor_data) {
            fprintf(stderr, "Erro: Falha ao alocar memória para TF_Tensor\n");
            return nullptr;
        }
        
        // Copia os dados do ponteiro temporário do Rust para memória própria
        memcpy(tensor_data, values, data_size);
        
        // Cria tensor com deallocator que libera a memória alocada
        TF_Tensor* tensor = TF_NewTensor(
            TF_FLOAT,
            dims,
            num_dims,
            tensor_data,
            data_size,
            [](void* data, size_t, void*) { free(data); },
            nullptr);
            
        if (!tensor) {
            fprintf(stderr, "Erro: Falha ao criar TF_Tensor\n");
            free(tensor_data);
            return nullptr;
        }
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em CreateTFTensor: %s\n", e.what());
        return nullptr;
    }
}

EXPORT float* GetTensorData(void* tensor_ptr) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em GetTensorData\n");
            return nullptr;
        }
        
        // Verifica se o tipo do tensor é float
        if (TF_TensorType(tensor) != TF_FLOAT) {
            fprintf(stderr, "Erro: Tensor não é do tipo TF_FLOAT em GetTensorData\n");
            return nullptr;
        }
        
        float* data = static_cast<float*>(TF_TensorData(tensor));
        if (!data) {
            fprintf(stderr, "Erro: TF_TensorData retornou nullptr\n");
            return nullptr;
        }
        
        return data;
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em GetTensorData: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void FreeTFTensor(void* tensor_ptr) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (tensor) {
            TF_DeleteTensor(tensor);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em FreeTFTensor: %s\n", e.what());
    }
}

EXPORT void FreeModel(void* model_handle) {
    try {
        auto* handle = static_cast<ModelHandle*>(model_handle);
        if (!handle) {
            fprintf(stderr, "Erro: ModelHandle inválido em FreeModel\n");
            return;
        }
        TF_Status* status = TF_NewStatus();
        if (handle->session) {
            TF_CloseSession(handle->session, status);
            TF_DeleteSession(handle->session, status);
        }
        if (handle->graph) {
            TF_DeleteGraph(handle->graph);
        }
        TF_DeleteStatus(status);
        delete handle;
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em FreeModel: %s\n", e.what());
    }
}








// Torch C++ API functions
EXPORT void* CreateLinear(int in_features, int out_features) {
    try {
        auto* linear = new torch::nn::LinearImpl(in_features, out_features);
        // Ensure gradients are enabled for parameters
        for (auto& param : linear->parameters()) {
            param.set_requires_grad(true);
        }
        return static_cast<void*>(linear);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em CreateLinear: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* LinearForward(void* linear_ptr, void* input_tensor_ptr) {
    try {
        auto* linear = static_cast<torch::nn::LinearImpl*>(linear_ptr);
        auto* input = static_cast<at::Tensor*>(input_tensor_ptr);
        if (!linear || !input) {
            fprintf(stderr, "Erro: Ponteiros inválidos em LinearForward\n");
            return nullptr;
        }
        
        // Debug: Print input shape
        printf("LinearForward - Input shape: [");
        for (int64_t i = 0; i < input->dim(); ++i) {
            printf("%lld", input->size(i));
            if (i < input->dim() - 1) printf(", ");
        }
        printf("]\n");
        
        at::Tensor* output = new at::Tensor(linear->forward(*input));
        
        // Debug: Print output shape
        printf("LinearForward - Output shape: [");
        for (int64_t i = 0; i < output->dim(); ++i) {
            printf("%lld", output->size(i));
            if (i < output->dim() - 1) printf(", ");
        }
        printf("]\n");
        
        return static_cast<void*>(output);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em LinearForward: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* MSELoss(void* prediction_tensor_ptr, void* target_tensor_ptr) {
    try {
        auto* prediction = static_cast<at::Tensor*>(prediction_tensor_ptr);
        auto* target = static_cast<at::Tensor*>(target_tensor_ptr);
        if (!prediction || !target) {
            fprintf(stderr, "Erro: Tensores inválidos em MSELoss\n");
            return nullptr;
        }
        
        // Debug: Print shapes
        printf("MSELoss - Prediction shape: [");
        for (int64_t i = 0; i < prediction->dim(); ++i) {
            printf("%lld", prediction->size(i));
            if (i < prediction->dim() - 1) printf(", ");
        }
        printf("], Target shape: [");
        for (int64_t i = 0; i < target->dim(); ++i) {
            printf("%lld", target->size(i));
            if (i < target->dim() - 1) printf(", ");
        }
        printf("]\n");
        
        at::Tensor* loss = new at::Tensor(torch::mse_loss(*prediction, *target));
        
        // Debug: Print loss info
        printf("Loss computed - shape: [");
        for (int64_t i = 0; i < loss->dim(); ++i) {
            printf("%lld", loss->size(i));
            if (i < loss->dim() - 1) printf(", ");
        }
        printf("], value: %f\n", loss->item<float>());
        
        return static_cast<void*>(loss);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em MSELoss: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* CreateSGD(void* linear_ptr, float lr) {
    try {
        auto* linear = static_cast<torch::nn::LinearImpl*>(linear_ptr);
        if (!linear) {
            fprintf(stderr, "Erro: Ponteiro linear inválido em CreateSGD\n");
            return nullptr;
        }
        auto* optimizer = new torch::optim::SGD(linear->parameters(), lr);
        return static_cast<void*>(optimizer);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em CreateSGD: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void Backward(void* loss_ptr) {
    try {
        auto* loss = static_cast<at::Tensor*>(loss_ptr);
        if (!loss) {
            fprintf(stderr, "Erro: Tensor de perda inválido em Backward\n");
            return;
        }
        loss->backward();
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em Backward: %s\n", e.what());
    }
}

EXPORT void OptimizerStep(void* optimizer_ptr) {
    try {
        auto* optimizer = static_cast<torch::optim::Optimizer*>(optimizer_ptr);
        if (!optimizer) {
            fprintf(stderr, "Erro: Otimizador inválido em OptimizerStep\n");
            return;
        }
        optimizer->step();
        optimizer->zero_grad();  // Reset gradients after step
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em OptimizerStep: %s\n", e.what());
    }
}

EXPORT void OptimizerZeroGrad(void* optimizer_ptr) {
    try {
        auto* optimizer = static_cast<torch::optim::Optimizer*>(optimizer_ptr);
        if (!optimizer) {
            fprintf(stderr, "Erro: Otimizador inválido em OptimizerZeroGrad\n");
            return;
        }
        optimizer->zero_grad();
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em OptimizerZeroGrad: %s\n", e.what());
    }
}

EXPORT void FreeOptimizer(void* ptr) {
    try {
        auto* optimizer = static_cast<torch::optim::Optimizer*>(ptr);
        if (optimizer) {
            delete optimizer;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em FreeOptimizer: %s\n", e.what());
    }
}

EXPORT void* CreateMatrixTensor(float* values, int rows, int cols) {
    try {
        if (!values || rows <= 0 || cols <= 0) {
            fprintf(stderr, "Erro: Parâmetros inválidos em CreateMatrixTensor\n");
            return nullptr;
        }
        at::Tensor* tensor = new at::Tensor(torch::from_blob(values, {rows, cols}, torch::kFloat32).clone());
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em CreateMatrixTensor: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* CreateTensorOnes(int rows, int cols) {
    try {
        if (rows <= 0 || cols <= 0) {
            fprintf(stderr, "Erro: Dimensões inválidas em CreateTensorOnes\n");
            return nullptr;
        }
        at::Tensor* tensor = new at::Tensor(torch::ones({rows, cols}, torch::kFloat32));
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em CreateTensorOnes: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* CreateTensorRand(int rows, int cols) {
    try {
        if (rows <= 0 || cols <= 0) {
            fprintf(stderr, "Erro: Dimensões inválidas em CreateTensorRand\n");
            return nullptr;
        }
        at::Tensor* tensor = new at::Tensor(torch::rand({rows, cols}, torch::kFloat32));
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em CreateTensorRand: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void FreeTensor(void* ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(ptr);
        if (tensor) {
            delete tensor;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em FreeTensor: %s\n", e.what());
    }
}

EXPORT float* TensorData(void* ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorData\n");
            return nullptr;
        }
        return tensor->data_ptr<float>();
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorData: %s\n", e.what());
        return nullptr;
    }
}

EXPORT int TensorRows(void* ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorRows\n");
            return -1;
        }
        // Verificar se o tensor tem dimensões suficientes
        if (tensor->dim() == 0) {
            // Tensor escalar (0-D) - retorna 1
            return 1;
        }
        if (tensor->dim() == 1) {
            // Tensor 1-D - retorna o tamanho da primeira dimensão
            return tensor->size(0);
        }
        // Tensor 2-D ou maior - retorna o tamanho da primeira dimensão
        return tensor->size(0);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorRows: %s\n", e.what());
        return -1;
    }
}

EXPORT int TensorCols(void* ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorCols\n");
            return -1;
        }
        // Verificar se o tensor tem dimensões suficientes
        if (tensor->dim() == 0) {
            // Tensor escalar (0-D) - retorna 1
            return 1;
        }
        if (tensor->dim() == 1) {
            // Tensor 1-D - consideramos como vetor coluna
            return 1;
        }
        // Tensor 2-D ou maior - retorna o tamanho da segunda dimensão
        return tensor->size(1);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorCols: %s\n", e.what());
        return -1;
    }
}

// Activation Functions
EXPORT void* TensorReLU(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorReLU\n");
            return nullptr;
        }
        at::Tensor* result = new at::Tensor(torch::relu(*tensor));
        return static_cast<void*>(result);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorReLU: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* TensorSigmoid(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorSigmoid\n");
            return nullptr;
        }
        at::Tensor* result = new at::Tensor(torch::sigmoid(*tensor));
        return static_cast<void*>(result);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorSigmoid: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* TensorTanh(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorTanh\n");
            return nullptr;
        }
        at::Tensor* result = new at::Tensor(torch::tanh(*tensor));
        return static_cast<void*>(result);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorTanh: %s\n", e.what());
        return nullptr;
    }
}

// Mathematical Functions
EXPORT void* TensorSin(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorSin\n");
            return nullptr;
        }
        at::Tensor* result = new at::Tensor(torch::sin(*tensor));
        return static_cast<void*>(result);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorSin: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* TensorCos(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorCos\n");
            return nullptr;
        }
        at::Tensor* result = new at::Tensor(torch::cos(*tensor));
        return static_cast<void*>(result);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorCos: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* TensorExp(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorExp\n");
            return nullptr;
        }
        at::Tensor* result = new at::Tensor(torch::exp(*tensor));
        return static_cast<void*>(result);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorExp: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* TensorLog(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorLog\n");
            return nullptr;
        }
        at::Tensor* result = new at::Tensor(torch::log(*tensor));
        return static_cast<void*>(result);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorLog: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* TensorSqrt(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorSqrt\n");
            return nullptr;
        }
        at::Tensor* result = new at::Tensor(torch::sqrt(*tensor));
        return static_cast<void*>(result);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorSqrt: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* TensorAbs(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorAbs\n");
            return nullptr;
        }
        at::Tensor* result = new at::Tensor(torch::abs(*tensor));
        return static_cast<void*>(result);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorAbs: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* TensorPow(void* tensor_ptr, float exponent) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorPow\n");
            return nullptr;
        }
        at::Tensor* result = new at::Tensor(torch::pow(*tensor, exponent));
        return static_cast<void*>(result);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorPow: %s\n", e.what());
        return nullptr;
    }
}

// Tensor Creation Functions
EXPORT void* CreateTensorRandn(int rows, int cols) {
    try {
        if (rows <= 0 || cols <= 0) {
            fprintf(stderr, "Erro: Dimensões inválidas em CreateTensorRandn\n");
            return nullptr;
        }
        at::Tensor* tensor = new at::Tensor(torch::randn({rows, cols}, torch::kFloat32));
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em CreateTensorRandn: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* CreateTensorZeros(int rows, int cols) {
    try {
        if (rows <= 0 || cols <= 0) {
            fprintf(stderr, "Erro: Dimensões inválidas em CreateTensorZeros\n");
            return nullptr;
        }
        at::Tensor* tensor = new at::Tensor(torch::zeros({rows, cols}, torch::kFloat32));
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em CreateTensorZeros: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* CreateTensorEye(int size) {
    try {
        if (size <= 0) {
            fprintf(stderr, "Erro: Tamanho inválido em CreateTensorEye\n");
            return nullptr;
        }
        at::Tensor* tensor = new at::Tensor(torch::eye(size, torch::kFloat32));
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em CreateTensorEye: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* TensorZerosLike(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorZerosLike\n");
            return nullptr;
        }
        at::Tensor* result = new at::Tensor(torch::zeros_like(*tensor));
        return static_cast<void*>(result);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorZerosLike: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* TensorOnesLike(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorOnesLike\n");
            return nullptr;
        }
        at::Tensor* result = new at::Tensor(torch::ones_like(*tensor));
        return static_cast<void*>(result);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorOnesLike: %s\n", e.what());
        return nullptr;
    }
}

// Statistical Functions
EXPORT float TensorStd(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorStd\n");
            return 0.0f;
        }
        return tensor->std().item<float>();
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorStd: %s\n", e.what());
        return 0.0f;
    }
}

EXPORT float TensorVar(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorVar\n");
            return 0.0f;
        }
        return tensor->var().item<float>();
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorVar: %s\n", e.what());
        return 0.0f;
    }
}

EXPORT int TensorArgmax(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorArgmax\n");
            return -1;
        }
        return tensor->argmax().item<int>();
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorArgmax: %s\n", e.what());
        return -1;
    }
}

EXPORT int TensorArgmin(void* tensor_ptr) {
    try {
        auto* tensor = static_cast<at::Tensor*>(tensor_ptr);
        if (!tensor) {
            fprintf(stderr, "Erro: Tensor inválido em TensorArgmin\n");
            return -1;
        }
        return tensor->argmin().item<int>();
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em TensorArgmin: %s\n", e.what());
        return -1;
    }
}

// Loss Functions
EXPORT void* CrossEntropyLoss(void* prediction_tensor_ptr, void* target_tensor_ptr) {
    try {
        auto* prediction = static_cast<at::Tensor*>(prediction_tensor_ptr);
        auto* target = static_cast<at::Tensor*>(target_tensor_ptr);
        if (!prediction || !target) {
            fprintf(stderr, "Erro: Tensores inválidos em CrossEntropyLoss\n");
            return nullptr;
        }
        at::Tensor* loss = new at::Tensor(torch::cross_entropy_loss(*prediction, *target));
        return static_cast<void*>(loss);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em CrossEntropyLoss: %s\n", e.what());
        return nullptr;
    }
}

// Adam Optimizer
EXPORT void* CreateAdam(void* linear_ptr, float lr) {
    try {
        auto* linear = static_cast<torch::nn::LinearImpl*>(linear_ptr);
        if (!linear) {
            fprintf(stderr, "Erro: Ponteiro linear inválido em CreateAdam\n");
            return nullptr;
        }
        auto* optimizer = new torch::optim::Adam(linear->parameters(), lr);
        return static_cast<void*>(optimizer);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em CreateAdam: %s\n", e.what());
        return nullptr;
    }
}

} // Fim do extern "C"