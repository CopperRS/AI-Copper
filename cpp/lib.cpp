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
#include <mutex>
#include <unordered_map>
#include <sstream>

#include <cmath>
#include <algorithm>
#include <random>

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

// ----- Typed tensor creation helpers -----
EXPORT void* CreateTFTensor_double(double* values, int64_t* dims, int num_dims) {
    try {
        if (!values || !dims || num_dims <= 0) {
            fprintf(stderr, "Erro: Parâmetros inválidos em CreateTFTensor_double\n");
            return nullptr;
        }
        size_t num_elements = std::accumulate(dims, dims + num_dims, size_t(1), std::multiplies<size_t>());
        size_t data_size = sizeof(double) * num_elements;
        double* tensor_data = static_cast<double*>(malloc(data_size));
        if (!tensor_data) { fprintf(stderr, "Erro: Falha ao alocar memória para TF_Tensor\n"); return nullptr; }
        memcpy(tensor_data, values, data_size);
        TF_Tensor* tensor = TF_NewTensor(
            TF_DOUBLE,
            dims,
            num_dims,
            tensor_data,
            data_size,
            [](void* data, size_t, void*) { free(data); },
            nullptr);
        if (!tensor) { free(tensor_data); fprintf(stderr, "Erro: Falha ao criar TF_Tensor double\n"); return nullptr; }
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) {
        fprintf(stderr, "Erro em CreateTFTensor_double: %s\n", e.what());
        return nullptr;
    }
}

EXPORT void* CreateTFTensor_int32(int32_t* values, int64_t* dims, int num_dims) {
    try {
        if (!values || !dims || num_dims <= 0) { fprintf(stderr, "Erro: Parâmetros inválidos em CreateTFTensor_int32\n"); return nullptr; }
        size_t num_elements = std::accumulate(dims, dims + num_dims, size_t(1), std::multiplies<size_t>());
        size_t data_size = sizeof(int32_t) * num_elements;
        int32_t* tensor_data = static_cast<int32_t*>(malloc(data_size));
        if (!tensor_data) { fprintf(stderr, "Erro: Falha ao alocar memória para TF_Tensor\n"); return nullptr; }
        memcpy(tensor_data, values, data_size);
        TF_Tensor* tensor = TF_NewTensor(
            TF_INT32,
            dims,
            num_dims,
            tensor_data,
            data_size,
            [](void* data, size_t, void*) { free(data); },
            nullptr);
        if (!tensor) { free(tensor_data); fprintf(stderr, "Erro: Falha ao criar TF_Tensor int32\n"); return nullptr; }
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) { fprintf(stderr, "Erro em CreateTFTensor_int32: %s\n", e.what()); return nullptr; }
}

EXPORT void* CreateTFTensor_int64(int64_t* values, int64_t* dims, int num_dims) {
    try {
        if (!values || !dims || num_dims <= 0) { fprintf(stderr, "Erro: Parâmetros inválidos em CreateTFTensor_int64\n"); return nullptr; }
        size_t num_elements = std::accumulate(dims, dims + num_dims, size_t(1), std::multiplies<size_t>());
        size_t data_size = sizeof(int64_t) * num_elements;
        int64_t* tensor_data = static_cast<int64_t*>(malloc(data_size));
        if (!tensor_data) { fprintf(stderr, "Erro: Falha ao alocar memória para TF_Tensor\n"); return nullptr; }
        memcpy(tensor_data, values, data_size);
        TF_Tensor* tensor = TF_NewTensor(
            TF_INT64,
            dims,
            num_dims,
            tensor_data,
            data_size,
            [](void* data, size_t, void*) { free(data); },
            nullptr);
        if (!tensor) { free(tensor_data); fprintf(stderr, "Erro: Falha ao criar TF_Tensor int64\n"); return nullptr; }
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) { fprintf(stderr, "Erro em CreateTFTensor_int64: %s\n", e.what()); return nullptr; }
}

EXPORT void* CreateTFTensor_int8(int8_t* values, int64_t* dims, int num_dims) {
    try {
        if (!values || !dims || num_dims <= 0) { fprintf(stderr, "Erro: Parâmetros inválidos em CreateTFTensor_int8\n"); return nullptr; }
        size_t num_elements = std::accumulate(dims, dims + num_dims, size_t(1), std::multiplies<size_t>());
        size_t data_size = sizeof(int8_t) * num_elements;
        int8_t* tensor_data = static_cast<int8_t*>(malloc(data_size));
        if (!tensor_data) { fprintf(stderr, "Erro: Falha ao alocar memória para TF_Tensor\n"); return nullptr; }
        memcpy(tensor_data, values, data_size);
        TF_Tensor* tensor = TF_NewTensor(
            TF_INT8,
            dims,
            num_dims,
            tensor_data,
            data_size,
            [](void* data, size_t, void*) { free(data); },
            nullptr);
        if (!tensor) { free(tensor_data); fprintf(stderr, "Erro: Falha ao criar TF_Tensor int8\n"); return nullptr; }
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) { fprintf(stderr, "Erro em CreateTFTensor_int8: %s\n", e.what()); return nullptr; }
}

EXPORT void* CreateTFTensor_int16(int16_t* values, int64_t* dims, int num_dims) {
    try {
        if (!values || !dims || num_dims <= 0) { fprintf(stderr, "Erro: Parâmetros inválidos em CreateTFTensor_int16\n"); return nullptr; }
        size_t num_elements = std::accumulate(dims, dims + num_dims, size_t(1), std::multiplies<size_t>());
        size_t data_size = sizeof(int16_t) * num_elements;
        int16_t* tensor_data = static_cast<int16_t*>(malloc(data_size));
        if (!tensor_data) { fprintf(stderr, "Erro: Falha ao alocar memória para TF_Tensor\n"); return nullptr; }
        memcpy(tensor_data, values, data_size);
        TF_Tensor* tensor = TF_NewTensor(
            TF_INT16,
            dims,
            num_dims,
            tensor_data,
            data_size,
            [](void* data, size_t, void*) { free(data); },
            nullptr);
        if (!tensor) { free(tensor_data); fprintf(stderr, "Erro: Falha ao criar TF_Tensor int16\n"); return nullptr; }
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) { fprintf(stderr, "Erro em CreateTFTensor_int16: %s\n", e.what()); return nullptr; }
}

EXPORT void* CreateTFTensor_uint8(uint8_t* values, int64_t* dims, int num_dims) {
    try {
        if (!values || !dims || num_dims <= 0) { fprintf(stderr, "Erro: Parâmetros inválidos em CreateTFTensor_uint8\n"); return nullptr; }
        size_t num_elements = std::accumulate(dims, dims + num_dims, size_t(1), std::multiplies<size_t>());
        size_t data_size = sizeof(uint8_t) * num_elements;
        uint8_t* tensor_data = static_cast<uint8_t*>(malloc(data_size));
        if (!tensor_data) { fprintf(stderr, "Erro: Falha ao alocar memória para TF_Tensor\n"); return nullptr; }
        memcpy(tensor_data, values, data_size);
        TF_Tensor* tensor = TF_NewTensor(
            TF_UINT8,
            dims,
            num_dims,
            tensor_data,
            data_size,
            [](void* data, size_t, void*) { free(data); },
            nullptr);
        if (!tensor) { free(tensor_data); fprintf(stderr, "Erro: Falha ao criar TF_Tensor uint8\n"); return nullptr; }
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) { fprintf(stderr, "Erro em CreateTFTensor_uint8: %s\n", e.what()); return nullptr; }
}

EXPORT void* CreateTFTensor_uint16(uint16_t* values, int64_t* dims, int num_dims) {
    try {
        if (!values || !dims || num_dims <= 0) { fprintf(stderr, "Erro: Parâmetros inválidos em CreateTFTensor_uint16\n"); return nullptr; }
        size_t num_elements = std::accumulate(dims, dims + num_dims, size_t(1), std::multiplies<size_t>());
        size_t data_size = sizeof(uint16_t) * num_elements;
        uint16_t* tensor_data = static_cast<uint16_t*>(malloc(data_size));
        if (!tensor_data) { fprintf(stderr, "Erro: Falha ao alocar memória para TF_Tensor\n"); return nullptr; }
        memcpy(tensor_data, values, data_size);
        TF_Tensor* tensor = TF_NewTensor(
            TF_UINT16,
            dims,
            num_dims,
            tensor_data,
            data_size,
            [](void* data, size_t, void*) { free(data); },
            nullptr);
        if (!tensor) { free(tensor_data); fprintf(stderr, "Erro: Falha ao criar TF_Tensor uint16\n"); return nullptr; }
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) { fprintf(stderr, "Erro em CreateTFTensor_uint16: %s\n", e.what()); return nullptr; }
}

EXPORT void* CreateTFTensor_bool(uint8_t* values, int64_t* dims, int num_dims) {
    try {
        if (!values || !dims || num_dims <= 0) { fprintf(stderr, "Erro: Parâmetros inválidos em CreateTFTensor_bool\n"); return nullptr; }
        size_t num_elements = std::accumulate(dims, dims + num_dims, size_t(1), std::multiplies<size_t>());
        size_t data_size = sizeof(uint8_t) * num_elements;
        uint8_t* tensor_data = static_cast<uint8_t*>(malloc(data_size));
        if (!tensor_data) { fprintf(stderr, "Erro: Falha ao alocar memória para TF_Tensor\n"); return nullptr; }
        memcpy(tensor_data, values, data_size);
        TF_Tensor* tensor = TF_NewTensor(
            TF_BOOL,
            dims,
            num_dims,
            tensor_data,
            data_size,
            [](void* data, size_t, void*) { free(data); },
            nullptr);
        if (!tensor) { free(tensor_data); fprintf(stderr, "Erro: Falha ao criar TF_Tensor bool\n"); return nullptr; }
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) { fprintf(stderr, "Erro em CreateTFTensor_bool: %s\n", e.what()); return nullptr; }
}

// Complex: store interleaved floats/doubles
EXPORT void* CreateTFTensor_complex64(float* values, int64_t* dims, int num_dims) {
    try {
        if (!values || !dims || num_dims <= 0) { fprintf(stderr, "Erro: Parâmetros inválidos em CreateTFTensor_complex64\n"); return nullptr; }
        size_t num_elements = std::accumulate(dims, dims + num_dims, size_t(1), std::multiplies<size_t>());
        // complex64 occupies 2 floats per element (real, imag)
        size_t data_size = sizeof(float) * 2 * num_elements;
        float* tensor_data = static_cast<float*>(malloc(data_size));
        if (!tensor_data) { fprintf(stderr, "Erro: Falha ao alocar memória para TF_Tensor\n"); return nullptr; }
        memcpy(tensor_data, values, data_size);
        TF_Tensor* tensor = TF_NewTensor(
            TF_COMPLEX64,
            dims,
            num_dims,
            tensor_data,
            data_size,
            [](void* data, size_t, void*) { free(data); },
            nullptr);
        if (!tensor) { free(tensor_data); fprintf(stderr, "Erro: Falha ao criar TF_Tensor complex64\n"); return nullptr; }
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) { fprintf(stderr, "Erro em CreateTFTensor_complex64: %s\n", e.what()); return nullptr; }
}

EXPORT void* CreateTFTensor_complex128(double* values, int64_t* dims, int num_dims) {
    try {

        
        if (!values || !dims || num_dims <= 0) { fprintf(stderr, "Erro: Parâmetros inválidos em CreateTFTensor_complex128\n"); return nullptr; }
        size_t num_elements = std::accumulate(dims, dims + num_dims, size_t(1), std::multiplies<size_t>());
        // complex128: 2 doubles per element
        size_t data_size = sizeof(double) * 2 * num_elements;
        double* tensor_data = static_cast<double*>(malloc(data_size));
        if (!tensor_data) { fprintf(stderr, "Erro: Falha ao alocar memória para TF_Tensor\n"); return nullptr; }
        memcpy(tensor_data, values, data_size);
        TF_Tensor* tensor = TF_NewTensor(
            TF_COMPLEX128,
            dims,
            num_dims,
            tensor_data,
            data_size,
            [](void* data, size_t, void*) { free(data); },
            nullptr);
        if (!tensor) { free(tensor_data); fprintf(stderr, "Erro: Falha ao criar TF_Tensor complex128\n"); return nullptr; }
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) { fprintf(stderr, "Erro em CreateTFTensor_complex128: %s\n", e.what()); return nullptr; }
}

// Strings: encode each string into TF_STRING buffer
EXPORT void* CreateTFTensor_string(const char** values, int64_t* dims, int num_dims) {
    try {
        if (!values || !dims || num_dims <= 0) { fprintf(stderr, "Erro: Parâmetros inválidos em CreateTFTensor_string\n"); return nullptr; }
        size_t num_elements = std::accumulate(dims, dims + num_dims, size_t(1), std::multiplies<size_t>());
        std::vector<std::string> strs;
        strs.reserve(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            const char* s = values[i];
            if (!s) s = "";
            strs.emplace_back(s);
        }

        // Fallback placeholder: store concatenated NUL-terminated strings in a TF_UINT8 tensor.
        size_t byte_size = 0;
        for (const auto& s : strs) byte_size += s.size() + 1;
        char* tensor_data = static_cast<char*>(malloc(byte_size));
        if (!tensor_data) { fprintf(stderr, "Erro: Falha ao alocar memória para tensor string\n"); return nullptr; }
        char* p = tensor_data;
        for (const auto& s : strs) {
            memcpy(p, s.data(), s.size());
            p += s.size();
            *p = '\0';
            p++;
        }
        // Create a UINT8 tensor containing concatenated strings as a placeholder representation
        TF_Tensor* tensor = TF_NewTensor(
            TF_UINT8,
            dims,
            num_dims,
            tensor_data,
            byte_size,
            [](void* data, size_t, void*) { free(data); },
            nullptr);
        if (!tensor) { free(tensor_data); fprintf(stderr, "Erro: Falha ao criar TF_Tensor string placeholder\n"); return nullptr; }
        return static_cast<void*>(tensor);
    } catch (const std::exception& e) { fprintf(stderr, "Erro em CreateTFTensor_string: %s\n", e.what()); return nullptr; }
}

// Decode TF_STRING tensor and return array of C strings. Caller must free via FreeStringArray.
EXPORT char** GetTensorData_string(void* tensor_ptr, int64_t* out_count) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) { fprintf(stderr, "Erro: Tensor inválido em GetTensorData_string\n"); return nullptr; }
        // We created placeholder string tensors as TF_UINT8 containing NUL-terminated concatenated strings
        if (TF_TensorType(tensor) != TF_UINT8) { fprintf(stderr, "Erro: expected UINT8 placeholder for string tensor in GetTensorData_string\n"); return nullptr; }
        int num_dims = TF_NumDims(tensor);
        size_t num_elements = 1;
        for (int i = 0; i < num_dims; ++i) num_elements *= TF_Dim(tensor, i);
        if (out_count) *out_count = static_cast<int64_t>(num_elements);

        unsigned char* data = static_cast<unsigned char*>(TF_TensorData(tensor));
        if (!data) { fprintf(stderr, "Erro: TF_TensorData retornou nullptr em GetTensorData_string\n"); return nullptr; }
        size_t total_size = TF_TensorByteSize(tensor);

        // Parse concatenated NUL-terminated strings
        char** out = static_cast<char**>(malloc(sizeof(char*) * num_elements));
        if (!out) { fprintf(stderr, "Erro: Falha ao alocar array de strings\n"); return nullptr; }

        size_t idx = 0;
        size_t pos = 0;
        for (size_t i = 0; i < num_elements && pos < total_size; ++i) {
            char* start = reinterpret_cast<char*>(data + pos);
            size_t len = strnlen(start, total_size - pos);
            char* s = static_cast<char*>(malloc(len + 1));
            if (!s) { s = nullptr; }
            else {
                memcpy(s, start, len);
                s[len] = '\0';
            }
            out[idx++] = s;
            pos += len + 1; // skip NUL
        }
        // If we allocated fewer strings than expected, fill rest with empty
        while (idx < num_elements) { out[idx++] = nullptr; }
        return out;
    } catch (const std::exception& e) { fprintf(stderr, "Erro em GetTensorData_string: %s\n", e.what()); return nullptr; }
}

EXPORT void FreeStringArray(char** arr, int64_t count) {
    if (!arr) return;
    for (int64_t i = 0; i < count; ++i) {
        if (arr[i]) free(arr[i]);
    }
    free(arr);
}

// ----- Typed Getters -----
EXPORT double* GetTensorData_double(void* tensor_ptr) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) { fprintf(stderr, "Erro: Tensor inválido em GetTensorData_double\n"); return nullptr; }
        if (TF_TensorType(tensor) != TF_DOUBLE) { fprintf(stderr, "Erro: Tensor não é TF_DOUBLE em GetTensorData_double\n"); return nullptr; }
        return static_cast<double*>(TF_TensorData(tensor));
    } catch (const std::exception& e) { fprintf(stderr, "Erro em GetTensorData_double: %s\n", e.what()); return nullptr; }
}

EXPORT int32_t* GetTensorData_int32(void* tensor_ptr) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) { fprintf(stderr, "Erro: Tensor inválido em GetTensorData_int32\n"); return nullptr; }
        if (TF_TensorType(tensor) != TF_INT32) { fprintf(stderr, "Erro: Tensor não é TF_INT32 em GetTensorData_int32\n"); return nullptr; }
        return static_cast<int32_t*>(TF_TensorData(tensor));
    } catch (const std::exception& e) { fprintf(stderr, "Erro em GetTensorData_int32: %s\n", e.what()); return nullptr; }
}

EXPORT int64_t* GetTensorData_int64(void* tensor_ptr) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) { fprintf(stderr, "Erro: Tensor inválido em GetTensorData_int64\n"); return nullptr; }
        if (TF_TensorType(tensor) != TF_INT64) { fprintf(stderr, "Erro: Tensor não é TF_INT64 em GetTensorData_int64\n"); return nullptr; }
        return static_cast<int64_t*>(TF_TensorData(tensor));
    } catch (const std::exception& e) { fprintf(stderr, "Erro em GetTensorData_int64: %s\n", e.what()); return nullptr; }
}

EXPORT int8_t* GetTensorData_int8(void* tensor_ptr) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) { fprintf(stderr, "Erro: Tensor inválido em GetTensorData_int8\n"); return nullptr; }
        if (TF_TensorType(tensor) != TF_INT8) { fprintf(stderr, "Erro: Tensor não é TF_INT8 em GetTensorData_int8\n"); return nullptr; }
        return static_cast<int8_t*>(TF_TensorData(tensor));
    } catch (const std::exception& e) { fprintf(stderr, "Erro em GetTensorData_int8: %s\n", e.what()); return nullptr; }
}

EXPORT int16_t* GetTensorData_int16(void* tensor_ptr) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) { fprintf(stderr, "Erro: Tensor inválido em GetTensorData_int16\n"); return nullptr; }
        if (TF_TensorType(tensor) != TF_INT16) { fprintf(stderr, "Erro: Tensor não é TF_INT16 em GetTensorData_int16\n"); return nullptr; }
        return static_cast<int16_t*>(TF_TensorData(tensor));
    } catch (const std::exception& e) { fprintf(stderr, "Erro em GetTensorData_int16: %s\n", e.what()); return nullptr; }
}

EXPORT uint8_t* GetTensorData_uint8(void* tensor_ptr) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) { fprintf(stderr, "Erro: Tensor inválido em GetTensorData_uint8\n"); return nullptr; }
        if (TF_TensorType(tensor) != TF_UINT8) { fprintf(stderr, "Erro: Tensor não é TF_UINT8 em GetTensorData_uint8\n"); return nullptr; }
        return static_cast<uint8_t*>(TF_TensorData(tensor));
    } catch (const std::exception& e) { fprintf(stderr, "Erro em GetTensorData_uint8: %s\n", e.what()); return nullptr; }
}

EXPORT uint16_t* GetTensorData_uint16(void* tensor_ptr) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) { fprintf(stderr, "Erro: Tensor inválido em GetTensorData_uint16\n"); return nullptr; }
        if (TF_TensorType(tensor) != TF_UINT16) { fprintf(stderr, "Erro: Tensor não é TF_UINT16 em GetTensorData_uint16\n"); return nullptr; }
        return static_cast<uint16_t*>(TF_TensorData(tensor));
    } catch (const std::exception& e) { fprintf(stderr, "Erro em GetTensorData_uint16: %s\n", e.what()); return nullptr; }
}

EXPORT uint8_t* GetTensorData_bool(void* tensor_ptr) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) { fprintf(stderr, "Erro: Tensor inválido em GetTensorData_bool\n"); return nullptr; }
        if (TF_TensorType(tensor) != TF_BOOL) { fprintf(stderr, "Erro: Tensor não é TF_BOOL em GetTensorData_bool\n"); return nullptr; }
        return static_cast<uint8_t*>(TF_TensorData(tensor));
    } catch (const std::exception& e) { fprintf(stderr, "Erro em GetTensorData_bool: %s\n", e.what()); return nullptr; }
}

EXPORT float* GetTensorData_complex64(void* tensor_ptr) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) { fprintf(stderr, "Erro: Tensor inválido em GetTensorData_complex64\n"); return nullptr; }
        if (TF_TensorType(tensor) != TF_COMPLEX64) { fprintf(stderr, "Erro: Tensor não é TF_COMPLEX64 em GetTensorData_complex64\n"); return nullptr; }
        return static_cast<float*>(TF_TensorData(tensor));
    } catch (const std::exception& e) { fprintf(stderr, "Erro em GetTensorData_complex64: %s\n", e.what()); return nullptr; }
}

EXPORT double* GetTensorData_complex128(void* tensor_ptr) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) { fprintf(stderr, "Erro: Tensor inválido em GetTensorData_complex128\n"); return nullptr; }
        if (TF_TensorType(tensor) != TF_COMPLEX128) { fprintf(stderr, "Erro: Tensor não é TF_COMPLEX128 em GetTensorData_complex128\n"); return nullptr; }
        return static_cast<double*>(TF_TensorData(tensor));
    } catch (const std::exception& e) { fprintf(stderr, "Erro em GetTensorData_complex128: %s\n", e.what()); return nullptr; }
}

// For string placeholder (we used TF_UINT8 to store concatenated strings). Return pointer to data and size via out param.
EXPORT unsigned char* GetTensorData_string_placeholder(void* tensor_ptr) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) { fprintf(stderr, "Erro: Tensor inválido em GetTensorData_string_placeholder\n"); return nullptr; }
        if (TF_TensorType(tensor) != TF_UINT8) { fprintf(stderr, "Aviso: expected UINT8 placeholder for string tensor\n"); return nullptr; }
        return static_cast<unsigned char*>(TF_TensorData(tensor));
    } catch (const std::exception& e) { fprintf(stderr, "Erro em GetTensorData_string_placeholder: %s\n", e.what()); return nullptr; }
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

// Return an allocated int64_t* containing the dims for a TF_Tensor.
// Caller must free with FreeInt64Array.
EXPORT int64_t* GetTFTensorDims(void* tensor_ptr, int* out_len) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) return nullptr;
        int nd = TF_NumDims(tensor);
        if (out_len) *out_len = nd;
        if (nd == 0) return nullptr;
        int64_t* arr = static_cast<int64_t*>(malloc(sizeof(int64_t) * nd));
        if (!arr) return nullptr;
        for (int i = 0; i < nd; ++i) arr[i] = TF_Dim(tensor, i);
        return arr;
    } catch (const std::exception& e) { fprintf(stderr, "Erro em GetTFTensorDims: %s\n", e.what()); return nullptr; }
}

EXPORT void FreeInt64Array(int64_t* arr) {
    if (arr) free(arr);
}

// Map TF_DataType to the Rust-side DType codes (see Rust enum ordering in tensors_flow.rs)
EXPORT int GetTFTensorDType(void* tensor_ptr) {
    try {
        TF_Tensor* tensor = static_cast<TF_Tensor*>(tensor_ptr);
        if (!tensor) return 12; // Unknown
        switch (TF_TensorType(tensor)) {
            case TF_FLOAT: return 0; // F32
            case TF_DOUBLE: return 1; // F64
            case TF_INT32: return 2; // I32
            case TF_INT64: return 3; // I64
            case TF_INT8: return 4; // I8
            case TF_INT16: return 5; // I16
            case TF_UINT8: return 6; // U8
            case TF_UINT16: return 7; // U16 (approx)
            case TF_BOOL: return 8; // Bool
            case TF_COMPLEX64: return 9; // Complex64
            case TF_COMPLEX128: return 10; // Complex128
            default: return 12; // Unknown / fallback
        }
    } catch (const std::exception& e) { fprintf(stderr, "Erro em GetTFTensorDType: %s\n", e.what()); return 12; }
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

// --- Native op implementations (CPU-side using TF_Tensor API) ---
// These implement a native CPU fallback by reading TF_Tensor inputs, computing
// the op on the CPU and returning a newly-allocated TF_Tensor. They use the
// TensorFlow C API TF_NewTensor/TF_TensorData for interoperability with Rust.


static size_t num_elements_from_dims(const int64_t* dims, int num_dims) {
    size_t n = 1;
    for (int i = 0; i < num_dims; ++i) n *= static_cast<size_t>(dims[i]);
    return n;
}

// Softmax (axis can be negative to indicate from the end)
EXPORT void* TF_Softmax_Native(void* tensor_ptr, int axis) {
    TF_Tensor* tin = static_cast<TF_Tensor*>(tensor_ptr);
    if (!tin) { fprintf(stderr, "TF_Softmax_Native: null input\n"); return nullptr; }
    if (TF_TensorType(tin) != TF_FLOAT) { fprintf(stderr, "TF_Softmax_Native: non-float input\n"); return nullptr; }

    int nd = TF_NumDims(tin);
    if (nd <= 0) { fprintf(stderr, "TF_Softmax_Native: input must have >=1 dims\n"); return nullptr; }
    int ax = axis < 0 ? nd + axis : axis;
    if (ax < 0 || ax >= nd) { fprintf(stderr, "TF_Softmax_Native: axis out of range\n"); return nullptr; }

    std::vector<int64_t> dims(nd);
    for (int i = 0; i < nd; ++i) dims[i] = TF_Dim(tin, i);
    size_t total = num_elements_from_dims(dims.data(), nd);
    float* in_data = static_cast<float*>(TF_TensorData(tin));
    if (!in_data) { fprintf(stderr, "TF_Softmax_Native: TF_TensorData null\n"); return nullptr; }

    // compute outer, axis_dim, inner like standard flattened iteration
    size_t axis_dim = static_cast<size_t>(dims[ax]);
    size_t inner = 1;
    for (int i = ax + 1; i < nd; ++i) inner *= static_cast<size_t>(dims[i]);
    size_t outer = total / (axis_dim * inner);

    size_t out_bytes = total * sizeof(float);
    float* out_data = static_cast<float*>(malloc(out_bytes));
    if (!out_data) { fprintf(stderr, "TF_Softmax_Native: malloc failed\n"); return nullptr; }

    // For each slice compute numerically stable softmax
    for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < inner; ++i) {
            // find max
            float m = -INFINITY;
            for (size_t a = 0; a < axis_dim; ++a) {
                size_t idx = (o * axis_dim + a) * inner + i;
                m = std::max(m, in_data[idx]);
            }
            // sum exps
            double s = 0.0;
            for (size_t a = 0; a < axis_dim; ++a) {
                size_t idx = (o * axis_dim + a) * inner + i;
                double e = std::exp(static_cast<double>(in_data[idx] - m));
                s += e;
                out_data[idx] = static_cast<float>(e); // temporarily store exp
            }
            double logs = std::log(s);
            for (size_t a = 0; a < axis_dim; ++a) {
                size_t idx = (o * axis_dim + a) * inner + i;
                out_data[idx] = static_cast<float>(static_cast<double>(out_data[idx]) / s);
            }
        }
    }

    TF_Tensor* tout = TF_NewTensor(TF_FLOAT, dims.data(), nd, out_data, out_bytes,
                                    [](void* data, size_t, void*) { free(data); }, nullptr);
    if (!tout) { free(out_data); fprintf(stderr, "TF_Softmax_Native: TF_NewTensor failed\n"); return nullptr; }
    return static_cast<void*>(tout);
}

// BiasAdd: add a 1-D bias along axis
EXPORT void* TF_BiasAdd_Native(void* tensor_ptr, void* bias_ptr, int axis) {
    TF_Tensor* tin = static_cast<TF_Tensor*>(tensor_ptr);
    TF_Tensor* tbias = static_cast<TF_Tensor*>(bias_ptr);
    if (!tin || !tbias) { fprintf(stderr, "TF_BiasAdd_Native: null input\n"); return nullptr; }
    if (TF_TensorType(tin) != TF_FLOAT || TF_TensorType(tbias) != TF_FLOAT) { fprintf(stderr, "TF_BiasAdd_Native: non-float input\n"); return nullptr; }
    int nd = TF_NumDims(tin);
    if (nd <= 0) { fprintf(stderr, "TF_BiasAdd_Native: input must have >=1 dims\n"); return nullptr; }
    int ax = axis < 0 ? nd + axis : axis;
    if (ax < 0 || ax >= nd) { fprintf(stderr, "TF_BiasAdd_Native: axis out of range\n"); return nullptr; }
    int64_t axis_dim = TF_Dim(tin, ax);
    if (TF_NumDims(tbias) != 1 || TF_Dim(tbias, 0) != axis_dim) { fprintf(stderr, "TF_BiasAdd_Native: bias shape mismatch\n"); return nullptr; }

    std::vector<int64_t> dims(nd);
    size_t total = 1;
    for (int i = 0; i < nd; ++i) { dims[i] = TF_Dim(tin, i); total *= static_cast<size_t>(dims[i]); }
    float* in_data = static_cast<float*>(TF_TensorData(tin));
    float* bias = static_cast<float*>(TF_TensorData(tbias));
    if (!in_data || !bias) { fprintf(stderr, "TF_BiasAdd_Native: TF_TensorData null\n"); return nullptr; }

    size_t inner = 1;
    for (int i = ax + 1; i < nd; ++i) inner *= static_cast<size_t>(dims[i]);
    size_t axis_sz = static_cast<size_t>(axis_dim);
    size_t outer = total / (axis_sz * inner);

    size_t out_bytes = total * sizeof(float);
    float* out_data = static_cast<float*>(malloc(out_bytes));
    if (!out_data) { fprintf(stderr, "TF_BiasAdd_Native: malloc failed\n"); return nullptr; }

    for (size_t o = 0; o < outer; ++o) {
        for (size_t a = 0; a < axis_sz; ++a) {
            for (size_t i = 0; i < inner; ++i) {
                size_t idx = (o * axis_sz + a) * inner + i;
                out_data[idx] = in_data[idx] + bias[a];
            }
        }
    }

    TF_Tensor* tout = TF_NewTensor(TF_FLOAT, dims.data(), nd, out_data, out_bytes,
                                    [](void* data, size_t, void*) { free(data); }, nullptr);
    if (!tout) { free(out_data); fprintf(stderr, "TF_BiasAdd_Native: TF_NewTensor failed\n"); return nullptr; }
    return static_cast<void*>(tout);
}

// BatchNorm: simplified per-channel batch normalization. mean/var/scale/offset may be null.
EXPORT void* TF_BatchNorm_Native(void* tensor_ptr, void* mean_ptr, void* variance_ptr, void* scale_ptr, void* offset_ptr, float epsilon, int axis) {
    TF_Tensor* tin = static_cast<TF_Tensor*>(tensor_ptr);
    if (!tin) { fprintf(stderr, "TF_BatchNorm_Native: null input\n"); return nullptr; }
    if (TF_TensorType(tin) != TF_FLOAT) { fprintf(stderr, "TF_BatchNorm_Native: non-float input\n"); return nullptr; }
    int nd = TF_NumDims(tin);
    if (nd <= 0) { fprintf(stderr, "TF_BatchNorm_Native: input must have >=1 dims\n"); return nullptr; }
    int ax = axis < 0 ? nd + axis : axis;
    if (ax < 0 || ax >= nd) { fprintf(stderr, "TF_BatchNorm_Native: axis out of range\n"); return nullptr; }

    std::vector<int64_t> dims(nd);
    size_t total = 1;
    for (int i = 0; i < nd; ++i) { dims[i] = TF_Dim(tin, i); total *= static_cast<size_t>(dims[i]); }
    float* in_data = static_cast<float*>(TF_TensorData(tin));
    if (!in_data) { fprintf(stderr, "TF_BatchNorm_Native: TF_TensorData null\n"); return nullptr; }

    size_t axis_dim = static_cast<size_t>(dims[ax]);
    size_t inner = 1;
    for (int i = ax + 1; i < nd; ++i) inner *= static_cast<size_t>(dims[i]);
    size_t outer = total / (axis_dim * inner);

    // read provided tensors or compute
    std::vector<float> mean(axis_dim, 0.0f);
    std::vector<float> var(axis_dim, 0.0f);

    if (mean_ptr && variance_ptr) {
        TF_Tensor* tmean = static_cast<TF_Tensor*>(mean_ptr);
        TF_Tensor* tvar = static_cast<TF_Tensor*>(variance_ptr);
        if (!tmean || !tvar) { fprintf(stderr, "TF_BatchNorm_Native: mean/var null\n"); return nullptr; }
        float* mdata = static_cast<float*>(TF_TensorData(tmean));
        float* vdata = static_cast<float*>(TF_TensorData(tvar));
        for (size_t i = 0; i < axis_dim; ++i) { mean[i] = mdata[i]; var[i] = vdata[i]; }
    } else {
        // compute mean
        for (size_t o = 0; o < outer; ++o) {
            for (size_t a = 0; a < axis_dim; ++a) {
                for (size_t i = 0; i < inner; ++i) {
                    size_t idx = (o * axis_dim + a) * inner + i;
                    mean[a] += in_data[idx];
                }
            }
        }
        float count = static_cast<float>(outer * inner);
        for (size_t a = 0; a < axis_dim; ++a) mean[a] /= count;
        // compute var
        for (size_t o = 0; o < outer; ++o) {
            for (size_t a = 0; a < axis_dim; ++a) {
                for (size_t i = 0; i < inner; ++i) {
                    size_t idx = (o * axis_dim + a) * inner + i;
                    float d = in_data[idx] - mean[a];
                    var[a] += d * d;
                }
            }
        }
        for (size_t a = 0; a < axis_dim; ++a) var[a] /= count;
    }

    std::vector<float> scale(axis_dim, 1.0f);
    std::vector<float> offset(axis_dim, 0.0f);
    if (scale_ptr) {
        TF_Tensor* tscale = static_cast<TF_Tensor*>(scale_ptr);
        float* sdata = static_cast<float*>(TF_TensorData(tscale));
        for (size_t i = 0; i < axis_dim; ++i) scale[i] = sdata[i];
    }
    if (offset_ptr) {
        TF_Tensor* toff = static_cast<TF_Tensor*>(offset_ptr);
        float* odata = static_cast<float*>(TF_TensorData(toff));
        for (size_t i = 0; i < axis_dim; ++i) offset[i] = odata[i];
    }

    size_t out_bytes = total * sizeof(float);
    float* out_data = static_cast<float*>(malloc(out_bytes));
    if (!out_data) { fprintf(stderr, "TF_BatchNorm_Native: malloc failed\n"); return nullptr; }

    for (size_t o = 0; o < outer; ++o) {
        for (size_t a = 0; a < axis_dim; ++a) {
            float mean_v = mean[a];
            float var_v = var[a];
            float sc = scale[a];
            float off = offset[a];
            float denom = std::sqrt(var_v + epsilon);
            for (size_t i = 0; i < inner; ++i) {
                size_t idx = (o * axis_dim + a) * inner + i;
                out_data[idx] = ((in_data[idx] - mean_v) / denom) * sc + off;
            }
        }
    }

    TF_Tensor* tout = TF_NewTensor(TF_FLOAT, dims.data(), nd, out_data, out_bytes,
                                    [](void* data, size_t, void*) { free(data); }, nullptr);
    if (!tout) { free(out_data); fprintf(stderr, "TF_BatchNorm_Native: TF_NewTensor failed\n"); return nullptr; }
    return static_cast<void*>(tout);
}

// SoftmaxCrossEntropy (dense one-hot labels)
// LogSoftmax implemented inline here so it's available for callers
EXPORT void* TF_LogSoftmax_Native(void* tensor_ptr, int axis) {
    TF_Tensor* tin = static_cast<TF_Tensor*>(tensor_ptr);
    if (!tin) { fprintf(stderr, "TF_LogSoftmax_Native: null input\n"); return nullptr; }
    if (TF_TensorType(tin) != TF_FLOAT) { fprintf(stderr, "TF_LogSoftmax_Native: non-float input\n"); return nullptr; }

    int nd = TF_NumDims(tin);
    int ax = axis < 0 ? nd + axis : axis;
    if (ax < 0 || ax >= nd) { fprintf(stderr, "TF_LogSoftmax_Native: axis out of range\n"); return nullptr; }

    std::vector<int64_t> dims(nd);
    for (int i = 0; i < nd; ++i) dims[i] = TF_Dim(tin, i);
    size_t total = num_elements_from_dims(dims.data(), nd);
    float* in_data = static_cast<float*>(TF_TensorData(tin));
    if (!in_data) { fprintf(stderr, "TF_LogSoftmax_Native: TF_TensorData null\n"); return nullptr; }

    size_t axis_dim = static_cast<size_t>(dims[ax]);
    size_t inner = 1;
    for (int i = ax + 1; i < nd; ++i) inner *= static_cast<size_t>(dims[i]);
    size_t outer = total / (axis_dim * inner);

    size_t out_bytes = total * sizeof(float);
    float* out_data = static_cast<float*>(malloc(out_bytes));
    if (!out_data) { fprintf(stderr, "TF_LogSoftmax_Native: malloc failed\n"); return nullptr; }

    for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < inner; ++i) {
            float m = -INFINITY;
            for (size_t a = 0; a < axis_dim; ++a) {
                size_t idx = (o * axis_dim + a) * inner + i;
                m = std::max(m, in_data[idx]);
            }
            double s = 0.0;
            for (size_t a = 0; a < axis_dim; ++a) {
                size_t idx = (o * axis_dim + a) * inner + i;
                s += std::exp(static_cast<double>(in_data[idx] - m));
            }
            double logsum = std::log(s);
            for (size_t a = 0; a < axis_dim; ++a) {
                size_t idx = (o * axis_dim + a) * inner + i;
                out_data[idx] = static_cast<float>(static_cast<double>(in_data[idx] - m) - logsum);
            }
        }
    }

    TF_Tensor* tout = TF_NewTensor(TF_FLOAT, dims.data(), nd, out_data, out_bytes,
                                    [](void* data, size_t, void*) { free(data); }, nullptr);
    if (!tout) { free(out_data); fprintf(stderr, "TF_LogSoftmax_Native: TF_NewTensor failed\n"); return nullptr; }
    return static_cast<void*>(tout);
}

EXPORT void* TF_SoftmaxCrossEntropy_Native(void* logits_ptr, void* labels_ptr, int axis) {
    TF_Tensor* tlog = static_cast<TF_Tensor*>(logits_ptr);
    TF_Tensor* tlabels = static_cast<TF_Tensor*>(labels_ptr);
    if (!tlog || !tlabels) { fprintf(stderr, "TF_SoftmaxCrossEntropy_Native: null input\n"); return nullptr; }
    if (TF_TensorType(tlog) != TF_FLOAT || TF_TensorType(tlabels) != TF_FLOAT) { fprintf(stderr, "TF_SoftmaxCrossEntropy_Native: non-float input\n"); return nullptr; }
    int nd = TF_NumDims(tlog);
    int ax = axis < 0 ? nd + axis : axis;
    if (ax < 0 || ax >= nd) { fprintf(stderr, "TF_SoftmaxCrossEntropy_Native: axis out of range\n"); return nullptr; }

    // reuse LogSoftmax native implementation to get log-probs
    TF_Tensor* tlogp = static_cast<TF_Tensor*>(TF_LogSoftmax_Native(tlog, ax));
    if (!tlogp) { fprintf(stderr, "TF_SoftmaxCrossEntropy_Native: logsoftmax failed\n"); return nullptr; }

    int64_t axis_dim = TF_Dim(tlog, ax);
    std::vector<int64_t> out_dims;
    for (int i = 0; i < nd; ++i) if (i != ax) out_dims.push_back(TF_Dim(tlog, i));
    if (out_dims.empty()) out_dims.push_back(1);

    float* lp = static_cast<float*>(TF_TensorData(tlogp));
    float* lab = static_cast<float*>(TF_TensorData(tlabels));
    // compute total number of elements
    size_t total = 1;
    for (int i = 0; i < nd; ++i) total *= static_cast<size_t>(TF_Dim(tlog, i));
    // compute outer, inner
    size_t inner = 1;
    for (int i = ax + 1; i < nd; ++i) inner *= static_cast<size_t>(TF_Dim(tlog, i));
    size_t axis_sz = static_cast<size_t>(axis_dim);
    size_t outer = 1;
    for (int i = 0; i < ax; ++i) outer *= static_cast<size_t>(TF_Dim(tlog, i));

    size_t out_len = outer * (inner==0?1:inner);
    float* out_data = static_cast<float*>(malloc(out_len * sizeof(float)));
    if (!out_data) { fprintf(stderr, "TF_SoftmaxCrossEntropy_Native: malloc failed\n"); TF_DeleteTensor(tlogp); return nullptr; }

    for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < (inner==0?1:inner); ++i) {
            double loss = 0.0;
            for (size_t a = 0; a < axis_sz; ++a) {
                size_t idx = (o * axis_sz + a) * inner + i;
                loss -= static_cast<double>(lab[idx]) * static_cast<double>(lp[idx]);
            }
            out_data[o * (inner==0?1:inner) + i] = static_cast<float>(loss);
        }
    }

    TF_Tensor* tout = TF_NewTensor(TF_FLOAT, out_dims.data(), (int)out_dims.size(), out_data, out_len * sizeof(float),
                                    [](void* data, size_t, void*) { free(data); }, nullptr);
    TF_DeleteTensor(tlogp);
    if (!tout) { free(out_data); fprintf(stderr, "TF_SoftmaxCrossEntropy_Native: TF_NewTensor failed\n"); return nullptr; }
    return static_cast<void*>(tout);
}

// SoftmaxCrossEntropy sparse labels (labels_idx is int64 vector)
EXPORT void* TF_SoftmaxCrossEntropy_Sparse_Native(void* logits_ptr, void* labels_idx_ptr, int axis) {
    TF_Tensor* tlog = static_cast<TF_Tensor*>(logits_ptr);
    TF_Tensor* tidx = static_cast<TF_Tensor*>(labels_idx_ptr);
    if (!tlog || !tidx) { fprintf(stderr, "TF_SoftmaxCrossEntropy_Sparse_Native: null input\n"); return nullptr; }
    if (TF_TensorType(tlog) != TF_FLOAT || TF_TensorType(tidx) != TF_INT64) { fprintf(stderr, "TF_SoftmaxCrossEntropy_Sparse_Native: bad dtypes\n"); return nullptr; }
    int nd = TF_NumDims(tlog);
    int ax = axis < 0 ? nd + axis : axis;
    if (ax < 0 || ax >= nd) { fprintf(stderr, "TF_SoftmaxCrossEntropy_Sparse_Native: axis out of range\n"); return nullptr; }

    TF_Tensor* tlogp = static_cast<TF_Tensor*>(TF_LogSoftmax_Native(tlog, ax));
    if (!tlogp) { fprintf(stderr, "TF_SoftmaxCrossEntropy_Sparse_Native: logsoftmax failed\n"); return nullptr; }

    int64_t axis_dim = TF_Dim(tlog, ax);
    int64_t inner = 1;
    for (int i = ax + 1; i < nd; ++i) inner *= TF_Dim(tlog, i);
    int64_t outer = 1;
    for (int i = 0; i < ax; ++i) outer *= TF_Dim(tlog, i);

    int64_t out_len = outer * (inner==0?1:inner);
    int64_t* idxs = static_cast<int64_t*>(TF_TensorData(tidx));
    float* lp = static_cast<float*>(TF_TensorData(tlogp));

    float* out_data = static_cast<float*>(malloc(out_len * sizeof(float)));
    if (!out_data) { fprintf(stderr, "TF_SoftmaxCrossEntropy_Sparse_Native: malloc failed\n"); TF_DeleteTensor(tlogp); return nullptr; }

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < (inner==0?1:inner); ++i) {
            int64_t out_idx = o * (inner==0?1:inner) + i;
            int64_t lbl = idxs[out_idx];
            if (lbl < 0 || lbl >= axis_dim) { out_data[out_idx] = NAN; continue; }
            int64_t flat = o * (axis_dim * inner) + lbl * inner + i;
            out_data[out_idx] = - lp[flat];
        }
    }

    std::vector<int64_t> out_dims;
    for (int i = 0; i < nd; ++i) if (i != ax) out_dims.push_back(TF_Dim(tlog, i));
    if (out_dims.empty()) out_dims.push_back(1);

    TF_Tensor* tout = TF_NewTensor(TF_FLOAT, out_dims.data(), (int)out_dims.size(), out_data, out_len * sizeof(float),
                                    [](void* data, size_t, void*) { free(data); }, nullptr);
    TF_DeleteTensor(tlogp);
    if (!tout) { free(out_data); fprintf(stderr, "TF_SoftmaxCrossEntropy_Sparse_Native: TF_NewTensor failed\n"); return nullptr; }
    return static_cast<void*>(tout);
}

// Dropout: random mask with keep_prob, scales kept elements by 1/keep_prob
EXPORT void* TF_Dropout_Native(void* tensor_ptr, float keep_prob, uint64_t seed) {
    TF_Tensor* tin = static_cast<TF_Tensor*>(tensor_ptr);
    if (!tin) { fprintf(stderr, "TF_Dropout_Native: null input\n"); return nullptr; }
    if (TF_TensorType(tin) != TF_FLOAT) { fprintf(stderr, "TF_Dropout_Native: non-float input\n"); return nullptr; }
    int nd = TF_NumDims(tin);
    std::vector<int64_t> dims(nd);
    size_t total = 1;
    for (int i = 0; i < nd; ++i) { dims[i] = TF_Dim(tin, i); total *= static_cast<size_t>(dims[i]); }
    float* in_data = static_cast<float*>(TF_TensorData(tin));
    if (!in_data) { fprintf(stderr, "TF_Dropout_Native: TF_TensorData null\n"); return nullptr; }

    size_t out_bytes = total * sizeof(float);
    float* out_data = static_cast<float*>(malloc(out_bytes));
    if (!out_data) { fprintf(stderr, "TF_Dropout_Native: malloc failed\n"); return nullptr; }

    std::mt19937_64 rng((seed==0) ? std::random_device{}() : seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < total; ++i) {
        double r = dist(rng);
        if (r < static_cast<double>(keep_prob)) out_data[i] = in_data[i] / keep_prob; else out_data[i] = 0.0f;
    }

    TF_Tensor* tout = TF_NewTensor(TF_FLOAT, dims.data(), nd, out_data, out_bytes,
                                    [](void* data, size_t, void*) { free(data); }, nullptr);
    if (!tout) { free(out_data); fprintf(stderr, "TF_Dropout_Native: TF_NewTensor failed\n"); return nullptr; }
    return static_cast<void*>(tout);
}

// Conv2D naive: supports NHWC and NCHW via layout string. padding: "SAME" or "VALID".
EXPORT void* TF_Conv2D_Native(void* input, void* filter, int stride_h, int stride_w,
                             int dilation_h, int dilation_w, const char* padding, const char* layout) {
    TF_Tensor* tin = static_cast<TF_Tensor*>(input);
    TF_Tensor* tfilter = static_cast<TF_Tensor*>(filter);
    if (!tin || !tfilter) { fprintf(stderr, "TF_Conv2D_Native: null input/filter\n"); return nullptr; }
    if (TF_TensorType(tin) != TF_FLOAT || TF_TensorType(tfilter) != TF_FLOAT) { fprintf(stderr, "TF_Conv2D_Native: non-float tensors\n"); return nullptr; }

    int in_nd = TF_NumDims(tin);
    int f_nd = TF_NumDims(tfilter);
    if (in_nd != 4 || f_nd != 4) { fprintf(stderr, "TF_Conv2D_Native: expects 4D input and filter\n"); return nullptr; }

    bool is_nchw = false;
    if (layout && std::strcmp(layout, "NCHW") == 0) is_nchw = true;

    int64_t batch = TF_Dim(tin, 0);
    int64_t in_h = is_nchw ? TF_Dim(tin, 2) : TF_Dim(tin, 1);
    int64_t in_w = is_nchw ? TF_Dim(tin, 3) : TF_Dim(tin, 2);
    int64_t in_c = is_nchw ? TF_Dim(tin, 1) : TF_Dim(tin, 3);

    int64_t k_h = TF_Dim(tfilter, 0);
    int64_t k_w = TF_Dim(tfilter, 1);
    int64_t f_in_c = TF_Dim(tfilter, 2);
    int64_t out_c = TF_Dim(tfilter, 3);
    if (f_in_c != in_c) { fprintf(stderr, "TF_Conv2D_Native: channel mismatch\n"); return nullptr; }

    int s_h = std::max(1, stride_h);
    int s_w = std::max(1, stride_w);
    int d_h = std::max(1, dilation_h);
    int d_w = std::max(1, dilation_w);

    std::string pad = padding ? std::string(padding) : "VALID";
    int64_t out_h=0, out_w=0, pad_top=0, pad_left=0;
    if (pad == "SAME") {
        out_h = (in_h + s_h - 1) / s_h;
        out_w = (in_w + s_w - 1) / s_w;
        int64_t pad_h = std::max<int64_t>(0, (out_h - 1) * s_h + (k_h - 1) * d_h + 1 - in_h);
        int64_t pad_w = std::max<int64_t>(0, (out_w - 1) * s_w + (k_w - 1) * d_w + 1 - in_w);
        pad_top = pad_h / 2;
        pad_left = pad_w / 2;
    } else { // VALID
        out_h = (in_h - (k_h - 1) * d_h - 1) / s_h + 1;
        out_w = (in_w - (k_w - 1) * d_w - 1) / s_w + 1;
    }

    // prepare dims for output
    std::vector<int64_t> out_dims(4);
    if (is_nchw) {
        out_dims[0] = batch;
        out_dims[1] = out_c;
        out_dims[2] = out_h;
        out_dims[3] = out_w;
    } else {
        out_dims[0] = batch;
        out_dims[1] = out_h;
        out_dims[2] = out_w;
        out_dims[3] = out_c;
    }

    size_t out_count = static_cast<size_t>(batch) * out_h * out_w * static_cast<size_t>(out_c);
    size_t out_bytes = out_count * sizeof(float);
    float* out_data = static_cast<float*>(calloc(1, out_bytes));
    if (!out_data) { fprintf(stderr, "TF_Conv2D_Native: calloc failed\n"); return nullptr; }

    float* in_data = static_cast<float*>(TF_TensorData(tin));
    float* filt_data = static_cast<float*>(TF_TensorData(tfilter));

    // iterate
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oh = 0; oh < out_h; ++oh) {
            for (int64_t ow = 0; ow < out_w; ++ow) {
                for (int64_t oc = 0; oc < out_c; ++oc) {
                    double acc = 0.0;
                    for (int64_t kh = 0; kh < k_h; ++kh) {
                        for (int64_t kw = 0; kw < k_w; ++kw) {
                            int64_t ih = oh * s_h + kh * d_h - pad_top;
                            int64_t iw = ow * s_w + kw * d_w - pad_left;
                            if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) continue;
                            for (int64_t ic = 0; ic < in_c; ++ic) {
                                size_t in_idx;
                                if (is_nchw) {
                                    in_idx = ((size_t)n * in_c + ic) * (in_h * in_w) + (ih * in_w + iw);
                                } else {
                                    in_idx = ((size_t)n * in_h + ih) * (in_w * in_c) + (iw * in_c) + ic;
                                }
                                size_t filt_idx = ((size_t)kh * k_w + kw) * (in_c * out_c) + (ic * out_c) + oc;
                                acc += static_cast<double>(in_data[in_idx]) * static_cast<double>(filt_data[filt_idx]);
                            }
                        }
                    }
                    // set output
                    size_t out_idx;
                    if (is_nchw) {
                        out_idx = ((size_t)n * out_c + oc) * (out_h * out_w) + (oh * out_w + ow);
                    } else {
                        out_idx = ((size_t)n * out_h + oh) * (out_w * out_c) + (ow * out_c) + oc;
                    }
                    out_data[out_idx] = static_cast<float>(acc);
                }
            }
        }
    }

    TF_Tensor* tout = TF_NewTensor(TF_FLOAT, out_dims.data(), static_cast<int>(out_dims.size()), out_data, out_bytes,
                                    [](void* data, size_t, void*) { free(data); }, nullptr);
    if (!tout) { free(out_data); fprintf(stderr, "TF_Conv2D_Native: TF_NewTensor failed\n"); return nullptr; }
    return static_cast<void*>(tout);
}

// Conv3D implemented by building a tiny TF graph with Const(input), Const(filter)
// and a Conv3D op, then running it via TF_SessionRun. This delegates the
// computation to TensorFlow's kernels (optimized/parallelized) when available.
EXPORT void* TF_Conv3D_Native(void* input, void* filter, int stride_d, int stride_h, int stride_w,
                             int dilation_d, int dilation_h, int dilation_w, const char* padding, const char* layout) {
    // Implement a simple cache for Conv3D graphs+sessions keyed by the operation signature
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::pair<TF_Graph*, TF_Session*>> cache;

    TF_Tensor* tin = static_cast<TF_Tensor*>(input);
    TF_Tensor* tfilter = static_cast<TF_Tensor*>(filter);
    if (!tin || !tfilter) { fprintf(stderr, "TF_Conv3D_Native: null input/filter\n"); return nullptr; }
    if (TF_TensorType(tin) != TF_FLOAT || TF_TensorType(tfilter) != TF_FLOAT) { fprintf(stderr, "TF_Conv3D_Native: non-float tensors\n"); return nullptr; }

    // build a key based on filter shape, strides, dilations, padding and layout
    int f_nd = TF_NumDims(tfilter);
    std::ostringstream keyss;
    keyss << "conv3d:";
    for (int i = 0; i < f_nd; ++i) keyss << TF_Dim(tfilter, i) << ",";
    keyss << "s:" << stride_d << "," << stride_h << "," << stride_w;
    keyss << "d:" << dilation_d << "," << dilation_h << "," << dilation_w;
    keyss << "p:" << (padding?padding:"VAL") << ",l:" << (layout?layout:"NDHWC");
    std::string key = keyss.str();

    TF_Graph* graph = nullptr;
    TF_Session* sess = nullptr;

    {
        std::lock_guard<std::mutex> lg(cache_mutex);
        auto it = cache.find(key);
        if (it != cache.end()) {
            graph = it->second.first;
            sess = it->second.second;
        }
    }

    TF_Status* status = TF_NewStatus();

    if (!graph) {
        // build graph with placeholders (so we can feed different inputs later)
        graph = TF_NewGraph();
        if (!graph) { fprintf(stderr, "TF_Conv3D_Native: TF_NewGraph failed\n"); TF_DeleteStatus(status); return nullptr; }

        TF_OperationDescription* desc_in = TF_NewOperation(graph, "Placeholder", "conv3d_input");
        TF_SetAttrType(desc_in, "dtype", TF_FLOAT);
        TF_Operation* op_in = TF_FinishOperation(desc_in, status);
        if (TF_GetCode(status) != TF_OK) { fprintf(stderr, "TF_Conv3D_Native: placeholder input failed: %s\n", TF_Message(status)); TF_DeleteGraph(graph); TF_DeleteStatus(status); return nullptr; }

        TF_OperationDescription* desc_f = TF_NewOperation(graph, "Placeholder", "conv3d_filter");
        TF_SetAttrType(desc_f, "dtype", TF_FLOAT);
        TF_Operation* op_f = TF_FinishOperation(desc_f, status);
        if (TF_GetCode(status) != TF_OK) { fprintf(stderr, "TF_Conv3D_Native: placeholder filter failed: %s\n", TF_Message(status)); TF_DeleteGraph(graph); TF_DeleteStatus(status); return nullptr; }

        // Build Conv3D op that consumes the placeholders
        TF_OperationDescription* desc_conv = TF_NewOperation(graph, "Conv3D", "conv3d_native");
        TF_Output in_out; in_out.oper = op_in; in_out.index = 0;
        TF_Output f_out; f_out.oper = op_f; f_out.index = 0;
        TF_AddInput(desc_conv, in_out);
        TF_AddInput(desc_conv, f_out);

        bool is_ncdhw = (layout && std::strcmp(layout, "NCDHW") == 0);
        int64_t strides_attr[5];
        if (!is_ncdhw) { strides_attr[0]=1; strides_attr[1]=stride_d; strides_attr[2]=stride_h; strides_attr[3]=stride_w; strides_attr[4]=1; }
        else { strides_attr[0]=1; strides_attr[1]=1; strides_attr[2]=stride_d; strides_attr[3]=stride_h; strides_attr[4]=stride_w; }
        TF_SetAttrIntList(desc_conv, "strides", strides_attr, 5);

        int64_t dilations_attr[5];
        if (!is_ncdhw) { dilations_attr[0]=1; dilations_attr[1]=dilation_d; dilations_attr[2]=dilation_h; dilations_attr[3]=dilation_w; dilations_attr[4]=1; }
        else { dilations_attr[0]=1; dilations_attr[1]=1; dilations_attr[2]=dilation_d; dilations_attr[3]=dilation_h; dilations_attr[4]=dilation_w; }
        TF_SetAttrIntList(desc_conv, "dilations", dilations_attr, 5);

        const char* padstr = padding ? padding : "VALID";
        TF_SetAttrString(desc_conv, "padding", padstr, static_cast<size_t>(std::strlen(padstr)));
        const char* df = is_ncdhw ? "NCDHW" : "NDHWC";
        TF_SetAttrString(desc_conv, "data_format", df, static_cast<size_t>(std::strlen(df)));

        TF_Operation* op_conv = TF_FinishOperation(desc_conv, status);
        if (TF_GetCode(status) != TF_OK) { fprintf(stderr, "TF_Conv3D_Native: FinishOperation conv failed: %s\n", TF_Message(status)); TF_DeleteGraph(graph); TF_DeleteStatus(status); return nullptr; }

        // create session for the graph
        TF_SessionOptions* sess_opts = TF_NewSessionOptions();
        sess = TF_NewSession(graph, sess_opts, status);
        TF_DeleteSessionOptions(sess_opts);
        if (TF_GetCode(status) != TF_OK || !sess) { fprintf(stderr, "TF_Conv3D_Native: TF_NewSession failed: %s\n", TF_Message(status)); TF_DeleteGraph(graph); TF_DeleteStatus(status); return nullptr; }

        // store in cache
        {
            std::lock_guard<std::mutex> lg(cache_mutex);
            cache.emplace(key, std::make_pair(graph, sess));
        }
    }

    // Run the cached session. For safety, guard session run with the cache mutex to serialize use of the session
    TF_Status* run_status = TF_NewStatus();
    TF_Output input_ops[2];
    TF_Operation* op_in = TF_GraphOperationByName(cache[key].first, "conv3d_input");
    TF_Operation* op_filt = TF_GraphOperationByName(cache[key].first, "conv3d_filter");
    if (!op_in || !op_filt) { fprintf(stderr, "TF_Conv3D_Native: cached ops missing\n"); TF_DeleteStatus(run_status); return nullptr; }
    input_ops[0].oper = op_in; input_ops[0].index = 0;
    input_ops[1].oper = op_filt; input_ops[1].index = 0;

    TF_Tensor* in_t = static_cast<TF_Tensor*>(input);
    TF_Tensor* filt_t = static_cast<TF_Tensor*>(filter);
    TF_Tensor* out_tensors[1] = { nullptr };

    {
        std::lock_guard<std::mutex> lg(cache_mutex);
    TF_Operation* op_conv = TF_GraphOperationByName(cache[key].first, "conv3d_native");
    TF_Output outputs_arr[1]; outputs_arr[0].oper = op_conv; outputs_arr[0].index = 0;
    TF_SessionRun(cache[key].second, nullptr,
              input_ops, (TF_Tensor**)&in_t, 2,
              outputs_arr, out_tensors, 1,
              nullptr, 0, nullptr, run_status);
    }

    if (TF_GetCode(run_status) != TF_OK) {
        fprintf(stderr, "TF_Conv3D_Native: TF_SessionRun failed: %s\n", TF_Message(run_status));
        if (out_tensors[0]) TF_DeleteTensor(out_tensors[0]);
        TF_DeleteStatus(run_status);
        return nullptr;
    }
    TF_DeleteStatus(run_status);

    // Return the output TF_Tensor (caller owns it)
    return static_cast<void*>(out_tensors[0]);
}

// Pooling (NHWC/NCHW supported). k_h/k_w kernel sizes, stride, padding SAME/VALID.
EXPORT void* TF_MaxPool_Native(void* input, int k_h, int k_w, int stride_h, int stride_w, const char* padding, const char* layout) {
    TF_Tensor* tin = static_cast<TF_Tensor*>(input);
    if (!tin) { fprintf(stderr, "TF_MaxPool_Native: null input\n"); return nullptr; }
    if (TF_TensorType(tin) != TF_FLOAT) { fprintf(stderr, "TF_MaxPool_Native: non-float input\n"); return nullptr; }
    bool is_nchw = (layout && std::strcmp(layout, "NCHW") == 0);
    int nd = TF_NumDims(tin);
    if (nd != 4) { fprintf(stderr, "TF_MaxPool_Native: expects 4D input\n"); return nullptr; }

    int64_t batch = TF_Dim(tin, 0);
    int64_t in_h = is_nchw ? TF_Dim(tin, 2) : TF_Dim(tin, 1);
    int64_t in_w = is_nchw ? TF_Dim(tin, 3) : TF_Dim(tin, 2);
    int64_t channels = is_nchw ? TF_Dim(tin, 1) : TF_Dim(tin, 3);

    int s_h = std::max(1, stride_h);
    int s_w = std::max(1, stride_w);
    int64_t out_h=0, out_w=0, pad_top=0, pad_left=0;
    std::string pad = padding ? std::string(padding) : "VALID";
    if (pad == "SAME") {
        out_h = (in_h + s_h - 1) / s_h;
        out_w = (in_w + s_w - 1) / s_w;
        int64_t pad_h = std::max<int64_t>(0, (out_h - 1) * s_h + k_h - in_h);
        int64_t pad_w = std::max<int64_t>(0, (out_w - 1) * s_w + k_w - in_w);
        pad_top = pad_h / 2; pad_left = pad_w / 2;
    } else {
        out_h = (in_h - k_h) / s_h + 1;
        out_w = (in_w - k_w) / s_w + 1;
    }

    std::vector<int64_t> out_dims(4);
    if (is_nchw) { out_dims = {batch, channels, out_h, out_w}; }
    else { out_dims = {batch, out_h, out_w, channels}; }
    size_t out_count = static_cast<size_t>(batch) * out_h * out_w * static_cast<size_t>(channels);
    float* out_data = static_cast<float*>(malloc(out_count * sizeof(float)));
    if (!out_data) { fprintf(stderr, "TF_MaxPool_Native: malloc failed\n"); return nullptr; }
    // initialize to very small number
    for (size_t i = 0; i < out_count; ++i) out_data[i] = -INFINITY;

    float* in_data = static_cast<float*>(TF_TensorData(tin));
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oh = 0; oh < out_h; ++oh) {
            for (int64_t ow = 0; ow < out_w; ++ow) {
                for (int64_t c = 0; c < channels; ++c) {
                    float best = -INFINITY;
                    for (int64_t kh = 0; kh < k_h; ++kh) {
                        for (int64_t kw = 0; kw < k_w; ++kw) {
                            int64_t ih = oh * s_h + kh - pad_top;
                            int64_t iw = ow * s_w + kw - pad_left;
                            if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) continue;
                            size_t in_idx;
                            if (is_nchw) in_idx = ((size_t)n * channels + c) * (in_h * in_w) + (ih * in_w + iw);
                            else in_idx = ((size_t)n * in_h + ih) * (in_w * channels) + (iw * channels) + c;
                            best = std::max(best, in_data[in_idx]);
                        }
                    }
                    size_t out_idx;
                    if (is_nchw) out_idx = ((size_t)n * channels + c) * (out_h * out_w) + (oh * out_w + ow);
                    else out_idx = ((size_t)n * out_h + oh) * (out_w * channels) + (ow * channels) + c;
                    out_data[out_idx] = best;
                }
            }
        }
    }

    TF_Tensor* tout = TF_NewTensor(TF_FLOAT, out_dims.data(), static_cast<int>(out_dims.size()), out_data, out_count * sizeof(float),
                                    [](void* data, size_t, void*) { free(data); }, nullptr);
    if (!tout) { free(out_data); fprintf(stderr, "TF_MaxPool_Native: TF_NewTensor failed\n"); return nullptr; }
    return static_cast<void*>(tout);
}

EXPORT void* TF_AvgPool_Native(void* input, int k_h, int k_w, int stride_h, int stride_w, const char* padding, const char* layout) {
    TF_Tensor* tin = static_cast<TF_Tensor*>(input);
    if (!tin) { fprintf(stderr, "TF_AvgPool_Native: null input\n"); return nullptr; }
    if (TF_TensorType(tin) != TF_FLOAT) { fprintf(stderr, "TF_AvgPool_Native: non-float input\n"); return nullptr; }
    bool is_nchw = (layout && std::strcmp(layout, "NCHW") == 0);
    int nd = TF_NumDims(tin);
    if (nd != 4) { fprintf(stderr, "TF_AvgPool_Native: expects 4D input\n"); return nullptr; }

    int64_t batch = TF_Dim(tin, 0);
    int64_t in_h = is_nchw ? TF_Dim(tin, 2) : TF_Dim(tin, 1);
    int64_t in_w = is_nchw ? TF_Dim(tin, 3) : TF_Dim(tin, 2);
    int64_t channels = is_nchw ? TF_Dim(tin, 1) : TF_Dim(tin, 3);

    int s_h = std::max(1, stride_h);
    int s_w = std::max(1, stride_w);
    int64_t out_h=0, out_w=0, pad_top=0, pad_left=0;
    std::string pad = padding ? std::string(padding) : "VALID";
    if (pad == "SAME") {
        out_h = (in_h + s_h - 1) / s_h;
        out_w = (in_w + s_w - 1) / s_w;
        int64_t pad_h = std::max<int64_t>(0, (out_h - 1) * s_h + k_h - in_h);
        int64_t pad_w = std::max<int64_t>(0, (out_w - 1) * s_w + k_w - in_w);
        pad_top = pad_h / 2; pad_left = pad_w / 2;
    } else {
        out_h = (in_h - k_h) / s_h + 1;
        out_w = (in_w - k_w) / s_w + 1;
    }

    std::vector<int64_t> out_dims(4);
    if (is_nchw) { out_dims = {batch, channels, out_h, out_w}; }
    else { out_dims = {batch, out_h, out_w, channels}; }
    size_t out_count = static_cast<size_t>(batch) * out_h * out_w * static_cast<size_t>(channels);
    float* out_data = static_cast<float*>(calloc(out_count, sizeof(float)));
    if (!out_data) { fprintf(stderr, "TF_AvgPool_Native: calloc failed\n"); return nullptr; }

    float* in_data = static_cast<float*>(TF_TensorData(tin));
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oh = 0; oh < out_h; ++oh) {
            for (int64_t ow = 0; ow < out_w; ++ow) {
                for (int64_t c = 0; c < channels; ++c) {
                    double acc = 0.0;
                    int64_t count = 0;
                    for (int64_t kh = 0; kh < k_h; ++kh) {
                        for (int64_t kw = 0; kw < k_w; ++kw) {
                            int64_t ih = oh * s_h + kh - pad_top;
                            int64_t iw = ow * s_w + kw - pad_left;
                            if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) continue;
                            size_t in_idx;
                            if (is_nchw) in_idx = ((size_t)n * channels + c) * (in_h * in_w) + (ih * in_w + iw);
                            else in_idx = ((size_t)n * in_h + ih) * (in_w * channels) + (iw * channels) + c;
                            acc += static_cast<double>(in_data[in_idx]);
                            ++count;
                        }
                    }
                    size_t out_idx;
                    if (is_nchw) out_idx = ((size_t)n * channels + c) * (out_h * out_w) + (oh * out_w + ow);
                    else out_idx = ((size_t)n * out_h + oh) * (out_w * channels) + (ow * channels) + c;
                    out_data[out_idx] = count > 0 ? static_cast<float>(acc / count) : 0.0f;
                }
            }
        }
    }

    TF_Tensor* tout = TF_NewTensor(TF_FLOAT, out_dims.data(), static_cast<int>(out_dims.size()), out_data, out_count * sizeof(float),
                                    [](void* data, size_t, void*) { free(data); }, nullptr);
    if (!tout) { free(out_data); fprintf(stderr, "TF_AvgPool_Native: TF_NewTensor failed\n"); return nullptr; }
    return static_cast<void*>(tout);
}
}