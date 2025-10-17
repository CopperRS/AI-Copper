cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ai_copper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# Variáveis de ambiente
if(NOT DEFINED ENV{LIBTORCH})
    message(FATAL_ERROR "Variável de ambiente LIBTORCH não definida.")
endif()
if(NOT DEFINED ENV{TENSORFLOW_ROOT})
    message(FATAL_ERROR "Variável de ambiente TENSORFLOW_ROOT não definida.")
endif()

set(LIBTORCH "$ENV{LIBTORCH}")
set(TENSORFLOW_ROOT "$ENV{TENSORFLOW_ROOT}")

# Encontra o Torch
set(Torch_DIR "${LIBTORCH}/share/cmake/Torch")
find_package(Torch REQUIRED PATHS ${Torch_DIR} NO_DEFAULT_PATH)

# Encontra o TensorFlow
find_library(TF_LIBRARY
    NAMES tensorflow.lib tensorflow
    HINTS "${TENSORFLOW_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)
find_path(TF_INCLUDE_DIR
    NAMES tensorflow/c/c_api.h
    HINTS "${TENSORFLOW_ROOT}/include"
    NO_DEFAULT_PATH
    REQUIRED
)

if(NOT TF_LIBRARY)
    message(FATAL_ERROR "TensorFlow library not found. Check TENSORFLOW_ROOT.")
endif()
if(NOT TF_INCLUDE_DIR)
    message(FATAL_ERROR "TensorFlow include directory not found. Check TENSORFLOW_ROOT.")
endif()

# Biblioteca principal
add_library(ai_copper SHARED lib.cpp)

# Define símbolos de exportação para Windows
if(MSVC)
    set_target_properties(ai_copper PROPERTIES
        OUTPUT_NAME "ai_copper"
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
    )
endif()

# Linkagem com Torch e TensorFlow
target_link_libraries(ai_copper PRIVATE "${TORCH_LIBRARIES}" "${TF_LIBRARY}")

# Inclui diretórios
target_include_directories(ai_copper PRIVATE "${TORCH_INCLUDE_DIRS}" "${TF_INCLUDE_DIR}")

# Mensagens de debug
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libs: ${TORCH_LIBRARIES}")
message(STATUS "LIBTORCH path: ${LIBTORCH}")
message(STATUS "TensorFlow include dir: ${TF_INCLUDE_DIR}")
message(STATUS "TensorFlow library: ${TF_LIBRARY}")