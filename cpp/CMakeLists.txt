cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ai_copper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

if(NOT DEFINED LIBTORCH AND NOT DEFINED ENV{LIBTORCH})
    message(FATAL_ERROR "Você precisa definir a variável LIBTORCH como -DLIBTORCH=... ou exportar no ambiente.")
endif()

if(NOT DEFINED LIBTORCH)
    set(LIBTORCH $ENV{LIBTORCH})
endif()

set(Torch_DIR "${LIBTORCH}/share/cmake/Torch")
find_package(Torch REQUIRED)

# Configuração do TensorFlow
find_library(TF_LIBRARY
    NAMES tensorflow tensorflow.lib
    HINTS
        $ENV{TENSORFLOW_ROOT}/lib
        "C:/libtensorflow/lib"
        "/usr/local/lib"
        "/usr/lib"
    DOC "Caminho para a biblioteca TensorFlow"
)

find_path(TF_INCLUDE_DIR
    NAMES tensorflow/c/c_api.h
    HINTS
        $ENV{TENSORFLOW_ROOT}/include
        "C:/libtensorflow/include"
        "/usr/local/include"
        "/usr/include"
    DOC "Caminho para os cabeçalhos do TensorFlow"
)

if(NOT TF_LIBRARY OR NOT TF_INCLUDE_DIR)
    message(FATAL_ERROR "TensorFlow não encontrado. Verifique TENSORFLOW_ROOT ou instale em C:/libtensorflow.")
endif()

include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${TF_INCLUDE_DIR})

set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE) # Usar Debug para alinhar com o Rust

if(WIN32)
    add_library(ai_copper STATIC lib.cpp)
else()
    add_library(ai_copper SHARED lib.cpp)
endif()

target_link_libraries(ai_copper "${TORCH_LIBRARIES}" ${TF_LIBRARY})
target_include_directories(ai_copper PRIVATE "${TORCH_INCLUDE_DIRS}" "${TF_INCLUDE_DIR}")

message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libs: ${TORCH_LIBRARIES}")
message(STATUS "LIBTORCH path: ${LIBTORCH}")
message(STATUS "TensorFlow include dir: ${TF_INCLUDE_DIR}")
message(STATUS "TensorFlow library: ${TF_LIBRARY}")