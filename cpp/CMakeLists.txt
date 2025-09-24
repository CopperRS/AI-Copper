cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ai_copper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

if(NOT DEFINED ENV{LIBTORCH})
    message(FATAL_ERROR "Variável de ambiente LIBTORCH não definida.")
endif()
if(NOT DEFINED ENV{TENSORFLOW_ROOT})
    message(FATAL_ERROR "Variável de ambiente TENSORFLOW_ROOT não definida.")
endif()

set(LIBTORCH "$ENV{LIBTORCH}")
set(TENSORFLOW_ROOT "$ENV{TENSORFLOW_ROOT}")

set(Torch_DIR "${LIBTORCH}/share/cmake/Torch")
find_package(Torch REQUIRED PATHS ${Torch_DIR} NO_DEFAULT_PATH)

# TensorFlow lib
find_library(TF_LIBRARY
    NAMES tensorflow tensorflow.lib
    HINTS ${TENSORFLOW_ROOT}/lib
    NO_DEFAULT_PATH
    DOC "Caminho para a biblioteca TensorFlow"
)

# Ajuste do Protobuf (LibTorch fornece libprotobuf.lib)
if(WIN32)
    find_library(PROTOBUF_LIBRARY
        NAMES libprotobuf protobuf
        HINTS ${LIBTORCH}/lib ${TENSORFLOW_ROOT}/lib
        NO_DEFAULT_PATH
        DOC "Caminho para a biblioteca protobuf"
    )
else()
    find_library(PROTOBUF_LIBRARY
        NAMES protobuf
        HINTS ${LIBTORCH}/lib ${TENSORFLOW_ROOT}/lib
        NO_DEFAULT_PATH
    )
endif()

find_path(TF_INCLUDE_DIR
    NAMES tensorflow/c/c_api.h
    HINTS ${TENSORFLOW_ROOT}/include
    NO_DEFAULT_PATH
    DOC "Caminho para os cabeçalhos do TensorFlow"
)

if(NOT TF_LIBRARY OR NOT TF_INCLUDE_DIR)
    message(FATAL_ERROR "TensorFlow não encontrado. Verifique TENSORFLOW_ROOT.")
endif()

include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${TF_INCLUDE_DIR})

set(TORCH_DISABLE_MKL ON CACHE BOOL "Desabilita MKL para evitar ittnotify.lib")
set(TORCH_DISABLE_TBB ON CACHE BOOL "Desabilita TBB para evitar ittnotify.lib")

add_library(ai_copper SHARED lib.cpp)

target_link_libraries(ai_copper PRIVATE "${TORCH_LIBRARIES}" ${TF_LIBRARY} ${PROTOBUF_LIBRARY})
target_include_directories(ai_copper PRIVATE "${TORCH_INCLUDE_DIRS}" "${TF_INCLUDE_DIR}")

message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libs: ${TORCH_LIBRARIES}")
message(STATUS "LIBTORCH path: ${LIBTORCH}")
message(STATUS "TensorFlow include dir: ${TF_INCLUDE_DIR}")
message(STATUS "TensorFlow library: ${TF_LIBRARY}")
message(STATUS "Protobuf library: ${PROTOBUF_LIBRARY}")
