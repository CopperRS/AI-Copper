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

# Torch
set(Torch_DIR "${LIBTORCH}/share/cmake/Torch")
find_package(Torch REQUIRED PATHS ${Torch_DIR} NO_DEFAULT_PATH)

# TensorFlow
find_library(TF_LIBRARY
    NAMES tensorflow tensorflow.lib
    HINTS ${TENSORFLOW_ROOT}/lib
    NO_DEFAULT_PATH
)
find_path(TF_INCLUDE_DIR
    NAMES tensorflow/c/c_api.h
    HINTS ${TENSORFLOW_ROOT}/include
    NO_DEFAULT_PATH
)
if(NOT TF_LIBRARY OR NOT TF_INCLUDE_DIR)
    message(FATAL_ERROR "TensorFlow não encontrado. Verifique TENSORFLOW_ROOT.")
endif()

# Protobuf (Windows e Linux)
if(WIN32)
    set(PROTOBUF_CANDIDATES
        libprotobuf
        protobuf
        protobuf3
    )

    set(PROTOBUF_FOUND FALSE)
    foreach(name IN LISTS PROTOBUF_CANDIDATES)
        find_library(PROTOBUF_LIBRARY
            NAMES ${name}
            HINTS ${LIBTORCH}/lib
            NO_DEFAULT_PATH
        )
        if(PROTOBUF_LIBRARY)
            set(PROTOBUF_FOUND TRUE)
            break()
        endif()
    endforeach()

    if(NOT PROTOBUF_FOUND)
        message(FATAL_ERROR "PROTOBUF_LIBRARY não encontrado em ${LIBTORCH}/lib. Verifique se libprotobuf.lib existe ou se sua versão do LibTorch contém protobuf.")
    endif()
else()
    find_library(PROTOBUF_LIBRARY
        NAMES protobuf
        HINTS ${LIBTORCH}/lib
        NO_DEFAULT_PATH
    )
    if(NOT PROTOBUF_LIBRARY)
        message(FATAL_ERROR "PROTOBUF_LIBRARY não encontrado em ${LIBTORCH}/lib")
    endif()
endif()

# Includes
include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${TF_INCLUDE_DIR})

# Desativar MKL e TBB para evitar conflitos
set(TORCH_DISABLE_MKL ON CACHE BOOL "Desabilita MKL para evitar ittnotify.lib")
set(TORCH_DISABLE_TBB ON CACHE BOOL "Desabilita TBB para evitar ittnotify.lib")

# Biblioteca principal
add_library(ai_copper SHARED lib.cpp)

target_link_libraries(ai_copper PRIVATE 
    "${TORCH_LIBRARIES}" 
    ${TF_LIBRARY} 
    ${PROTOBUF_LIBRARY}
)
target_include_directories(ai_copper PRIVATE "${TORCH_INCLUDE_DIRS}" "${TF_INCLUDE_DIR}")

# Mensagens de debug
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libs: ${TORCH_LIBRARIES}")
message(STATUS "LIBTORCH path: ${LIBTORCH}")
message(STATUS "TensorFlow include dir: ${TF_INCLUDE_DIR}")
message(STATUS "TensorFlow library: ${TF_LIBRARY}")
message(STATUS "Protobuf library: ${PROTOBUF_LIBRARY}")
