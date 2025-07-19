cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ai_copper)

# C++ padrão
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define runtime correto para MSVC
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# Verifica se LIBTORCH foi definido
if(NOT DEFINED LIBTORCH AND NOT DEFINED ENV{LIBTORCH})
    message(FATAL_ERROR "Você precisa definir a variável LIBTORCH como -DLIBTORCH=... ou exportar no ambiente.")
endif()

# Usa ENV{LIBTORCH} se LIBTORCH não foi passado diretamente
if(NOT DEFINED LIBTORCH)
    set(LIBTORCH $ENV{LIBTORCH})
endif()

# Aponta para o diretório cmake do libtorch
set(Torch_DIR "${LIBTORCH}/share/cmake/Torch")
find_package(Torch REQUIRED)

# Inclui headers
include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${TORCH_INCLUDE_DIRS})

# Força build Release
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# Adiciona a biblioteca (formato depende do SO)
if(WIN32)
    add_library(ai_copper STATIC lib.cpp)
else()
    add_library(ai_copper SHARED lib.cpp)
endif()

# Linka com libtorch
target_link_libraries(ai_copper "${TORCH_LIBRARIES}")
target_include_directories(ai_copper PRIVATE "${TORCH_INCLUDE_DIRS}")

# Opcional: melhora logs
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libs: ${TORCH_LIBRARIES}")
message(STATUS "LIBTORCH path: ${LIBTORCH}")
