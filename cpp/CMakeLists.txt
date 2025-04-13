cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ai_copper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{LIBTORCH})
    message(FATAL_ERROR "Defina a variável de ambiente LIBTORCH com o caminho para libtorch.")
endif()

set(Torch_DIR "$ENV{LIBTORCH}/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_SOURCE_DIR}/../include)

add_library(ai_copper SHARED lib.cpp)
target_link_libraries(ai_copper "${TORCH_LIBRARIES}")
target_include_directories(ai_copper PRIVATE "${TORCH_INCLUDE_DIRS}")