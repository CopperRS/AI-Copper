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

include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${TORCH_INCLUDE_DIRS})


set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

if(WIN32)
    add_library(ai_copper STATIC lib.cpp)
else()
    add_library(ai_copper SHARED lib.cpp)
endif()

target_link_libraries(ai_copper "${TORCH_LIBRARIES}")
target_include_directories(ai_copper PRIVATE "${TORCH_INCLUDE_DIRS}")

message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libs: ${TORCH_LIBRARIES}")
message(STATUS "LIBTORCH path: ${LIBTORCH}")
