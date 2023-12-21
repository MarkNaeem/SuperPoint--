cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(SuperPoint++)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)


find_package(OpenCV REQUIRED)

set(USE_CUDNN 1 CACHE BOOL "Compile with cuDNN support")
set(USE_CUSPARSELT 1 CACHE BOOL "Compile with cuSPARSELt support")

# Set the paths for LibTorch
set(CMAKE_PREFIX_PATH  "~/libtorch/share/cmake/Torch" ${CMAKE_PREFIX_PATH})

# Find CUDA
# set(CMAKE_PREFIX_PATH  "/usr/local/cuda" ${CMAKE_PREFIX_PATH})
# find_package(CUDA REQUIRED)
# message(STATUS "Found CUDA: ${CUDA_VERSION_STRING} located at ${CUDA_TOOLKIT_ROOT_DIR}")


# Find the Torch library
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(include ${OpenCV_INCLUDE_DIRS}  ${TORCH_INCLUDE_DIRS})

# Add SuperPoint++ as a library
add_library(toolsLib src/tools.cpp)
# add_library(SuperPointLib src/SuperPoint.cpp)

# Define the main executable
add_executable(SuperPoint++ src/main.cpp)

# Link SuperPoint++Lib and LibTorch to the executable
target_link_libraries(SuperPoint++ toolsLib  ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} "${TORCH_LIBRARIES}")

# Setting this to YES helps with some compilation issues
set_property(TARGET SuperPoint++ PROPERTY CXX_STANDARD 17)
