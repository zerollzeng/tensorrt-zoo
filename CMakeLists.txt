cmake_minimum_required(VERSION 3.0)

project(tensorrt_zoo)
set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "opencv not found")
endif (NOT OpenCV_FOUND)

include_directories(tiny-tensorrt)
link_directories(/usr/lib/x86_64-linux-gnu)
link_directories(lib/)

find_library(CUDART cudart HINTS /usr/local/cuda/targets/x86_64-linux/lib/)

add_subdirectory(tiny-tensorrt)

add_subdirectory(yolov3)

add_subdirectory(openpose)

