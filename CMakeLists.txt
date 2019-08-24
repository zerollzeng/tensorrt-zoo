cmake_minimum_required(VERSION 3.0)

project(tensorrt_zoo)

find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "opencv not found")
endif (NOT OpenCV_FOUND)

link_directories(/usr/lib/x86_64-linux-gnu)

find_library(CUDART cudart HINTS /usr/local/cuda/targets/x86_64-linux/lib/)

add_subdirectory(yolov3)