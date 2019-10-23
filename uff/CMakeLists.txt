cmake_minimum_required(VERSION 3.0)

project(testuff)
# link_directories(/usr/lib/x86_64-linux-gnu)

file(GLOB uff_source
     *.cpp
     )
add_executable(testuff ${uff_source})
target_compile_options(testuff PUBLIC -std=c++11 -Wall -Wfloat-conversion)
target_link_libraries(testuff tinytrt)
target_link_libraries(testuff nvinfer)
target_link_libraries(testuff nvinfer_plugin)
target_link_libraries(testuff nvparsers)
target_link_libraries(testuff nvonnxparser)
target_link_libraries(testuff nvcaffe_parser)
target_link_libraries(testuff ${CUDART})
target_link_libraries(testuff ${OpenCV_LIBRARIES})