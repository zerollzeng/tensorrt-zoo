cmake_minimum_required(VERSION 3.0)

project(testonnx)
# link_directories(/usr/lib/x86_64-linux-gnu)

file(GLOB onnx_source
     *.cpp
     )
add_executable(testonnx ${onnx_source})
target_compile_options(testonnx PUBLIC -std=c++11 -Wall -Wfloat-conversion)
target_link_libraries(testonnx tinytrt)
target_link_libraries(testonnx nvinfer)
target_link_libraries(testonnx nvinfer_plugin)
target_link_libraries(testonnx nvparsers)
target_link_libraries(testonnx nvonnxparser)
target_link_libraries(testonnx nvcaffe_parser)
target_link_libraries(testonnx ${CUDART})
target_link_libraries(testonnx ${OpenCV_LIBRARIES})