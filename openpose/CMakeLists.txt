cmake_minimum_required(VERSION 3.0)

project(testopenpose)
set( CMAKE_CXX_FLAGS " -std=c++11" )

file(GLOB openpose_source
     *.cpp
     *.cu
     )
cuda_add_executable(testopenpose ${openpose_source})
target_compile_options(testopenpose PUBLIC -std=c++11 -Wall -Wfloat-conversion)
target_link_libraries(testopenpose tinytrt)
target_link_libraries(testopenpose nvinfer)
target_link_libraries(testopenpose nvinfer_plugin)
target_link_libraries(testopenpose nvparsers)
target_link_libraries(testopenpose nvonnxparser)
target_link_libraries(testopenpose nvcaffe_parser)
target_link_libraries(testopenpose ${CUDART})
target_link_libraries(testopenpose ${OpenCV_LIBRARIES})