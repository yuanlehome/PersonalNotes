project(onnx_runtime_demo)
cmake_minimum_required(VERSION 3.16)

include_directories(/weishengying/onnxruntime/include/onnxruntime/core/session)

add_executable(main main.cc)

target_link_libraries(main /weishengying/onnxruntime/build/Linux/Debug/libonnxruntime.so)