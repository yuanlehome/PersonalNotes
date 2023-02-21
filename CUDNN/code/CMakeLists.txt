project(cudnn_demo)
cmake_minimum_required(VERSION 3.16)

set(CUDA_PATH /usr/local/cuda)
include_directories(${CUDA_PATH}/include)

function(compile_demo target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} ${CUDA_PATH}/lib64/libcudart.so libcudnn.so)
endfunction(compile_demo)

compile_demo(minima_cudnn_cpp_example minima_cudnn_cpp_example.cc)
