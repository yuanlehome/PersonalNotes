project(paddle_inference_demo)
cmake_minimum_required(VERSION 3.16)

include_directories(/weishengying/Paddle/build/paddle_inference_install_dir/)

add_executable(a.out gpt_inference.cc)

target_link_libraries(a.out /weishengying/Paddle/build/paddle_inference_install_dir/paddle/lib/libpaddle_inference.so)