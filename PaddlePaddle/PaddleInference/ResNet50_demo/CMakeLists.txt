project(paddle_inference_demo)
cmake_minimum_required(VERSION 3.16)

include_directories(/weishengying/PersonalNotes/PaddlePaddle/PaddleInference/ResNet50_demo/paddle)
add_executable(a.out resnet50_gpu.cc)

target_link_libraries(a.out /weishengying/PersonalNotes/PaddlePaddle/PaddleInference/ResNet50_demo/paddle/paddle/lib/libpaddle_inference.so)
