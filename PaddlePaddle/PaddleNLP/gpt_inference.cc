#include "paddle/include/paddle_inference_api.h"

#include <numeric>
#include <iostream>
#include <algorithm>
#include <chrono>

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  auto diff = t2 - t1;
  auto counter = std::chrono::duration_cast<std::chrono::microseconds>(diff);
  return counter.count() / 1000.0;
}

#define WARM_TIMES 10
#define REPEAT_TIMES 100

template<typename T>
void post_process(std::vector<T>& vec){
  auto it = std::max_element(vec.begin(), vec.end());
  std::cout << "max_ele_index: " << std::distance(vec.begin(), it) << std::endl;
  std::cout << "max_ele: " << *it << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "Usage: ./a.out model_file params_file" << std::endl;
    return -1;
  }
  std::string model_file = argv[1];
  std::string params_file = argv[2];

  // 创建默认配置对象
  paddle_infer::Config config;

  // 设置预测模型路径
  config.SetModel(model_file, params_file);

  // 启用 GPU
  config.EnableUseGpu(100, 0);

  // 开启 内存/显存 复用
  config.EnableMemoryOptim();

  config.SwitchIrDebug(1);

  auto predictor = paddle_infer::CreatePredictor(config);

  // 获取输入 Tensor
  auto input_names = predictor->GetInputNames();
  auto input_tensor = predictor->GetInputHandle(input_names[0]);

  // 设置输入 Tensor 的维度信息
  std::vector<int> INPUT_SHAPE = {1, 3, 224, 224};
  input_tensor->Reshape(INPUT_SHAPE);

  // 准备输入数据
  int input_size = 1 * 3 * 224 * 224;
  std::vector<float> input_data(input_size, 1);
  // 设置输入 Tensor 数据
  input_tensor->CopyFromCpu(input_data.data());

  // 执行预测
  for(int i = 0; i < WARM_TIMES; i++)
    predictor->Run();

  auto start_time = time();
  for(int i = 0; i < REPEAT_TIMES; i++)
    predictor->Run();
  std::cout << "avg time coast: " << time_diff(start_time, time()) / REPEAT_TIMES << "ms" << std::endl;

  // 获取 Output Tensor
  auto output_names = predictor->GetOutputNames();
  auto output_tensor = predictor->GetOutputHandle(output_names[0]);

  // 获取 Output Tensor 的维度信息
  std::vector<int> output_shape = output_tensor->shape();
  int output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                    std::multiplies<int>());

  // 获取 Output Tensor 的数据
  std::vector<float> output_data;
  output_data.resize(output_size);
  output_tensor->CopyToCpu(output_data.data());

  post_process<float>(output_data);
}