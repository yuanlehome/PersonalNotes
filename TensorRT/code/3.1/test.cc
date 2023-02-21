
#include <iostream>
#include <cuda_runtime_api.h>
#include <vector>
#include <assert.h>
#include <chrono>
#include <typeinfo> 

#include "cuda_fp16.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "LRelu_Dynamic_Plugin.h"

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

class Logger : public ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kERROR)
            std::cout << msg << std::endl;
    }
};

int main() {
  Logger logger;

  // Create a Network Definition
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
  uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

  // Add the Input layer to the network
  nvinfer1::ITensor* input_tensor = network->addInput("input", DataType::kFLOAT, nvinfer1::Dims4{-1, 64, 256, 256});
  
  // Add the plugin layer with hidden layer
  nvinfer1::IPluginV2* pluginObj = new LReLU(-0.1);
  assert(pluginObj);
  auto layer = network->addPluginV2(&input_tensor, 1, *pluginObj);
  assert(layer);

  // Add a name for the output of the layer so that the tensor can be bound to a memory buffer at inference time:
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  output_tensor->setName("output");
  // Mark it as the output of the entire network:
  network->markOutput(*output_tensor);

  // Creat a BuilderConfig
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  config->setFlag(BuilderFlag::kFP16);
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, Dims4(1, 64, 256, 256));
  profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, Dims4(3, 64, 256, 256));
  profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, Dims4(4, 64, 256, 256));
  config->addOptimizationProfile(profile);

  // Building an Engine(optimize the network)
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

  // Show the layer in Engine
  nvinfer1::IEngineInspector *inspector = engine->createEngineInspector();
  std::cout << inspector->getEngineInformation(LayerInformationFormat::kJSON) << std::endl;

  // Prepare input_data
  using in_data_type = float;
  using out_data_type = float;
  std::vector<in_data_type> input(3*64*256*256, 1.0); //runtime shape {3,64,256,256}
  std::vector<out_data_type> output(3*64*256*256);

  void *GPU_input_Buffer_ptr;  // a host ptr point to a GPU buffer
  void *GPU_output_Buffer_ptr;  // a host ptr point to a GPU buffer
  void* buffers[2];

  cudaMalloc(&GPU_input_Buffer_ptr, sizeof(in_data_type)*input.size()); //malloc gpu buffer for input
  cudaMalloc(&GPU_output_Buffer_ptr, sizeof(out_data_type)*output.size()); //malloc gpu buffer for output
  cudaMemcpy(GPU_input_Buffer_ptr, input.data(), input.size()*sizeof(in_data_type), cudaMemcpyHostToDevice); // copy input data from cpu to gpu
  
  int32_t inputIndex = engine->getBindingIndex("input");
  int32_t outputIndex = engine->getBindingIndex("output");
  buffers[inputIndex] = static_cast<void*>(GPU_input_Buffer_ptr);
  buffers[outputIndex] = static_cast<void*>(GPU_output_Buffer_ptr);

  // Creat ExecutionContext
  IExecutionContext *context = engine->createExecutionContext();

  // Performing Inference
  context->setBindingDimensions(inputIndex, Dims4(3, 64, 256, 256));

  auto start_time = time();
  for(int i = 0; i < 1; i++) {
    context->executeV2(buffers); // Synchronously
  }

  std::cout << "run avg time is " << time_diff(start_time, time()) / 1000 
            << " ms" << std::endl;


  // Copy result data from gpu to cpu
  cudaMemcpy(output.data(), GPU_output_Buffer_ptr, output.size()*sizeof(out_data_type), cudaMemcpyDeviceToHost); 

  // Display output
  std::cout << "output data : \n";
  for(int i = 0; i < 20; i++) {
      std::cout << output[i] << " ";
      // std::cout << output[i] << " ";
  }
  std::cout << std::endl;
}
