
#include <iostream>
#include <cuda_runtime_api.h>
#include <vector>
#include <assert.h>
#include <sstream>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "custom_relu_dynamic_plugin.h"


using namespace nvinfer1;

inline std::string DimsToStr(Dims dims) {
  std::stringstream ss;
  for(size_t i = 0; i < dims.nbDims; i++) {
    ss << dims.d[i] << " ";
  }
  return ss.str();
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
  nvinfer1::IBuilder* builder = createInferBuilder(logger);
  uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

  // Add the Input tensor to the network
  Dims2 input_shape{-1, 4};
  nvinfer1::ITensor *input_tensor = network->addInput("input", DataType::kFLOAT, input_shape);
 
  // Add the plugin layer with hidden layer
  IPluginV2* pluginObj = new CustomReluDynamic();
  auto layer = network->addPluginV2(&input_tensor, 1, *pluginObj);

  // Add a name for the output of the layer so that the tensor can be bound to a memory buffer at inference time:
  layer->getOutput(0)->setName("output");
  // Mark it as the output of the entire network:
  network->markOutput(*layer->getOutput(0));

  // Building an Engine(optimize the network)
  IBuilderConfig* config = builder->createBuilderConfig();
  IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions("input", OptProfileSelector::kMIN, Dims2(1, 4));
  profile->setDimensions("input", OptProfileSelector::kOPT, Dims2(2, 4));
  profile->setDimensions("input", OptProfileSelector::kMAX, Dims2(4, 4));
  config->addOptimizationProfile(profile);

  // 序列化和反序列化
  IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config);
  IRuntime* runtime = createInferRuntime(logger);

  ICudaEngine* engine = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());
  // ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

  // Prepare input_data
  int32_t inputIndex = engine->getBindingIndex("input");
  int32_t outputIndex = engine->getBindingIndex("output");
  std::cout << "input format: " << engine->getBindingFormatDesc(inputIndex) << std::endl;
  std::cout << "output format: " << engine->getBindingFormatDesc(outputIndex) << std::endl;
  std::cout << "engine layers: " << engine->getNbLayers() << std::endl;

  std::vector<float> input(16, 0.1);
  std::vector<float> output(16);

  void *GPU_input_Buffer_ptr;  // a host ptr point to a GPU buffer
  void *GPU_output_Buffer_ptr;  // a host ptr point to a GPU buffer
  void* buffers[2];

  cudaMalloc(&GPU_input_Buffer_ptr, sizeof(float)*input.size()); //malloc gpu buffer for input
  cudaMalloc(&GPU_output_Buffer_ptr, sizeof(float)*output.size()); //malloc gpu buffer for output
  cudaMemcpy(GPU_input_Buffer_ptr, input.data(), input.size()*sizeof(float), cudaMemcpyHostToDevice); // copy input data from cpu to gpu
  
  buffers[inputIndex] = static_cast<void*>(GPU_input_Buffer_ptr);
  buffers[outputIndex] = static_cast<void*>(GPU_output_Buffer_ptr);

  // Performing Inference
  IExecutionContext *context = engine->createExecutionContext();
  context->setBindingDimensions(inputIndex, Dims2(4, 4));
  context->executeV2(buffers);
  // context->executeV2(buffers); // 多次执行

  // copy result data from gpu to cpu
  cudaMemcpy(output.data(), GPU_output_Buffer_ptr, output.size()*sizeof(float), cudaMemcpyDeviceToHost); 

  // display output
  std::cout << "output shape : " << DimsToStr(engine->getBindingDimensions(outputIndex)) << "\n";
  std::cout << "output data : \n";
  for(auto i : output)
    std::cout << i << " ";
  std::cout << std::endl;
}
