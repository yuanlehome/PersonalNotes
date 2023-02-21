#include "NvInfer.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <assert.h>

// using namespace nvinfer1;

#define DEFAULT_VALUE 1.0

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

size_t ProductOfDims(nvinfer1::Dims dims) {
  size_t result = 1;
  for(size_t i = 0; i < dims.nbDims; i++) {
    result *= dims.d[i];
  }
  return result;
}

std::string DimsToStr(nvinfer1::Dims dims) {
  std::stringstream ss;
  for(size_t i = 0; i < dims.nbDims; i++) {
    ss << dims.d[i] << " ";
  }
  return ss.str();
}

nvinfer1::Dims4 CalculateConv2dOutput(nvinfer1::Dims4 input, nvinfer1::Dims4 filter, nvinfer1::DimsHW kernel_size, nvinfer1::DimsHW stride, nvinfer1::DimsHW paddings) {
  /******dataformat is NCHW******/
  assert(input.d[1] == filter.d[1]); // assert same channel value
  assert(filter.d[2] == kernel_size.d[0]);
  assert(filter.d[3] == kernel_size.d[1]);

  nvinfer1::Dims4 output;
  output.d[0] = input.d[0]; //batch
  output.d[1] = filter.d[0]; //output channel

  int32_t kh = kernel_size.d[0];
  int32_t kw = kernel_size.d[1];
  int32_t sh = stride.d[0];
  int32_t sw = stride.d[1];
  
  output.d[2] = ((input.d[2] + 2 * paddings.d[0] - kh) / sh) + 1; //cal output height
  output.d[3] = ((input.d[3] + 2 * paddings.d[1] - kw) / sw) + 1; //cal output weight

  return output;
}

int main() {
  Logger logger;

  // Create a Network Definition
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
  uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);
  
  nvinfer1::Dims4 input_shape{3, 3, 8, 8};
  nvinfer1::Dims4 filter_shape{3, 3, 2, 2};
  nvinfer1::DimsHW kernel_size{2, 2};
  nvinfer1::DimsHW stride{1, 1};
  nvinfer1::DimsHW paddings{1, 1}; //2d padding (padding is symmetric)
  nvinfer1::Dims4 output_shape = CalculateConv2dOutput(input_shape, filter_shape, kernel_size, stride, paddings);

  // Add the Input layer to the network
  auto input_data = network->addInput("input", nvinfer1::DataType::kFLOAT, input_shape);
  
  // Add the Convolution layer with hidden layer input nodes, strides and weights for filter and bias.
  std::vector<float>filter(ProductOfDims(filter_shape), DEFAULT_VALUE);
  nvinfer1::Weights filter_w{nvinfer1::DataType::kFLOAT, filter.data(), filter.size()};
  nvinfer1::Weights bias_w{nvinfer1::DataType::kFLOAT, nullptr, 0}; // no bias
  int32_t output_channel = filter_shape.d[0];
  auto conv2d = network->addConvolutionNd(*input_data, output_channel, kernel_size, filter_w, bias_w);
  conv2d->setStrideNd(stride);
  conv2d->setPaddingNd(paddings);

  // Add a name for the output of the conv2d layer so that the tensor can be bound to a memory buffer at inference time:
  conv2d->getOutput(0)->setName("output");
  // Mark it as the output of the entire network:
  network->markOutput(*conv2d->getOutput(0));

  // Building an Engine(optimize the network)
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  nvinfer1::IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config);

  // Creat Runtime
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());

  // Prepare input_data
  int32_t inputIndex = engine->getBindingIndex("input");
  int32_t outputIndex = engine->getBindingIndex("output");
  std::vector<float> input(ProductOfDims(input_shape), DEFAULT_VALUE);
  std::vector<float> output(ProductOfDims(output_shape));
  void *GPU_input_Buffer_ptr;  // a host ptr point to a GPU buffer
  void *GPU_output_Buffer_ptr;  // a host ptr point to a GPU buffer
  void* buffers[2];
  cudaMalloc(&GPU_input_Buffer_ptr, sizeof(float)*input.size()); //malloc gpu buffer for input
  cudaMalloc(&GPU_output_Buffer_ptr, sizeof(float)*output.size()); //malloc gpu buffer for output
  cudaMemcpy(GPU_input_Buffer_ptr, input.data(), input.size()*sizeof(float), cudaMemcpyHostToDevice); // copy input data from cpu to gpu
  buffers[inputIndex] = static_cast<void*>(GPU_input_Buffer_ptr);
  buffers[outputIndex] = static_cast<void*>(GPU_output_Buffer_ptr);

  // Performing Inference
  nvinfer1::IExecutionContext *context = engine->createExecutionContext();
  context->executeV2(buffers);

  // copy result data from gpu to cpu
  cudaMemcpy(output.data(), GPU_output_Buffer_ptr, output.size()*sizeof(float), cudaMemcpyDeviceToHost); 

  // display output
  std::cout << "output shape : " << DimsToStr(output_shape) << "\n";
  std::cout << "output data : \n";
  for(auto i : output)
    std::cout << i << " ";
  std::cout << std::endl;
}