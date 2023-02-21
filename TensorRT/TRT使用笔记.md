# TensorRT 8.4.X 使用笔记
tensorrt 使用笔记，方便日后查阅。

# 1.安装
略

# 2.最简单的使用
[IConvolutionLayer.cc](./IConvolutionLayer.cc) 中使用 trt 构建了一个 network， 该 network 中只有一个 conv layer。

大致构建流程如下

```cpp
// 1. Create a Network Definition
nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

// 2. Add the Input layer to the network
auto input_data = network->addInput("input", nvinfer1::DataType::kFLOAT, input_shape);

// 3. Add the Convolution layer with hidden layer input nodes, strides and weights for filter and bias.
auto conv2d = network->addConvolutionNd(*input_data, output_channel, kernel_size, filter_w, bias_w);
conv2d->setStrideNd(stride);
conv2d->setPaddingNd(paddings);
...

// 4. Building an Engine(optimize the network)
nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
nvinfer1::IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config);

// 5. Creat Runtime
nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());

// 6. Prepare input_data
...

// 7. Performing Inference
nvinfer1::IExecutionContext *context = engine->createExecutionContext();
context->executeV2(buffers);

// 8. Process Output
...
```
完整代码见文件 [IConvolutionLayer.cc](./IConvolutionLayer.cc)。

使用上大致有上述 8 个过程，主要也只有三个阶段: `构建 network`， `创建一个 Engine`，`Runtime`。

network 的构建是核心阶段，可以使用官方提供的 Layer 或者 Plugin 来组网，上面的例子只使用了 addConvolutionNd 这个 api，构建了一个只有 conv 的网络。需要跑什么样的模型，就组什么样的 network。“实际模型的结构可能非常复杂，但只需要根据算子的执行顺序（topo排序）构建 network 即可”

创建好 network 之后，构建一个 Engine，可以对 network 进行一些优化，trt 内部会根据 network 创建一个 graph，然后进行一些图层面上的优化，比如算子融合，常量消除等等。然后将优化之后的模型创建为一个 engine。可以把这个 engine 序列化保存。以后可以直接反序列化为 engine 直接使用。

Runtime 阶段，先将之前保存的 engine 反序列化，创建一个执行环境，配置输入数据，然后就可以调用 execute 运行得到结果了。

# 
