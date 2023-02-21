#include "NvInferPlugin.h"
#include "LRelu_Dynamic_Plugin.h"
#include "cuda_fp16.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <memory.h>

template<typename data_type>
__global__ void add(data_type* x, data_type* y, float negSlope, int n)
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        if(x[i] > 0)
            y[i] = x[i];
        else
            y[i] = negSlope * x[i];
    }
}

template< >__global__ void add(__half* x, __half* y, float negSlope, int n)
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        if(__hgt(x[i], 0.0))
            y[i] = x[i];
        else
            y[i] = __hmul(negSlope, x[i]);
    }
}

template<typename data_type>
int ReluCudaKernel(float negSlope, const void* input, void* output, int nbInputEle) {
    // 定义kernel的执行配置
    dim3 gridSize(1);
    dim3 blockSize(16);
    // 执行kernel
    add<data_type> <<< gridSize, blockSize >>>((data_type*)input, (data_type*)output, negSlope, nbInputEle);
    return 0;
}   

LReLU::LReLU(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    mNegSlope = read<float>(d);
    input_dims_ = read<Dims>(d);
    assert(d == a + length);
}

void LReLU::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write<float>(d, mNegSlope);
    write<Dims>(d, input_dims_);
    assert(d == a + getSerializationSize());
}

DimsExprs LReLU::getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    assert(nbInputs == 1);
    return inputs[0];
}

void LReLU::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept 
{
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
    input_dims_ = in[0].desc.dims;
}

int32_t LReLU::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
     
    int status = 0;
    // if(inputDesc[0].type == nvinfer1::DataType::kFLOAT){
        status = ReluCudaKernel<float>(mNegSlope, inputData, outputData, ProductOfDims(input_dims_));
    // }
    // else if(inputDesc[0].type == nvinfer1::DataType::kHALF) {
    //     status = ReluCudaKernel<__half>(mNegSlope, inputData, outputData, ProductOfDims(input_dims_));
    // }
    // else {
    //     assert(status);
    // }
    return status;
}