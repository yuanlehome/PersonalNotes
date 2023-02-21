#include "NvInferPlugin.h"
#include "LeakRelu_Plugin.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <memory.h>

using namespace nvinfer1;

__global__ void add(float* x, float* y, float negSlope, int n)
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

int ReluCudaKernel(float negSlope, const void* input, void* output, int nbInputEle) {
    // 定义kernel的执行配置
    dim3 gridSize(1);
    dim3 blockSize(16);
    // 执行kernel
    add <<< gridSize, blockSize >>>((float*)input, (float*)output, negSlope, nbInputEle);
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


IPluginV2DynamicExt* LReLU::clone() const noexcept {
    auto* plugin = new LReLU(mNegSlope, input_dims_);
    return plugin;
}


DimsExprs LReLU::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    assert(nbInputs == 1);
    return inputs[0];
    return DimsExprs();
}

void LReLU::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
        const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept 
{
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
    input_dims_ = in[0].desc.dims;
}

int32_t LReLU::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    int status = ReluCudaKernel(mNegSlope, inputData, outputData, ProductOfDims(input_dims_));
    return status;
}