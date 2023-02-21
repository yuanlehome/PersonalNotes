
#pragma once

#include "NvInferPlugin.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "util.h"
#include "cuda_fp16.h"

class LReLU : public IPluginV2DynamicExt{
public:
    LReLU(float negSlope) : mNegSlope(negSlope) { }

    LReLU(float negSlope, nvinfer1::Dims input_dims):mNegSlope(negSlope), input_dims_(input_dims) { }

    LReLU(const void* buffer, size_t length);

    ~LReLU() override = default;

    const char* getPluginType() const noexcept override { return "leak_relu"; }

    const char* getPluginVersion() const noexcept override { return "1"; }

    int getNbOutputs() const noexcept override { return 1; }

    int initialize() noexcept override { return 0; }

    void terminate() noexcept override { }

    size_t getSerializationSize() const noexcept override { 
        return sizeof(float) + sizeof(Dims); // mNegSlope + input_dims
    }

    void serialize(void* buffer) const noexcept override;

    void destroy() noexcept override { 
        delete this; 
    }
    
    void setPluginNamespace(const char* libNamespace) noexcept override { namespace_ = libNamespace; }

    const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }


    /*IPluginV2Ext method*/
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override {return inputTypes[index];}


    /*IPluginV2DynamicExt method*/
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override { return new LReLU(mNegSlope, input_dims_); };

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override{ 
          return inOut[0].type == nvinfer1::DataType::kFLOAT;
        // return true;
    }

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc* outputs,
        int32_t nbOutputs) const noexcept override{ return 0; }

    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    float mNegSlope = 0;
    Dims input_dims_;
    std::string namespace_;
    int a_ = 0;
};

class LReluPluginCreator : public nvinfer1::IPluginCreator{
public:
    LReluPluginCreator() { }

    ~LReluPluginCreator() override = default;

    const char* getPluginName() const noexcept override { return "leak_relu"; }

    const char* getPluginVersion() const noexcept override { return "1"; }

    void setPluginNamespace(char const* pluginNamespace) noexcept override { 
        plugin_namespace_ = pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept override { 
        return plugin_namespace_.c_str();
    }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return nullptr; }

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override {
        return nullptr;
    }

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override { 
        return new LReLU(serialData, serialLength); 
    }

private:
    std::string plugin_namespace_;
};

REGISTER_TENSORRT_PLUGIN(LReluPluginCreator);