/*******************************************************************************
参考 https://oneapi-src.github.io/oneDNN/page_inner_product_example_cpp.html#doxid-inner-product-example-cpp
编译方式: 
g++ -I ../thirdparty/install/onednn/include/ -L ../thirdparty/install/onednn/lib -ldnnl InnerProductPrimitiveExample_quant.cc
export LD_LIBRARY_PATH=/weishengying/third_party/install/onednn/lib

相比 InnerProductPrimitiveExample.cc, 本单测测试 int8 计算。

src_md 和 weight_md 的 dtype 设置为 s8， bias_md 的 dtype 设置为 f32， dst_md 的 dtype 设置为 f32
则运行时原语内部的计算过程为： (Q_src * Q_weight) *（src_scale * weight_scale） + F32_bias
其中 Q_src * Q_weight 的计算结果使用 s32 保存，如果传入了 dst_scale，该值会被忽略， 如果没有给原语传入 src_scale 和 weight_scale，
则计算简化为 F32_out = (Q_src * Q_weight) + F32_bias


src_md 和 weight_md 的 dtype 设置为 s8， bias_md 的 dtype 设置为 f32， dst_md 的 dtype 设置为 s8
则运行时原语内部的计算过程为： （(Q_src * Q_weight) *（src_scale * weight_scale）+ F32_bias）/ dst_scale
其中 Q_src * Q_weight 的计算结果使用 s32 保存，如果没有给原语传入 src_scale，weight_scale，dst_scale
则计算简化为 s8_out = (Q_src * Q_weight) + F32_bias

onednn 2.7 和 3.0 的区别
使用上： 
    2.7 创建原语 attr 的时候，使用 set_output_scales 接口直接传入 mask，scale 等信息；
    3.0 创建原语 attr 的时候，使用 set_scales_mask 接口直接传入 mask， scale 信息在原语执行的时候传。

计算流程上：
    3.0
        f32 输出时： (Q_src * Q_weight) *（src_scale * weight_scale）+ F32_bias
        s8 输出时： （(Q_src * Q_weight) *（src_scale * weight_scale）+ F32_bias）/ dst_scale
    2.3
        developer 需要提前使用 src_scale 和 weight_scale 将 f32 的 bias 量化为 s32 的 bias;
        s32_bias = f32_bias / (src_scale * weight_scale)

        fc 原语的内部计算过程为：
            f32 输出时： 
                output_scale = 1.0 / (src_scale * weight_scale)     #该 output_scale 便是传入 set_output_scales api 的参数
                (Q_src * Q_weight  + s32_bias ) / output_scale
    
            s8 输出时： 
                output_scale = dst_scale / (src_scale * weight_scale)   #该 output_scale 便是传入 set_output_scales api 的参数
                (Q_src * Q_weight  + s32_bias) / output_scale
        
        计算公式统一为： 
            (Q_src * Q_weight  + s32_bias) / output_scale
        根据输出类型的不同， developer 自行计算 output_scale，并使用 set_output_scales api传入。

        如果 fc 后面有激活函数：
          s8 输出时：
            output_scale = 1.0 / (src_scale * weight_scale)   #该 output_scale 便是传入 set_output_scales api 的参数
            swish((Q_src * Q_weight  + s32_bias ) / output_scale) / dst_scale
            dst_scale 由onednn post_ops api传入 #append_post_ops(swish, scale), 称为 activation_scale
          f32 输出时：
            output_scale = 1.0 / (src_scale * weight_scale)   #该 output_scale 便是传入 set_output_scales api 的参数
            swish((Q_src * Q_weight  + s32_bias ) / output_scale
            post_ops 的 activation_scale 可以认为是 1.0 
          

*******************************************************************************/


#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void inner_product_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 1, // batch size
            IC = 3, // input channels
            IH = 2, // tensor height
            IW = 2, // tensor width
            OC = 16; // output channels

    // Source (src), weights, bias, and destination (dst) tensors
    // dimensions.
    memory::dims src_dims = {N, IC, IH, IW};
    memory::dims weights_dims = {OC, IC, IH, IW};
    memory::dims bias_dims = {OC};
    memory::dims dst_dims = {N, OC};

    // Allocate user's buffers.
    std::vector<int8_t> src_data(product(src_dims));
    std::vector<int8_t> weights_data(product(weights_dims));
    std::vector<float> bias_data(OC);
    std::vector<int8_t> dst_data(product(dst_dims));

    // Initialize src, weights, and bias tensors.
    std::generate(src_data.begin(), src_data.end(), []() {
        return 1;
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        return 2;
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        return 0.1;
    });

    // Create memory descriptors and memory objects for src and weight. In this
    // example, NCHW and OIHW layout is assumed.
    auto user_src_md = memory::desc(src_dims, dt::s8, tag::nchw);
    auto user_weights_md = memory::desc(weights_dims, dt::s8, tag::oihw);
    auto user_bias_md = memory::desc(bias_dims, dt::f32, tag::a);
    auto user_dst_md = memory::desc(dst_dims, dt::s8, tag::nc);

    auto user_src_mem = memory(user_src_md, engine);
    auto user_weights_mem = memory(user_weights_md, engine);
    auto user_bias_mem = memory(user_bias_md, engine);
    auto user_dst_mem = memory(user_dst_md, engine);

    // Write data to memory object's handles.
    write_to_dnnl_memory(src_data.data(), user_src_mem);
    write_to_dnnl_memory(bias_data.data(), user_bias_mem);
    write_to_dnnl_memory(weights_data.data(), user_weights_mem);

    // Create memory descriptor for src/weights with format_tag::any. This enables
    // the inner product primitive to choose the memory layout for an optimized
    // primitive implementation, and this format may differ from the one
    // provided by the user.
    auto inner_product_src_md = memory::desc(src_dims, dt::s8, tag::any);
    auto inner_product_weights_md = memory::desc(weights_dims, dt::s8, tag::any);
    auto inner_product_bias_md = user_bias_md;
    auto inner_product_dst_md = user_dst_md; // For the destination tensor the memory format is always dnnl::memory::format_tag::nc
    
    // Create inner product primitive descriptor.
    primitive_attr inner_product_attr;
    const int src_mask = 0;
    const int weight_mask = 0;
    const int dst_mask = 0;
    inner_product_attr.set_scales_mask(DNNL_ARG_SRC, src_mask);
    inner_product_attr.set_scales_mask(DNNL_ARG_WEIGHTS, weight_mask);
    inner_product_attr.set_scales_mask(DNNL_ARG_DST, dst_mask);

    auto inner_product_pd = inner_product_forward::primitive_desc(engine,
            prop_kind::forward_inference, inner_product_src_md, inner_product_weights_md,
            inner_product_bias_md, inner_product_dst_md, inner_product_attr);
   
    // For now, assume that the src/weights memory layout generated by the primitive
    // and the one provided by the user are identical.
    auto inner_product_weights_mem = user_weights_mem;
    auto inner_product_src_mem = user_src_mem;

    // Reorder the data in case the src/weights memory layout generated by the
    // primitive and the one provided by the user are different. In this case,
    // we create additional memory objects with internal buffers that will
    // contain the reordered data.
    if (inner_product_pd.src_desc() != user_src_mem.get_desc()) {
        std::cout << "reorder src" << std::endl;
        inner_product_src_mem
                = memory(inner_product_pd.src_desc(), engine);
        reorder(user_src_mem, inner_product_src_mem)
                .execute(engine_stream, user_src_mem,
                        inner_product_src_mem);
    }
    if (inner_product_pd.weights_desc() != user_weights_mem.get_desc()) {
        std::cout << "reorder weight" << std::endl;
        inner_product_weights_mem
                = memory(inner_product_pd.weights_desc(), engine);
        reorder(user_weights_mem, inner_product_weights_mem)
                .execute(engine_stream, user_weights_mem,
                        inner_product_weights_mem);
    }

    // Create the primitive.
    auto inner_product_prim = inner_product_forward(inner_product_pd);

    // Primitive arguments.
    std::vector<float> src_scales = {2.0f};
    auto src_scale_md = memory::desc({1}, dt::f32, tag::x);
    auto src_scale_memory = memory(src_scale_md, engine);
    write_to_dnnl_memory(src_scales.data(), src_scale_memory);
    
    std::vector<float> weight_scales = {1.0f};
    auto weight_scale_md = memory::desc({1}, dt::f32, tag::x);
    auto weight_scale_memory = memory(src_scale_md, engine);
    write_to_dnnl_memory(weight_scales.data(), weight_scale_memory);

    std::vector<float> dst_scales = {2.0f};
    auto dst_scale_md = memory::desc({1}, dt::f32, tag::x);
    auto dst_scale_memory = memory(dst_scale_md, engine);
    write_to_dnnl_memory(dst_scales.data(), dst_scale_memory);

    std::unordered_map<int, memory> inner_product_args;
    inner_product_args.insert({DNNL_ARG_SRC, inner_product_src_mem});
    inner_product_args.insert({DNNL_ARG_WEIGHTS, inner_product_weights_mem});
    inner_product_args.insert({DNNL_ARG_BIAS, user_bias_mem});
    inner_product_args.insert({DNNL_ARG_DST, user_dst_mem});
    inner_product_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale_memory});
    inner_product_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, weight_scale_memory});
    inner_product_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scale_memory});

    // Primitive execution: inner-product with ReLU.
    inner_product_prim.execute(engine_stream, inner_product_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), user_dst_mem);

    std::cout << "Output: " << std::endl;
    for(int i = 0; i < dst_data.size(); i++) {
        std::cout << static_cast<int>(dst_data[i]) << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    return handle_example_errors(
            inner_product_example, parse_engine_kind(argc, argv));
}