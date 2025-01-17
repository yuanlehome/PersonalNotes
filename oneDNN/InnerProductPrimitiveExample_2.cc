/*******************************************************************************
参考 https://oneapi-src.github.io/oneDNN/page_inner_product_example_cpp.html#doxid-inner-product-example-cpp
编译方式: 
g++ -I ../thirdparty/install/onednn/include/  InnerProductPrimitiveExample.cc -L ../thirdparty/install/onednn/lib -ldnnl
export LD_LIBRARY_PATH=/weishengying/third_party/install/onednn/lib

相比如官方提供的demo做了一下修改：
根据官网文档提供的 Performance Tips：
Use dnnl::memory::format_tag::any for source, weights, and destinations memory format tags \
when create an inner product primitive to allow the library to choose the most appropriate memory format.

根据这个信息，我们在创建 inner_product_forward::primitive_desc 的时候， 输入、权重以及输出以及全部使用 format_tag::any
实际上，输出的 format_tag 总是 any （For the destination tensor the memory format is always dnnl::memory::format_tag::nc）
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
        return 1.1;
    });

    // Create memory descriptors and memory objects for src and weight. In this
    // example, NCHW and OIHW layout is assumed.
    auto user_src_md = memory::desc(src_dims, dt::s8, tag::nchw);
    auto user_bias_md = memory::desc(bias_dims, dt::f32, tag::a);
    auto user_src_mem = memory(user_src_md, engine);
    auto user_bias_mem = memory(user_bias_md, engine);
    auto user_weights_mem = memory({weights_dims, dt::s8, tag::oihw}, engine);

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
    auto inner_product_dst_md = memory::desc(dst_dims, dt::s8, tag::nc);

    // Create inner product primitive descriptor.
    auto inner_product_pd = inner_product_forward::primitive_desc(engine,
            prop_kind::forward_training, inner_product_src_md, inner_product_weights_md,
            user_bias_md, inner_product_dst_md);
   
    // Create memory object for output(dst)
    auto inner_product_dst_mem = memory(inner_product_dst_md, engine);

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
    auto src_scale_md = memory::desc({1}, dt::f32, tag::x);
    auto src_scale_memory = memory(src_scale_md, eng);
    write_to_dnnl_memory(src_scales.data(), src_scale_memory);

    std::unordered_map<int, memory> inner_product_args;
    inner_product_args.insert({DNNL_ARG_SRC, inner_product_src_mem});
    inner_product_args.insert({DNNL_ARG_WEIGHTS, inner_product_weights_mem});
    inner_product_args.insert({DNNL_ARG_BIAS, user_bias_mem});
    inner_product_args.insert({DNNL_ARG_DST, inner_product_dst_mem});
    inner_product_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale_memory});

    // Primitive execution: inner-product with ReLU.
    inner_product_prim.execute(engine_stream, inner_product_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // auto user_dst_md = memory::desc(dst_dims, memory::data_type::s32, tag::nc);
    // auto user_dst_mem = memory(user_dst_md, engine);

    // std::cout << int(inner_product_pd.dst_desc().get_data_type()) << std::endl;
    // std::cout << int(user_dst_md.get_data_type()) << std::endl;
    // // convert dst_mem


    // if (inner_product_pd.dst_desc() != user_dst_mem.get_desc()) {
    //   std::cout << "reorder dst" << std::endl;
    //   primitive_attr dst_attr;
    //   int dst_mask = 0;

    //   // Prepare dst scales
    //   auto dst_scale_md = memory::desc({1}, dt::f32, tag::x);
    //   auto dst_scale_memory = memory(dst_scale_md, engine);
    //   std::vector<float> dst_scales = {1.1f};
    //   write_to_dnnl_memory(dst_scales.data(), dst_scale_memory);

    //   dst_attr.set_scales_mask(DNNL_ARG_SRC, dst_mask);
    //   auto dst_reorder_pd
    //           = reorder::primitive_desc(engine, inner_product_dst_mem.get_desc(), engine,
    //                   user_dst_mem.get_desc(), dst_attr);
    //   auto dst_reorder = reorder(dst_reorder_pd);
    //   dst_reorder.execute(engine_stream,
    //           {{DNNL_ARG_FROM, inner_product_dst_mem}, {DNNL_ARG_TO, user_dst_mem},
    //                   {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dst_scale_memory}});
    // }
    
    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), inner_product_dst_mem);

    std::cout << "Output: " << dst_data.size() << std::endl;
    for(int i = 0; i < dst_data.size(); i++) {
        uint8_t *src = static_cast<uint8_t *>(inner_product_dst_mem.get_data_handle());
        
        std::cout << src[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    return handle_example_errors(
            inner_product_example, parse_engine_kind(argc, argv));
}