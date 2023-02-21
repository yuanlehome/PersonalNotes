/*
参考官网文档： https://oneapi-src.github.io/oneDNN/page_getting_started_cpp.html
这里把这个 demo 独立了出来，删除了不必要的注释，方便快速阅读

编译流程：
先编译 onednn 源码，得到头文件和库， 编译本 demo 的方式为： 

g++ -I ../thirdparty/oneDNN/include/ -I  ../thirdparty/oneDNN/build/include/  memory_format_propagation.cc -L /weishengying/oneDNN/build/src/ -ldnnl

注意修改链接文件路径。

执行：
export LD_LIBRARY_PATH=../thirdparty/oneDNN/build/src
ONEDNN_VERBOSE=1 ./a.out
*/

/*
这个 demo 和官网demo 有些不同，
对于 conv op， 输入和输出都使用 any， 对于 pool ，输入等于 conv 的输出， 输出也用 any(官方的demo输出等于输入, 实际情况 pool 的输出 format 就等于输入 format， 即 pool primitive 不对改变输入的 format)，我觉得按照文档的推荐，pool 的输出设置为 any 更合理。
因为 pool 设置为 any，他的输出由 pool primitive 自己推导，如果设置为和输入相同，实际运行得到的 format 可能不同，这样就需要多做一次 reorder（实际并不会，正如前面所说， pool primitive 的输出 format 就等于输入 format）。

那 pool 的输入 format 能否设置为 any 呢，不同，会报错（亲测）。 应该是设置为 any， 但 pool 无法向 conv primitive 那样根据硬件，参数等自己选择 format，所以会报错
*/

#include <iostream>
#include <sstream>
#include <string>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

void memory_format_propagation_tutorial(engine::kind engine_kind) {
    // Initialize engine and stream
    engine eng(engine_kind, 0);
    stream s(eng);

    // Create placeholder memory descriptors
    const int N = 1, H = 14, W = 14, IC = 128, OC = 256, KH = 3, KW = 3;
    auto conv_src_md = memory::desc({N, IC, H, W}, memory::data_type::f32,
            memory::format_tag::any // let convolution choose memory format
    );
    auto conv_weights_md = memory::desc(
            {OC, IC, KH, KW}, memory::data_type::f32,
            memory::format_tag::any // let convolution choose memory format
    );
    auto conv_dst_md = memory::desc({N, OC, H, W}, memory::data_type::f32,
            memory::format_tag::any // let convolution choose memory format
    );

    // Create convolution and pooling primitive descriptors
    auto conv_pd = convolution_forward::primitive_desc(
            eng, prop_kind::forward_inference, algorithm::convolution_auto,
            conv_src_md, conv_weights_md,
            conv_dst_md, // shape information
            {1, 1}, // strides
            {1, 1}, {1, 1} // left and right padding
    );
    
    const auto &pool_src_md = conv_pd.dst_desc();
    const auto &pool_dst_md = memory::desc({N, OC, H, W}, memory::data_type::f32,
            memory::format_tag::any // let pool choose memory format
    );

    auto pool_pd= pooling_forward::primitive_desc(eng, prop_kind::forward_inference,
                algorithm::pooling_max, pool_src_md,
                pool_dst_md, // shape information
                {1, 1}, {KH, KW}, // strides and kernel
                {0, 0}, // dilation
                {1, 1}, {1, 1} // left and right padding
    );

    // Create source and destination memory objects
    auto src_mem = memory(
            {{N, IC, H, W}, memory::data_type::f32, memory::format_tag::nchw},
            eng);
    auto weights_mem = memory({{OC, IC, KH, KW}, memory::data_type::f32,
                                      memory::format_tag::oihw},
            eng);
    auto dst_mem = memory(
            {{N, OC, H, W}, memory::data_type::f32, memory::format_tag::nchw},
            eng);

    // Determine if source needs to be reordered
    bool need_reorder_src = conv_pd.src_desc() != src_mem.get_desc();

    // Determine if weights and destination need to be reordered
    bool need_reorder_weights
            = conv_pd.weights_desc() != weights_mem.get_desc();

    bool need_reorder_dst = pool_pd.dst_desc() != dst_mem.get_desc();

    // Allocate intermediate buffers if necessary
    auto conv_src_mem
            = need_reorder_src ? memory(conv_pd.src_desc(), eng) : src_mem;
    auto conv_weights_mem = need_reorder_weights
            ? memory(conv_pd.weights_desc(), eng)
            : weights_mem;
    auto conv_dst_mem = memory(conv_pd.dst_desc(), eng);
    auto pool_dst_mem
            = need_reorder_dst ? memory(pool_pd.dst_desc(), eng) : dst_mem;

    std::cout <<  need_reorder_src << " " << need_reorder_weights << " " << need_reorder_dst << "\n";
    // Perform reorders for source data if necessary
    if (need_reorder_src) {
        auto reorder_src = reorder(src_mem, conv_src_mem);
        reorder_src.execute(
                s, {{DNNL_ARG_FROM, src_mem}, {DNNL_ARG_TO, conv_src_mem}});
        s.wait(); // wait for the reorder to complete
    }

    if (need_reorder_weights) {
        auto reorder_weights = reorder(weights_mem, conv_weights_mem);
        reorder_weights.execute(s,
                {{DNNL_ARG_FROM, weights_mem},
                        {DNNL_ARG_TO, conv_weights_mem}});
        s.wait(); // wait for the reorder to complete
    }

    // Create and execute convolution and pooling primitives
    auto conv = convolution_forward(conv_pd);
    conv.execute(s,
            {{DNNL_ARG_SRC, conv_src_mem}, {DNNL_ARG_WEIGHTS, conv_weights_mem},
                    {DNNL_ARG_DST, conv_dst_mem}});
    auto pool = pooling_forward(pool_pd);
    pool.execute(
            s, {{DNNL_ARG_SRC, conv_dst_mem}, {DNNL_ARG_DST, pool_dst_mem}});
    s.wait();

    // Reorder destination data if necessary
    if (need_reorder_dst) {
        auto reorder_dst = reorder(pool_dst_mem, dst_mem);
        reorder_dst.execute(
                s, {{DNNL_ARG_FROM, pool_dst_mem}, {DNNL_ARG_TO, dst_mem}});
        s.wait();
    }
}

int main(int argc, char **argv) {
    return handle_example_errors(
            memory_format_propagation_tutorial, parse_engine_kind(argc, argv));
}