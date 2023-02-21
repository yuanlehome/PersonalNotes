/*
参考官网文档： https://oneapi-src.github.io/oneDNN/page_memory_format_propagation_cpp.html
整体流程参考：https://oneapi-src.github.io/oneDNN/dev_guide_inference.html#fp32-inference

编译流程：
先编译 onednn 源码，得到头文件和库， 编译本 demo 的方式为： 

g++ -I ../thirdparty/oneDNN/include/ -I  ../thirdparty/oneDNN/build/include/  memory_format_propagation_1.cc -L /weishengying/oneDNN/build/src/ -ldnnl

注意修改链接文件路径。

执行：
export LD_LIBRARY_PATH=../thirdparty/oneDNN/build/src
ONEDNN_VERBOSE=1 ./a.out
*/


/*
与官网的 demo 相比，这里只使用一个conv primitive。

在创建 conv primitive desc 的时候，需要指定期望的 input、bias memory desc, output 的 memory dec 一般使用 any 占位符， 这样 output 的 memory desc 由 op primitive 自动推导得到。
实际上 op 的 primitive 也可以指定期望得到的 output 的 memory desc，这样 primitive 在计算完后，其内部会自己做 reorder，保证输出的 memory desc 符合期望。
总而言之，在创建 primitive desc 的时候，可以指定输入的 memery desc以及 primitive 最后得到输出的 memory desc


但是实际输入的 input， bias 与期望可能不同。 这时候就需要 reorder。

根据官网文档建议，https://oneapi-src.github.io/oneDNN/page_memory_format_propagation_cpp.html。
对于 conv 这类 op， 输入输出最好使用 dnnl::memory::format_tag::any 占位符，这样 conv primitive 内部自己会根据硬件，参数等信息自己选择最合适的 memory format，最有利于运算的加速。

以本 demo 为例, 实际用户输入的 format 为：nchw，oihw， 期望结果的 format 为 nchw；

如果创建 primitive_desc 的时候，指定的输入和输出也都是 nchw，oihw，nchw，这样运算过程中不需要任何 reorder。

如果创建 primitive_desc 的时候，输入和输出都使用 any 占位符，实际过程中就需要 reorder。因为 conv primitive 内存选择了更有利于计算的 memory format。

实际过程中，如果只指定 input 是 nchw，bias 和 output 都是 any，这样 conv primitive 内部其实选择的 memory format 就都是 nchw，效果和同时指定输入输出都是 nchw 一样。

总之，为了计算的加速，对于 conv primitive，我们总是对其输入和输出使用 any 占位符，不指定具体的 format， 让其内部自己选择, 以使性能最优化。

*/

/*
最后，在执行的时候，可以执行命令 ONEDNN_VERBOSE=1 ./a.out ，查看format 转换的情况：

1 1 1
onednn_verbose,info,oneDNN v3.0.0 (commit b4067dcd853adba5be40324b1bc5992dd91ccc3c)
onednn_verbose,info,cpu,runtime:OpenMP,nthr:96
onednn_verbose,info,cpu,isa:Intel AVX-512 with Intel DL Boost
onednn_verbose,info,gpu,runtime:none
onednn_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:acdb:f0,,,1x128x14x14,48.0439
onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:Acdb32a:f0,,,256x128x3x3,25.3081
onednn_verbose,exec,cpu,convolution,brgconv:avx512_core,forward_inference,src_f32::blocked:acdb:f0 wei_f32::blocked:Acdb32a:f0 bia_undef::undef:0:f0 dst_f32::blocked:acdb:f0,,alg:convolution_direct,mb1_ic128oc256_ih14oh14kh3sh1dh0ph1_iw14ow14kw3sw1dw0pw1,26.4592
onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:acdb:f0 dst_f32::blocked:abcd:f0,,,1x256x14x14,10.0461
Example passed on CPU.

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

    // Create convolution 
    auto conv_pd = convolution_forward::primitive_desc(
            eng, prop_kind::forward_inference, algorithm::convolution_auto,
            conv_src_md, conv_weights_md,
            conv_dst_md, // shape information
            {1, 1}, // strides
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
    bool need_reorder_weights = conv_pd.weights_desc() != weights_mem.get_desc();

    bool need_reorder_dst = conv_pd.dst_desc() != dst_mem.get_desc();

    // Allocate intermediate buffers if necessary
    auto conv_src_mem = need_reorder_src ? memory(conv_pd.src_desc(), eng) : src_mem;
    auto conv_weights_mem = need_reorder_weights
            ? memory(conv_pd.weights_desc(), eng)
            : weights_mem;
    auto conv_dst_mem = need_reorder_dst ? memory(conv_pd.dst_desc(), eng) : dst_mem;

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
    s.wait();

    // Reorder destination data if necessary
    if (need_reorder_dst) {
        auto reorder_dst = reorder(conv_dst_mem, dst_mem);
        reorder_dst.execute(
                s, {{DNNL_ARG_FROM, conv_dst_mem}, {DNNL_ARG_TO, dst_mem}});
        s.wait();
    }
}

int main(int argc, char **argv) {
    return handle_example_errors(
            memory_format_propagation_tutorial, parse_engine_kind(argc, argv));
}