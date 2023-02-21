// getting_started_cpp
/*
参考官网文档： https://oneapi-src.github.io/oneDNN/page_getting_started_cpp.html
这里把这个 demo 独立了出来，删除了不必要的注释，方便快速阅读

编译流程：
先编译 onednn 源码，得到头文件和库， 编译本 demo 的方式为： 

g++ -I ../thirdparty/oneDNN/include/ -I  ../thirdparty/oneDNN/build/include/  getting_started_cpp.cc -L /weishengying/oneDNN/build/src/ -ldnnl

注意修改链接文件路径。

执行：
export LD_LIBRARY_PATH=../thirdparty/oneDNN/build/src
./a.out
*/

/*
整体流程如下：
1. 创建一个 engine
2. 创建一个 stream
3. 创建输入和输出数据的 memory desc
4. 根据 memory desc， 创建 memory， onednn 会申请内存， 该内存 onednn 管理，无需用户管理。
5. 将用户数据拷贝到 input memory
6. 创建 op 的 primitive desc， 根据 primitive desc， 创建 op 的 primitive
7. 调用 op primitive 的 execute api， 传入 input memory 和 output memory
8. stream.wait
9. 将 output memory 的结果拷贝出来，检测结果
*/

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

#include "example_utils.hpp"
// using namespace dnnl;

void getting_started_tutorial(dnnl::engine::kind engine_kind) {
    // Initialize engine
    dnnl::engine eng(engine_kind, 0);

    // Initialize stream
    dnnl::stream engine_stream(eng);

    // Create user's data
    const int N = 1, H = 13, W = 13, C = 3;

    // An auxiliary function that maps logical index to the physical offset
    // format is NHWC
    auto offset = [=](int n, int h, int w, int c) {
        return n * H * W * C + h * W * C + w * C + c;
    };

    // The image size
    const int image_size = N * H * W * C;

    // Allocate a buffer for the image
    std::vector<float> image(image_size);

    // Initialize the image with some values
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < C; ++c) {
                    int off = offset(
                            n, h, w, c); // Get the physical offset of a pixel
                    image[off] = -std::cos(off / 10.f);
                }

    // Create user's data
    auto src_md = dnnl::memory::desc(
            {N, C, H, W,}, // logical dims, the order is defined by a primitive, 无论无 pysical order是什么，logical dims 顺序永远都是 NCHW。
            dnnl::memory::data_type::f32, // tensor's data type
            dnnl::memory::format_tag::nhwc // memory format, NHWC in this case。 （pysical order）
    );

    // for eltmemtwise, output desc is equal to input desc
    auto dst_md = src_md;

    // Create memory objects
    // src_mem contains a copy of image after write_to_dnnl_memory function
    auto src_mem = dnnl::memory(src_md, eng);
    write_to_dnnl_memory(image.data(), src_mem);

    // For dst_mem the library allocates buffer
    auto dst_mem = dnnl::memory(dst_md, eng);

    // Create a ReLU primitive
    // ReLU primitive descriptor, which corresponds to a particular
    // implementation in the library
    auto relu_pd = dnnl::eltwise_forward::primitive_desc(
            eng, // an engine the primitive will be created for
            dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_relu,
            src_md, // source memory descriptor for an operation to work on
            dst_md, // destination memory descriptor for an operation to work on
            0.f, // alpha parameter means negative slope in case of ReLU
            0.f // beta parameter is ignored in case of ReLU
    );

    // Create a ReLU primitive
    auto relu = dnnl::eltwise_forward(relu_pd); // !!! this can take quite some time
    
    // Execute ReLU primitive(out-of-place)
    relu.execute(engine_stream, // The execution stream
            {
                    // A map with all inputs and outputs
                    {DNNL_ARG_SRC, src_mem}, // Source tag and memory obj
                    {DNNL_ARG_DST, dst_mem}, // Destination tag and memory obj
            });

    // Wait the stream to complete the execution
    engine_stream.wait();
    

    // Check the results
    // Obtain a buffer for the `dst_mem` and cast it to `float *`.
    // This is safe since we created `dst_mem` as f32 tensor with known
    // memory format.
    std::vector<float> relu_image(image_size);
    read_from_dnnl_memory(relu_image.data(), dst_mem);
    // Check the results
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < C; ++c) {
                    int off = offset(
                            n, h, w, c); // get the physical offset of a pixel
                    float expected = image[off] < 0
                            ? 0.f
                            : image[off]; // expected value
                    if (relu_image[off] != expected) {
                        std::cout << "At index(" << n << ", " << c << ", " << h
                                  << ", " << w << ") expect " << expected
                                  << " but got " << relu_image[off]
                                  << std::endl;
                        throw std::logic_error("Accuracy check failed.");
                    }
                }
}


int main(int argc, char **argv) {
    int exit_code = 0;

    dnnl::engine::kind engine_kind = parse_engine_kind(argc, argv);
    try {
        getting_started_tutorial(engine_kind);
    } catch (dnnl::error &e) {
        std::cout << "oneDNN error caught: " << std::endl
                  << "\tStatus: " << dnnl_status2str(e.status) << std::endl
                  << "\tMessage: " << e.what() << std::endl;
        exit_code = 1;
    } catch (std::string &e) {
        std::cout << "Error in the example: " << e << "." << std::endl;
        exit_code = 2;
    }

    std::cout << "Example " << (exit_code ? "failed" : "passed") << " on "
              << engine_kind2str_upper(engine_kind) << "." << std::endl;
    return exit_code;
}
