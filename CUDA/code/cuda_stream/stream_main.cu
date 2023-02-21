// 两个向量加法kernel，grid和block均为一维
__global__ void add(float* x, float * y, float* z, int n)
{
    // 获取该线程的全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长(线程总数)
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}

#include <iostream>
#include "cuda_runtime.h"  

void lod_weight(float* dst, float* src, int n, cudaStream_t stream) {
  cudaMemcpyAsync((void*)dst, (void*)src, n, cudaMemcpyHostToDevice, stream);
}

int main()
{
    int N = 1 << 20; //(2^20)
    int nBytes = N * sizeof(float); // 2^20 * 4 = 4 MiB 
    
    // 申请 Host 内存，存放输入、输出和 权重数据
    float *input, *output, *weight_0, *weight_1, *weight_2, *weight_3;
    input = (float*)malloc(nBytes);
    output = (float*)malloc(nBytes);
    weight_0 = (float*)malloc(nBytes);
    // weight_1 = (float*)malloc(nBytes);
    // weight_2 = (float*)malloc(nBytes);
    // weight_3 = (float*)malloc(nBytes);

    cudaMallocHost(&weight_1, nBytes);
    cudaMallocHost(&weight_2, nBytes);
    cudaMallocHost(&weight_3, nBytes);

    // 初始化输入数据和权重数据
    for (int i = 0; i < N; ++i){
      input[i] = 10.0;
      weight_0[i] = weight_1[i] = weight_2[i] = weight_3[i] = 20.0;
    }

    // 创建一个 CUDA kernel 计算流
    cudaStream_t compute_stream;
    cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);

    // 创建 event
    cudaEvent_t pre_compute_event;
    cudaEvent_t compute_event;
    cudaEvent_t load_event;
    cudaEventCreate(&pre_compute_event);
    cudaEventCreate(&compute_event);
    cudaEventCreate(&load_event);

    // 申请 GPU 显存
    float *d_input, *d_output, *d_weight_0, *d_weight_1;
    cudaMalloc((void**)&d_input, nBytes);
    cudaMalloc((void**)&d_output, nBytes);
    cudaMalloc((void**)&d_weight_0, nBytes);
    cudaMalloc((void**)&d_weight_1, nBytes);

    // 这里只将输入数据和 weight_0 数据从 Host 拷贝到 GPU, weight_1, weight_2, weight_3在执行时按需加载
    cudaMemcpy((void*)d_input, (void*)input, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_weight_0, (void*)weight_0, nBytes, cudaMemcpyHostToDevice);
    
    // 定义kernel的执行配置
    dim3 blockSize(1024);
    dim3 gridSize(16);

    // 执行 kernel 和 数据加载，并做好同步, kernel 在 计算流上， 数据加载在空流上
    add <<< gridSize, blockSize, 0, compute_stream >>>(d_input, d_weight_0, d_output, N);    
    cudaEventRecord(pre_compute_event, compute_stream);

    lod_weight(d_weight_1, weight_1, nBytes, NULL);
    cudaEventRecord(load_event, NULL);

    cudaStreamWaitEvent(compute_stream, load_event);
    add <<< gridSize, blockSize, 0, compute_stream >>>(d_output, d_weight_1, d_input, N);
    cudaEventRecord(compute_event, compute_stream);
    
    cudaStreamWaitEvent(NULL, pre_compute_event);
    lod_weight(d_weight_0, weight_2, nBytes, NULL);
    cudaEventRecord(load_event, NULL);

    cudaStreamWaitEvent(compute_stream, load_event);
    add <<< gridSize, blockSize, 0, compute_stream >>>(d_input, d_weight_0, d_output, N);

    cudaStreamWaitEvent(NULL, compute_event);
    lod_weight(d_weight_1, weight_3, nBytes, NULL);
    cudaEventRecord(load_event, NULL);

    cudaStreamWaitEvent(compute_stream, load_event);
    add <<< gridSize, blockSize, 0, compute_stream >>>(d_output, d_weight_1, d_input, N);
    
    // 流同步
    cudaStreamSynchronize(NULL);
    cudaStreamSynchronize(compute_stream);
    
    // 将device得到的结果拷贝到host(注意 最后的结果存在了 d_input 中)
    cudaMemcpy((void*)output, (void*)d_input, nBytes, cudaMemcpyDeviceToHost);
    
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(output[i] - 90.0));
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放 GPU 显存
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weight_0);
    cudaFree(d_weight_1);

    // 释放 Host 内存
    free(input);
    free(output);
    free(weight_0);
    // free(weight_1);
    // free(weight_2);
    // free(weight_3);
    cudaFreeHost(weight_1);
    cudaFreeHost(weight_2);
    cudaFreeHost(weight_3);

    // 释放cuda stream
    cudaStreamDestroy(compute_stream);
    
    // 释放 event
    cudaEventDestroy(compute_event);
    cudaEventDestroy(pre_compute_event);
    cudaEventDestroy(load_event);
    return 0;
}
