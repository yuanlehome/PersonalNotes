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

int main()
{
    int N = 1 << 20; //(2^20)
    int nBytes = N * sizeof(float); // 2^20 * 4 = 4 MiB 
    
    // 申请 Host 内存，存放输入、输出和权重数据
    float *input, *weight_0, *weight_1, *weight_2, *output;
    input = (float*)malloc(nBytes);
    output = (float*)malloc(nBytes);
    weight_0 = (float*)malloc(nBytes);
    weight_1 = (float*)malloc(nBytes);
    weight_2 = (float*)malloc(nBytes);

    // 初始化输入数据和权重数据
    for (int i = 0; i < N; ++i){
        input[i] = 10.0;
        weight_0[i] = weight_1[i] = weight_2[i] = 20.0;
    }

    // 申请 GPU 显存
    float *d_input, *d_output, *d_weight_0, *d_weight_1, *d_weight_2;
    cudaMalloc((void**)&d_input, nBytes);
    cudaMalloc((void**)&d_output, nBytes);
    cudaMalloc((void**)&d_weight_0, nBytes);
    cudaMalloc((void**)&d_weight_1, nBytes);
    cudaMalloc((void**)&d_weight_2, nBytes);

    // 将输入、权重数据从 Host 拷贝到 GPU
    cudaMemcpy((void*)d_input, (void*)input, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_weight_0, (void*)weight_0, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_weight_1, (void*)weight_1, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_weight_2, (void*)weight_2, nBytes, cudaMemcpyHostToDevice);

    // 定义kernel的执行配置
    dim3 blockSize(16);
    dim3 gridSize(16);

    // 执行kernel
    add << < gridSize, blockSize >> >(d_input, d_weight_0, d_output, N);
    add << < gridSize, blockSize >> >(d_output, d_weight_1, d_input, N);
    add << < gridSize, blockSize >> >(d_input, d_weight_2, d_output, N);

    //阻塞 Host端，直到流里的 CUDA 调用完成。
    cudaStreamSynchronize(NULL);  //等待默认流上的 kernel 全部执行完毕

    // 将device得到的结果拷贝到host
    cudaMemcpy((void*)output, (void*)d_output, nBytes, cudaMemcpyDeviceToHost);
    
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(output[i] - 70.0));
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放 GPU 显存
    cudaFree(d_input);
    cudaFree(d_weight_0);
    cudaFree(d_weight_1);
    cudaFree(d_weight_2);
    cudaFree(d_output);

    // 释放 Host 内存
    free(input);
    free(weight_0);
    free(weight_1);
    free(weight_2);
    free(output);

    return 0;
}
