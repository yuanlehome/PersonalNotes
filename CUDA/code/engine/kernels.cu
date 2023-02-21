template<typename T>
__global__ void vector_add(const T* x, const T * y, T* z, int n)
{
  // 获取该线程的全局索引
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  // 步长(线程总数)
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
  {
      z[i] = x[i] + y[i];
  }
  // printf("Run vector_add op end\n");
}