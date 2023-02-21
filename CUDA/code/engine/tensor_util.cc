#include "tensor_util.h"
#include <cstring>

void TensorCopySync(Tensor& dst, Tensor& src) {
  assert(src.numel() == dst.numel());
  assert(src.data());
  assert(dst.data());
  if(src.GetPlace() == Place::KCPU && dst.GetPlace() == Place::KGPU)
    cudaMemcpy(dst.data(), src.data(), src.numel() * sizeof(float), cudaMemcpyHostToDevice);
  else if(src.GetPlace() == Place::KGPU && dst.GetPlace() == Place::KCPU)
    cudaMemcpy(dst.data(), src.data(), src.numel() * sizeof(float), cudaMemcpyDeviceToHost);
  else if(src.GetPlace() == Place::KCPU && dst.GetPlace() == Place::KCPU)
    memcpy(dst.data(), src.data(), src.numel() * sizeof(float));
  else 
    assert(false);
}

void TensorCopyASync(Tensor& dst, Tensor& src, cudaStream_t stream) {
  assert(src.numel() == dst.numel());
  assert(src.data());
  assert(dst.data());
  assert(src.GetPlace() == Place::KCPU);
  assert(dst.GetPlace() == Place::KGPU);
  cudaMemcpyAsync(dst.data(), src.data(), src.numel() * sizeof(float), cudaMemcpyHostToDevice, stream);
}

void DisplayTensor(Tensor& tensor) {
  if(tensor.GetPlace() == Place::KGPU) {
    Tensor tmp_cpu;
    tmp_cpu.set_shape(tensor.Shape());
    tmp_cpu.mutable_data<float>(Place::KCPU);
    TensorCopySync(tmp_cpu, tensor);
    for(int i = 0; i < 10; i++) {
      std::cout << tmp_cpu.data<float>()[i] << ", ";
    }
    std::cout << std::endl;
  } else {
    for(int i = 0; i < 10; i++) {
      std::cout << tensor.data<float>()[i] << ", ";
    }
    std::cout << std::endl;
  }
  
}