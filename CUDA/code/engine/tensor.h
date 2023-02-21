#pragma once

#include <vector>
#include <assert.h>
#include <typeindex>

#include "cuda_runtime.h"  
#include "memory_pool.h"

using shape = std::vector<int>;
struct Allocation;

class Tensor {
 public:
  template<typename T>
  const T* data() const {
    assert(allocation_);
    return static_cast<const T*>(allocation_->data_);
  }

  void* data() {
    if(allocation_) return allocation_->data_;
    return nullptr;
  }

  template<typename T>
  T* mutable_data(const Place& place) {
    if(data() && numel() * sizeof(T) <= capacity() && place == allocation_->place_) return static_cast<T*>(allocation_->data_);
    if(place == Place::KGPU)
      allocation_ = MemoryPool::Instance(Place::KGPU).Allocate(numel() * sizeof(T));
    else 
      allocation_ = MemoryPool::Instance(Place::KCPU).Allocate(numel() * sizeof(T));
    assert(allocation_);
    assert(allocation_->data_);
    return static_cast<T*>(allocation_->data_);
  }
  
  void clear() {
    assert(allocation_);
    allocation_->occupied_ = false;
  }

  const shape& set_shape(const shape& shape) {
    shape_ = shape;
    return shape_;
  }

  template <typename T>
  void CopyFromCpu(const T *data, int ele_size) {
    // Only float data in considered
    assert(std::type_index(typeid(T)) == std::type_index(typeid(float)));
    assert(ele_size == numel());
    mutable_data<T>(Place::KGPU);
    assert(allocation_->data_);
    cudaMemcpy((void*)allocation_->data_, (void*)data, ele_size * sizeof(T), cudaMemcpyHostToDevice);
  }

  template <typename T>
  void CopyToCpu(std::vector<T>& output) {
    // Only float data in considered
    assert(std::type_index(typeid(T)) == std::type_index(typeid(float)));
    assert(numel() == output.size());
    assert(allocation_->data_);
    assert(allocation_->place_ == Place::KGPU);
    cudaMemcpy((void*)output.data(), (void*)(allocation_->data_), numel() * sizeof(T), cudaMemcpyDeviceToHost);
  }

  const shape& Shape() {
    return shape_;
  }
  
  int numel() {
    int res = 1;
    for(int i : shape_) res *= i;
    return res;
  }
  
  int capacity() {
    if(allocation_) return allocation_->byte_size_;
    return 0;
  }

  Place GetPlace() {
    assert(allocation_);
    return allocation_->place_;
  }

 private:
  shape shape_;
  DataType data_type_ = DataType::Kfloat;
  LayOut lay_out_ = LayOut::KNCHW;
  Allocation *allocation_ = nullptr;
};