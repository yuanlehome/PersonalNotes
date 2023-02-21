#pragma once

#include <list>
#include <iostream>

#include "cuda_runtime.h"  

enum DataType : int {
  Kfloat = 0,
};

enum LayOut : int {
  KNCHW = 0,
};

enum Place : int {
  KCPU = 0,
  KGPU = 1,
};

struct Allocation {
  Allocation(void* data = nullptr, int byte_size = 0) : data_(data), byte_size_(byte_size) { }
  void* data_ = nullptr;
  int byte_size_ = 0;
  bool occupied_ = false;
  Place place_ = Place::KCPU;
};

class CudaMemoryPool;
class CPUMemoryPool;

class MemoryPool{
 public:
  static MemoryPool& Instance(const Place& place_type);

  virtual Allocation* Allocate(int byte_size) = 0 ;

  virtual ~MemoryPool() { }
};

class CPUMemoryPool : public MemoryPool {
 public:
  Allocation* Allocate(int byte_size);

  virtual ~CPUMemoryPool() {
    for(auto &node : memory_nodes) {
      cudaFree(node.data_);
    }
  }
 private:
  std::list<Allocation>memory_nodes;
};

class CudaMemoryPool : public MemoryPool {
 public:
  Allocation* Allocate(int byte_size);

  virtual ~CudaMemoryPool() {
    for(auto &node : memory_nodes) {
      cudaFreeHost(node.data_);
    }
 }

 private:
  std::list<Allocation>memory_nodes;
};
