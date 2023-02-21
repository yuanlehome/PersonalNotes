#include "memory_pool.h"  
#include <assert.h>

Allocation* CudaMemoryPool::Allocate(int byte_size) {
  // find free Allocation
  for(auto &i : memory_nodes) {
    if(!i.occupied_ && i.byte_size_ >= byte_size) {
      i.occupied_ = true;
      return &i;
    }
  }
  // alloc new Allocation
  Allocation alloc;
  alloc.byte_size_ = byte_size;
  cudaMalloc((void**)&(alloc.data_), byte_size);
  assert(alloc.data_);
  alloc.occupied_ = true;
  alloc.place_ = Place::KGPU;
  memory_nodes.push_back(alloc);
  std::cout << "CudaMemoryPool memory_nodes " << memory_nodes.size() << std::endl;
  assert(memory_nodes.back().data_);
  return &(memory_nodes.back());
}

Allocation* CPUMemoryPool::Allocate(int byte_size) { 
  // always alloc new Allocation
  Allocation alloc;
  alloc.byte_size_ = byte_size;
  cudaMallocHost(&alloc.data_, byte_size);
  alloc.occupied_ = true;
  alloc.place_ = Place::KCPU;
  memory_nodes.insert(memory_nodes.end(), alloc);
  // std::cout << "CPUMemoryPool memory_nodes " << memory_nodes.size() << std::endl;
  return &(memory_nodes.back());
}

MemoryPool& MemoryPool::Instance(const Place& place_type) {
  if(place_type == Place::KGPU) {
    static CudaMemoryPool pool;
    return pool;
  }else {
    static CPUMemoryPool pool;
    return pool;
  }
}