
#pragma once

#include "cuda_runtime.h"  

#include <map>
#include <string>

// we should creat a event for each load_params operation in "load_params" stream
class CudaEventPool {
 public:
  static CudaEventPool& Instance() {
    static CudaEventPool event_pool;
    return event_pool;
  }

  void Init(std::vector<std::string>& params_names) {
    for(std::string name : params_names){
      cudaEvent_t event;
      cudaEventCreate(&event);
      pool[name] = event;
    }
  }
  
  void EventRecord(std::string& params_name, cudaStream_t stream) {
    assert(pool.count(params_name));
    
    cudaEventRecord(pool[params_name], stream);
  }

  void StreamWaitEvent(cudaStream_t stream, std::string& params_name) {
    assert(pool.count(params_name));
    cudaStreamWaitEvent(stream, pool[params_name]);
  }

 private:
  CudaEventPool() { }
  std::map<std::string, cudaEvent_t> pool;
};