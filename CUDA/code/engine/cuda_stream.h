#pragma once

#include "cuda_runtime.h"  
#include <string>
#include <map>
#include <assert.h>

class StreamPool {
 public:
  static StreamPool& Instance() {
    static StreamPool pool;
    return pool;
  }

  cudaStream_t GetStream(std::string name) {
    assert(streams_.count(name));
    return streams_.at(name);
  }

 private:
  StreamPool() {
    cudaStream_t compute_stream;
    cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
    streams_.insert({"compute_stream", compute_stream});

    cudaStream_t load_weight_stream;
    cudaStreamCreateWithFlags(&load_weight_stream, cudaStreamNonBlocking);
    streams_.insert({"load_weight_stream", load_weight_stream});
  }
  std::map<std::string, cudaStream_t>streams_;
};