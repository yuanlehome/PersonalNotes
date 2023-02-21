#pragma once

#include <list>

#ifdef LOAD_WEIGHT_ON_RUNTIME
#include <thread> 
#include <mutex>
#include <condition_variable>
#endif

#include "tensor.h"
#include "operator.h"
#include "scope.h"
#include "tensor_util.h"

#include "cuda_runtime.h"  
#include "cuda_stream.h"
#include "cuda_event.h"

#ifdef LOAD_WEIGHT_ON_RUNTIME
class Predictor;
struct CallBackInfo {
  Tensor* used_weight_gpu_tensor_;
  Tensor* next_weight_cpu_tensor_;
  Tensor* next_weight_gpu_tensor_;
  std::string params_name_;
};    

struct CallBack {
 public:

  static void call_back_imp(CallBackInfo* info) {
    //clear used weight
    info->used_weight_gpu_tensor_->clear();
    // note this fuction cant't alloc new gpu memory, Only idle allocated gpu memory is allowed
    info->next_weight_gpu_tensor_->mutable_data<float>(Place::KGPU);

    std::thread load_thread(LoadNextWeight, info->params_name_, info->next_weight_gpu_tensor_, info->next_weight_cpu_tensor_);
    load_thread.join();

    cv.notify_all();
  }

  static void my_callback(cudaStream_t stream, cudaError_t status, void *data){
    std::unique_lock<std::mutex> lck(mtx);
    // std::cout << "call back\n";

    CallBackInfo* info = reinterpret_cast<CallBackInfo*>(data);
    //clear used weight
    info->used_weight_gpu_tensor_->clear();
    // note this fuction cant't alloc new gpu memory, Only idle allocated gpu memory is allowed
    info->next_weight_gpu_tensor_->mutable_data<float>(Place::KGPU);

    std::thread load_thread(LoadNextWeight, info->params_name_, info->next_weight_gpu_tensor_, info->next_weight_cpu_tensor_);
    load_thread.join();

    cv.notify_all();
    // std::cout << "call back end\n";
    // done = true;
  }

  static void LoadNextWeight(std::string params_name, Tensor *dst, Tensor *src) {
    TensorCopyASync(*dst, *src, StreamPool::Instance().GetStream("load_weight_stream"));
    CudaEventPool::Instance().EventRecord(params_name, StreamPool::Instance().GetStream("load_weight_stream"));
    // std::cout <<  cudaGetErrorString( cudaGetLastError() ) << std::endl;
  }

  static void* data;
  static std::list<std::string> weight_queue;
  static std::mutex mtx;
  static std::condition_variable cv;
  static bool done;
};

std::list<std::string> CallBack::weight_queue{"weight_2", "weight_3", "weight_4"};
void* CallBack::data = nullptr;
std::mutex CallBack::mtx;
std::condition_variable CallBack::cv;
bool CallBack::done = false;
#endif

class Predictor {
 public:
  void InitScope() {
    for(auto *op : ops_) {
      for(auto &input : op->Inputs()) {
        ENFORCE_NOT_NULL(scope_, "scope_ should not be null!");
        scope_->Var(input);
      }
      for(auto &output : op->Outputs()) {
        ENFORCE_NOT_NULL(scope_, "scope_ should not be null!");
        scope_->Var(output);
      }
    }
  }

  void InitCudaEvents() {
    std::vector<std::string>all_params_name;
    for(auto *op : ops_) {
      for(auto & params_name : op->WeightInput()) {
        all_params_name.push_back(params_name);
      }
    }
    CudaEventPool::Instance().Init(all_params_name);
  }

  void Run() {
    for(auto *op : ops_) {
      op->Run(*scope_);
#ifdef LOAD_WEIGHT_ON_RUNTIME
        if(op->HasWeightInput() && weight_queue.size()) {
          Variable* used_weight_var = scope_->GetVar(op->WeightInput()[0]);
          Tensor* used_weight_tensor = used_weight_var->GetMutable<Tensor>();
          info_.used_weight_gpu_tensor_ = used_weight_tensor;

          std::string next_weight_var_name = weight_queue.front();
          info_.params_name_ = next_weight_var_name;
          // std::cout << "next_weight_var_name: " << next_weight_var_name << "\n";
          weight_queue.pop_front();

          Variable* next_weight_var = scope_->GetVar(next_weight_var_name);
          Tensor* next_weight_gpu_tensor_ = next_weight_var->GetMutable<Tensor>();
          info_.next_weight_gpu_tensor_ = next_weight_gpu_tensor_;

          Tensor* next_weight_cpu_tensor_ = scope_->GetVar(next_weight_var_name + "_cpu")->GetMutable<Tensor>();
          info_.next_weight_cpu_tensor_ = next_weight_cpu_tensor_;
          cudaStreamAddCallback(StreamPool::Instance().GetStream("compute_stream"), CallBack::my_callback, (void *)(&info_), 0);
          std::unique_lock<std::mutex> lck(CallBack::mtx);
          CallBack::cv.wait(lck);
          // CallBack::ready = false;
        }
      // std::cout << "new Op run\n";
#endif
    }
    cudaStreamSynchronize(StreamPool::Instance().GetStream("compute_stream"));
  }

  void AttachScope(Scope* const scope) {
    scope_ = scope;
  }

  void AddOp(OperatorBase* op) {
    ops_.push_back(op);
  }

  void AddOps(const std::vector<OperatorBase*> ops) {
    ops_ = std::move(ops);
  }

  Tensor* GetTensor(std::string tensor_name) {
    Variable* var = scope_->FindVar(tensor_name);
    ENFORCE_NOT_NULL(var, 
                    "var %s is null", tensor_name.c_str());
    Tensor* tensor = var->GetMutable<Tensor>();
    ENFORCE_NOT_NULL(tensor, 
                    "tensor %s is null", tensor_name.c_str());
    return tensor;
  }
  
 private:
  std::list<std::string>weight_queue{"weight_3", "weight_4"};
  std::vector<OperatorBase*> ops_;
  Scope* scope_;
#ifdef LOAD_WEIGHT_ON_RUNTIME
  CallBackInfo info_;
#endif
};