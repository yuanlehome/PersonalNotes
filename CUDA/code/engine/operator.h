#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <unordered_map>

#include "variable.h"
#include "variant.h"
#include "scope.h"
#include "cuda_event.h"
#include "cuda_stream.h"

using VariableNameMap = std::map<std::string, std::vector<std::string>>;
using VariableValueMap = std::map<std::string, std::vector<Variable*>>;
using Attribute = paddle::variant<int,
                                  float,
                                  std::string,
                                  std::vector<int>,
                                  std::vector<float>,
                                  std::vector<std::string>,
                                  bool,
                                  std::vector<bool>,
                                  int64_t,
                                  std::vector<int64_t>,
                                  std::vector<double>,
                                  double>;

using AttributeMap = std::unordered_map<std::string, Attribute>;

class OperatorBase {
 public:
  OperatorBase(const std::string& type,
               const VariableNameMap& inputs,
               const VariableNameMap& outputs,
               const AttributeMap& attrs) : type_(type), inputs_(inputs), outputs_(outputs), attrs_(attrs) { }
  
  void Run(const Scope& scope) {
#ifdef LOAD_WEIGHT_ON_RUNTIME
    if(HasWeightInput()) {
      // std::cout << "StreamWaitEvent " << WeightInput()[0] << "\n";
      // CudaEventPool::Instance().StreamWaitEvent(StreamPool::Instance().GetStream("compute_stream"), WeightInput()[0]);
    }
#endif
    // std::cout << "Run" << std::endl;
    RunImp(scope);
  }

  virtual void RunImp(const Scope& scope) = 0;

  virtual bool HasWeightInput() = 0;

  std::vector<std::string> Inputs() const {
    std::vector<std::string>res;
    for(auto& input : inputs_) {
      res.insert(res.end(), input.second.begin(), input.second.end());
    }
    return res;
  }

  virtual std::vector<std::string> WeightInput() = 0;

  std::vector<std::string> Outputs() const {
    std::vector<std::string>res;
    for(auto& output : outputs_) {
      res.insert(res.end(), output.second.begin(), output.second.end());
    }
    return res;
  }

  std::string Type() {
    return type_;
  }
 protected:
  std::string type_;
  VariableNameMap inputs_;
  VariableNameMap outputs_;
  AttributeMap attrs_;
};
