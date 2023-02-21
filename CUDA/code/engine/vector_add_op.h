#pragma once

#include "operator.h"

class VectorAddOp : public OperatorBase {
 public:
  VectorAddOp(const std::string& type,
              const VariableNameMap& inputs,
              const VariableNameMap& outputs,
              const AttributeMap& attrs) : OperatorBase(type, inputs, outputs, attrs) { }
  
  void RunImp(const Scope& scope);

  std::vector<std::string> WeightInput() {
    return {Inputs()[1]};
  };

  bool HasWeightInput() { return true; }
};