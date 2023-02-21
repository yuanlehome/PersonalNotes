#include "../vector_add_op.h"
#include "../scope.h"
#include "../tensor.h"

#include <string>

int main() {
  VectorAddOp op_1("vector_add", {{"X", {"input_1"}}, {"Y", {"weight_1"}}}, {{"Out", {"tmp1"}}}, {});
  Scope scope;
  Variable* i_var = scope.Var("input_1");
  Variable* w_var = scope.Var("weight_1");
  Variable* o_var = scope.Var("tmp1");

  Tensor* tensor = i_var->GetMutable<Tensor>();
  tensor->set_shape({1,3,24,24});
  std::vector<float>input_data(tensor->numel(), 1.1);
  tensor->CopyFromCpu<float>(input_data.data(), input_data.size());

  Tensor* w_tensor = w_var->GetMutable<Tensor>();
  w_tensor->set_shape({1,3,24,24});
  std::vector<float>weight_1(w_tensor->numel(), 1.1);
  w_tensor->CopyFromCpu<float>(weight_1.data(), weight_1.size());

  op_1.Run(scope);
  Tensor* o_tensor = o_var->GetMutable<Tensor>();
  std::vector<float>output(o_tensor->numel());
  o_tensor->CopyToCpu(output);
  for(int i = 0; i < output.size(); i++)
    std::cout << output[i] << ", ";
  std::cout << std::endl;
}