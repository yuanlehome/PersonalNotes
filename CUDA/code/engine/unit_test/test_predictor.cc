#include "../predictor.h"
#include "../tensor_util.h"
#include "../vector_add_op.h"
#include "../cuda_event.h"

#include <cstring>

// init all the weight var in scope
void InitWeight(Scope* scope) {
  Tensor* weight_1_t = scope->GetVar("weight_1")->GetMutable<Tensor>();
  Tensor* weight_2_t = scope->GetVar("weight_2")->GetMutable<Tensor>();
  Tensor* weight_3_t = scope->GetVar("weight_3")->GetMutable<Tensor>();
  Tensor* weight_4_t = scope->GetVar("weight_4")->GetMutable<Tensor>();

  std::vector<float>weight_data(1*3*1024*1024, 1.0);
  weight_1_t->set_shape({1, 3, 1024, 1024});
  weight_2_t->set_shape({1, 3, 1024, 1024});
  weight_3_t->set_shape({1, 3, 1024, 1024});
  weight_4_t->set_shape({1, 3, 1024, 1024});

  auto set_vector_data = [&](int n) {
    for(int i = 0; i < weight_data.size(); i++)
      weight_data[i] = n;
  };

  float* data_1 = weight_1_t->mutable_data<float>(Place::KCPU);
  memcpy((void*)data_1, weight_data.data(), weight_data.size() * sizeof(float));

  float* data_2 = weight_2_t->mutable_data<float>(Place::KCPU);
  set_vector_data(2);
  memcpy((void*)data_2, weight_data.data(), weight_data.size() * sizeof(float));

  float* data_3 = weight_3_t->mutable_data<float>(Place::KCPU);
  set_vector_data(3);
  memcpy((void*)data_3, weight_data.data(), weight_data.size() * sizeof(float));

  float* data_4 = weight_4_t->mutable_data<float>(Place::KCPU);
  set_vector_data(4);
  memcpy((void*)data_4, weight_data.data(), weight_data.size() * sizeof(float));
}

void TransPartialWeightFromCpuToGpu(Scope* scope) {
  Tensor* weight_1 = scope->GetVar("weight_1")->GetMutable<Tensor>();
  Tensor* weight_2= scope->GetVar("weight_2")->GetMutable<Tensor>();
  Tensor* weight_3 = scope->GetVar("weight_3")->GetMutable<Tensor>();
  Tensor* weight_4= scope->GetVar("weight_4")->GetMutable<Tensor>();

  Tensor *weight_1_gpu = new Tensor();
  Tensor *weight_2_gpu = new Tensor();
  Tensor *weight_3_gpu = new Tensor();
  Tensor *weight_4_gpu = new Tensor();
  weight_1_gpu->set_shape(weight_1->Shape());
  weight_1_gpu->mutable_data<float>(Place::KGPU);
  weight_2_gpu->set_shape(weight_2->Shape());
  weight_2_gpu->mutable_data<float>(Place::KGPU);
  weight_3_gpu->set_shape(weight_3->Shape());
  weight_4_gpu->set_shape(weight_4->Shape());

  // only copy weigth_1 from cpu to gpu
  TensorCopySync(*weight_1_gpu, *weight_1);
  TensorCopySync(*weight_2_gpu, *weight_2);

  // creat new cpu tensor
  Tensor* new_weight_1 = scope->Var("weight_1_cpu")->GetMutable<Tensor>();
  Tensor* new_weight_2 = scope->Var("weight_2_cpu")->GetMutable<Tensor>();
  Tensor* new_weight_3 = scope->Var("weight_3_cpu")->GetMutable<Tensor>();
  Tensor* new_weight_4 = scope->Var("weight_4_cpu")->GetMutable<Tensor>();

  new_weight_1->set_shape(weight_1->Shape());
  new_weight_2->set_shape(weight_1->Shape());
  new_weight_3->set_shape(weight_3->Shape());
  new_weight_4->set_shape(weight_4->Shape());
  new_weight_1->mutable_data<float>(Place::KCPU);
  new_weight_2->mutable_data<float>(Place::KCPU);
  new_weight_3->mutable_data<float>(Place::KCPU);
  new_weight_4->mutable_data<float>(Place::KCPU);

  // copy all weights to new cpu tensor
  TensorCopySync(*new_weight_1, *weight_1);
  TensorCopySync(*new_weight_2, *weight_2);
  TensorCopySync(*new_weight_3, *weight_3);
  TensorCopySync(*new_weight_4, *weight_4);

  // clear all old cpu tensor
  weight_1->clear();
  weight_2->clear();
  weight_3->clear();
  weight_4->clear();

  scope->GetVar("weight_1")->Reset<Tensor>(weight_1_gpu);
  scope->GetVar("weight_2")->Reset<Tensor>(weight_2_gpu);
  scope->GetVar("weight_3")->Reset<Tensor>(weight_3_gpu);
  scope->GetVar("weight_4")->Reset<Tensor>(weight_4_gpu);
}

void TransAllWeightFromCpuToGpu(Scope* scope) {
  Tensor* weight_1_t = scope->GetVar("weight_1")->GetMutable<Tensor>();
  Tensor* weight_2_t = scope->GetVar("weight_2")->GetMutable<Tensor>();
  Tensor* weight_3_t = scope->GetVar("weight_3")->GetMutable<Tensor>();
  Tensor* weight_4_t = scope->GetVar("weight_4")->GetMutable<Tensor>();

  Tensor *weight_1_gpu_t = new Tensor();
  Tensor *weight_2_gpu_t = new Tensor();
  Tensor *weight_3_gpu_t = new Tensor();
  Tensor *weight_4_gpu_t = new Tensor();
  weight_1_gpu_t->set_shape(weight_1_t->Shape());
  weight_1_gpu_t->mutable_data<float>(Place::KGPU);
  weight_2_gpu_t->set_shape(weight_1_t->Shape());
  weight_2_gpu_t->mutable_data<float>(Place::KGPU);
  weight_3_gpu_t->set_shape(weight_1_t->Shape());
  weight_3_gpu_t->mutable_data<float>(Place::KGPU);
  weight_4_gpu_t->set_shape(weight_1_t->Shape());
  weight_4_gpu_t->mutable_data<float>(Place::KGPU);
  
  TensorCopySync(*weight_1_gpu_t, *weight_1_t);
  TensorCopySync(*weight_2_gpu_t, *weight_2_t);
  TensorCopySync(*weight_3_gpu_t, *weight_3_t);
  TensorCopySync(*weight_4_gpu_t, *weight_4_t);

  scope->GetVar("weight_1")->Reset<Tensor>(weight_1_gpu_t);
  scope->GetVar("weight_2")->Reset<Tensor>(weight_2_gpu_t);
  scope->GetVar("weight_3")->Reset<Tensor>(weight_3_gpu_t);
  scope->GetVar("weight_4")->Reset<Tensor>(weight_4_gpu_t);
}

int main() {
  VectorAddOp op_1("vector_add", {{"X", {"input"}}, {"Y", {"weight_1"}}}, {{"Out", {"output"}}}, {});
  VectorAddOp op_2("vector_add", {{"X", {"output"}}, {"Y", {"weight_2"}}}, {{"Out", {"input"}}}, {});
  VectorAddOp op_3("vector_add", {{"X", {"input"}}, {"Y", {"weight_3"}}}, {{"Out", {"output"}}}, {});
  VectorAddOp op_4("vector_add", {{"X", {"output"}}, {"Y", {"weight_4"}}}, {{"Out", {"input"}}}, {});

  Scope scope;
  Predictor predictor;
  predictor.AttachScope(&scope);
  predictor.AddOps({&op_1, &op_2, &op_3, &op_4});
  predictor.InitScope();
  predictor.InitCudaEvents();

  // init all the weight var in cpu
  InitWeight(&scope);

#ifdef LOAD_WEIGHT_ON_RUNTIME
  // transform all cpu weight to gpu, only transform weight_1
  TransPartialWeightFromCpuToGpu(&scope);
#else
  TransAllWeightFromCpuToGpu(&scope);
#endif

  auto *input_tensor = predictor.GetTensor("input");
  input_tensor->set_shape({1,3,1024,1024});
  std::vector<float>input_data(input_tensor->numel(), 1.0);
  input_tensor->CopyFromCpu<float>(input_data.data(), input_data.size());
  
  predictor.Run();
  auto* output_tensor = predictor.GetTensor("input");
  std::vector<float>output_data(output_tensor->numel());
  output_tensor->CopyToCpu(output_data);
  for(int i = 0; i < 10; i++)
    std::cout << output_data[i] << ", ";
  std::cout << std::endl;
}