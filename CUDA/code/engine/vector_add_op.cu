#include "vector_add_op.h"
#include "kernels.cu"
#include "tensor.h"
#include "cuda_stream.h"
#include "tensor_util.h"

void VectorAddOp::RunImp(const Scope& scope) {
  Tensor* x = scope.GetVar(Inputs()[0])->GetMutable<Tensor>();
  Tensor* y = scope.GetVar(Inputs()[1])->GetMutable<Tensor>();
  Tensor* z = scope.GetVar(Outputs()[0])->GetMutable<Tensor>();

  assert(x->Shape() == y->Shape());
  z->set_shape(x->Shape());

  dim3 blockSize(2);
  dim3 gridSize(2);

  ENFORCE_NOT_NULL(x->data(), "data x should not be null");
  ENFORCE_NOT_NULL(y->data(), "data y should not be null");
  ENFORCE_NOT_NULL(z->mutable_data<float>(Place::KGPU), "data z should not be null");
  ENFORCE_EQ(x->numel(), y->numel(), "intput x and y should be equal.");

  assert(x->GetPlace() == Place::KGPU);
  assert(y->GetPlace() == Place::KGPU);

  // std::cout << "x.data： " << x->data() << std::endl;
  // std::cout << "y.data： " << y->data() << std::endl;
  // std::cout << "z.data： " << z->data() << std::endl;
  // DisplayTensor(*x);
  // DisplayTensor(*y);
  vector_add<float> <<< gridSize, blockSize, 0, StreamPool::Instance().GetStream("compute_stream") >>> (x->data<float>(), y->data<float>(), (float*)(z->data()), x->numel());
}
