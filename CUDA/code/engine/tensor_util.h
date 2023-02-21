#include "tensor.h"

void TensorCopySync(Tensor& dst, Tensor& src);

void TensorCopyASync(Tensor& dst, Tensor& src, cudaStream_t stream);

void DisplayTensor(Tensor& tensor);