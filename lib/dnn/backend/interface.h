#pragma once
#include "dnn/context.h"
#include "dnn/node.h"

namespace dnn {
namespace backend {

using ::dnn::Add;

template <typename NodeTy, DeviceTy Device>
void forward(NodeTy& add);

template <>
void forward<Add, DeviceTy::Generic>(Add& add)
{
  auto* A = add.input0()->data();
  auto* B = add.input1()->data();
  auto* C = add.data();
  for (size_t i = 0; i < add.elems(); i++) { C[i] = A[i] + B[i]; }
}

} // namespace backend
} // namespace dnn