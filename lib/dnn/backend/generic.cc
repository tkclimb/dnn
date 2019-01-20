#include "dnn/backend/interface.h"
#include "dnn/node.h"

namespace dnn {

template <>
void Backend::forward<Placeholder, DeviceTy::Generic>(Placeholder& node)
{
  node.tensor_.copy_from_vec(*(node.data_));
}

template <>
void Backend::forward<Add, DeviceTy::Generic>(Add& node)
{
  auto* A = node.input0()->data();
  auto* B = node.input1()->data();
  auto* C = node.data();
  for (Index i = 0; i < node.elems(); ++i) { C[i] = A[i] + B[i]; }
}

template <>
void Backend::forward<Sub, DeviceTy::Generic>(Sub& node)
{
  auto* A = node.input0()->data();
  auto* B = node.input1()->data();
  auto* C = node.data();
  for (Index i = 0; i < node.elems(); ++i) { C[i] = A[i] - B[i]; }
}

} // namespace dnn