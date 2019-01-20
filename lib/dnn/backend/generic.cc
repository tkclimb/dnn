#include "dnn/backend/interface.h"
#include "dnn/node.h"

namespace dnn {
namespace backend {

template <>
void forward<Add, DeviceTy::Generic>(Add& add)
{
  auto* A = add.input0()->data();
  auto* B = add.input1()->data();
  auto* C = add.data();
  for (Index i = 0; i < add.elems(); ++i) { C[i] = A[i] + B[i]; }
}

} // namespace backend
} // namespace dnn