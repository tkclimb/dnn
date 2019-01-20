#include "dnn/backend/generic.h"
#include "dnn/node.h"
#include "dnn/type.h"

namespace dnn {
namespace backend {
namespace X86 {

/*
template <>
void Backend::forward<Add, DeviceTy::Generic>(Add& node)
{
  // auto* A = node.a()->data();
  // auto* B = node.b()->data();
  // auto* C = node.data();
  // for (Index i = 0; i < node.elems(); ++i) { C[i] = A[i] + B[i]; }
}

template <>
void Backend::forward<Sub, DeviceTy::Generic>(Sub& node)
{
  // auto* A = node.a()->data();
  // auto* B = node.b()->data();
  // auto* C = node.data();
  // for (Index i = 0; i < node.elems(); ++i) { C[i] = A[i] - B[i]; }
}
*/

} // namespace X86
} // namespace backend
} // namespace dnn