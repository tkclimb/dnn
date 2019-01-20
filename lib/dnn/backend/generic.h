#pragma once
#include "dnn/backend/macro.h"
#include "dnn/node.h"
#include "dnn/type.h"

namespace dnn {
namespace backend {
namespace Generic {

// DEF_DEFAULT_BACKEND_FORWARD(Generic, Placeholder)
template <DataTy T>
void forward(Placeholder* node)
{
  auto acc = node->tensor().get_access<T>();
  acc.init();
  auto ref = node->ref->get_access<T>();
  for (Index i = 0; i < node->elems(); ++i) { acc[i] = ref[i]; }
}

DEF_DEFAULT_BACKEND_FORWARD(Generic, Add)
DEF_DEFAULT_BACKEND_FORWARD(Generic, Sub)

} // namespace Generic
} // namespace backend
} // namespace dnn