#include "dnn/node.h"
#include "dnn/backend/interface.h"
#include "dnn/context.h"
#include "dnn/utils/support.h"

namespace dnn {

void Placeholder::forward() {}
void Placeholder::backward() {}

template <>
Shape BinaryOpNode<Add>::infer_shape(NodePtr a, NodePtr b) const
{
  return a->shape();
}

void Add::forward()
{
  if (device() == DeviceTy::Generic) {
    backend::forward<Add, DeviceTy::Generic>(*this);
  }
}
void Add::backward() {}

} // namespace dnn