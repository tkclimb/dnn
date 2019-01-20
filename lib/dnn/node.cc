#include "dnn/node.h"
#include "dnn/backend/interface.h"
#include "dnn/context.h"
#include "dnn/utils/support.h"

namespace dnn {

static Backend backend = Backend();

void Placeholder::forward()
{
  if (device() == DeviceTy::Generic) {
    backend.forward<Placeholder, DeviceTy::Generic>(*this);
  }
}
void Placeholder::backward() {}

template <>
Shape BinaryOpNode<Add>::infer_shape(NodePtr a, NodePtr) const
{
  return a->shape();
}

template <>
Shape BinaryOpNode<Sub>::infer_shape(NodePtr a, NodePtr) const
{
  return a->shape();
}

void Add::forward() { backend.dispatch_forward(*this); }
void Add::backward() {}
void Sub::forward() { backend.dispatch_forward(*this); }
void Sub::backward() {}

} // namespace dnn