#include "dnn/tensor/tensor.h"
#include "dnn/context.h"
#include "dnn/node.h"

namespace dnn {

Tensor::Tensor(Node* node)
  : name_{node->name() + "::tensor"}, owner_{node}, type_{node->type()} {};

Tensor* make_tensor(const std::string& name, const Type& ty, Context& ctx)
{
  return ctx.alloc_tensor(name, ty);
}

} // namespace dnn