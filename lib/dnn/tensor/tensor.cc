#include "dnn/tensor/tensor.h"
#include "dnn/node.h"

namespace dnn {

Tensor::Tensor(Node* node)
  : name_{node->name() + "::tensor"}, owner_{node}, type_{node->type()} {};

} // namespace dnn