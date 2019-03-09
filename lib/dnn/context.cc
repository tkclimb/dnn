#include "dnn/context.h"
#include "dnn/node.h"

namespace dnn {

Tensor* Context::alloc_tensor(Node* node)
{
  auto& name = node->name();
  if (tensor_map_.count(name)) {
    EXCEPTION_STR("tensor(" + name +
                  ") is already allocated in this context...");
  } else {
    tensor_map_.emplace(name, new Tensor(node));
  }
  return tensor_map_[name];
}

Tensor* Context::get_tensor(Node* node)
{
  auto& name = node->name();
  if (!tensor_map_.count(name)) {
    EXCEPTION_STR("tensor(" + name +
                  ") is not allocated in this context yet...");
  }
  return tensor_map_[name];
}

const Tensor* Context::get_tensor(const Node* node)
{
  return get_tensor(const_cast<Node*>(node));
}

} // namespace dnn