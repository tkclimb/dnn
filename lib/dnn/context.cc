#include "dnn/context.h"
#include "dnn/arrayref.h"
#include "dnn/node.h"
#include "dnn/tensor/tensor.h"

namespace dnn {

Tensor* Context::alloc_tensor(const std::string& name, const Type& ty)
{
  if (tensor_map_.count(name)) {
    EXCEPTION_STR("tensor(" + name +
                  ") is already allocated in this context...");
  } else {
    tensor_map_.emplace(name, new Tensor(name, ty));
  }
  return tensor_map_[name];
}

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

const Tensor* Context::assign_in_tensor(const Node* node, const Tensor* tensor)
{
  auto& name = node->name();
  in_tensor_map_[name].push_back(tensor);
  return tensor;
}

const Tensor* Context::get_in_tensor(const Node* node, const Index idx)
{
  auto& name = node->name();
  if (!in_tensor_map_.count(name)) {
    EXCEPTION_STR("tensor(" + name + ") is not allocated as in-tensor...");
  }
  return in_tensor_map_[name][idx];
}

} // namespace dnn