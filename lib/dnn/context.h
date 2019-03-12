#pragma once
#include <map>
#include <memory>
#include "dnn/tensor/tensor.h"
#include "dnn/type.h"

namespace dnn {

class Node;
using NodePtr = Node*;

class Context final
{
  using TensorMap = std::map<std::string, Tensor*>;

private:
  HostTy hostty_ = HostTy::X86;
  DeviceTy device_ = DeviceTy::Generic;
  TensorMap tensor_map_;

public:
  Context() = default;
  ~Context() = default;
  Context(const HostTy hostty, const DeviceTy device)
    : hostty_{hostty}, device_{device} {};

  inline HostTy hostty() const { return hostty_; }
  inline DeviceTy devty() const { return device_; }

  Tensor* alloc_tensor(const std::string&, const Type&);
  Tensor* alloc_tensor(Node*);
  Tensor* get_tensor(Node*);
  const Tensor* get_tensor(const Node*);

  // static Context CPU() { return Context(HostTy::CPU, DeviceTy::Generic); }
  // static Context GPU() { return Context(HostTy::GPU, DeviceTy::Geforce); }
};

} // namespace dnn