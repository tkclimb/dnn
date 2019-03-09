#pragma once
#include <map>
#include <memory>
#include "dnn/tensor/tensor.h"
#include "dnn/type.h"

namespace dnn {

class Node;
using NodePtr = std::shared_ptr<Node>;

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

  Tensor* alloc_tensor(const std::string& name, const Type& ty)
  {
    if (tensor_map_.count(name)) {
      EXCEPTION_STR("tensor(" + name +
                    ") is already allocated in this context...");
    } else {
      tensor_map_.emplace(name, new Tensor(ty));
    }
    return tensor_map_[name];
  }

  Tensor* get_tensor(const std::string& name)
  {
    if (!tensor_map_.count(name)) {
      EXCEPTION_STR("tensor(" + name +
                    ") is not allocated in this context yet...");
    }
    return tensor_map_[name];
  }

  void set_tensor(const std::string& name, Tensor& tensor)
  {
    tensor_map_[name] = &tensor;
  }

  // static Context CPU() { return Context(HostTy::CPU, DeviceTy::Generic); }
  // static Context GPU() { return Context(HostTy::GPU, DeviceTy::Geforce); }
};

} // namespace dnn