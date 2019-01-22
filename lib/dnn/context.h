#pragma once
#include <memory>
#include "dnn/type.h"

namespace dnn {

class Node;
using NodePtr = std::shared_ptr<Node>;

class Context final
{
private:
  HostTy target_ = HostTy::X86;
  DeviceTy device_ = DeviceTy::Generic;

public:
  Context() = default;
  ~Context() = default;
  Context(const HostTy target, const DeviceTy device)
    : target_{target}, device_{device} {};

  inline HostTy target() const { return target_; }
  inline DeviceTy devtype() const { return device_; }

  // static Context CPU() { return Context(HostTy::CPU, DeviceTy::Generic); }
  // static Context GPU() { return Context(HostTy::GPU, DeviceTy::Geforce); }
};

} // namespace dnn