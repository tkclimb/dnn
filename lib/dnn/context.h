#pragma once
#include <memory>
#include "dnn/type.h"

namespace dnn {

class Node;
using NodePtr = std::shared_ptr<Node>;

class Context final
{
private:
  HostTy hostty_ = HostTy::X86;
  DeviceTy device_ = DeviceTy::Generic;

public:
  Context() = default;
  ~Context() = default;
  Context(const HostTy hostty, const DeviceTy device)
    : hostty_{hostty}, device_{device} {};

  inline HostTy hostty() const { return hostty_; }
  inline DeviceTy devty() const { return device_; }

  // static Context CPU() { return Context(HostTy::CPU, DeviceTy::Generic); }
  // static Context GPU() { return Context(HostTy::GPU, DeviceTy::Geforce); }
};

} // namespace dnn