#pragma once
#include <memory>
#include "dnn/type.h"

namespace dnn {

class Node;
using NodePtr = std::shared_ptr<Node>;

class Context final
{
private:
  TargetTy target_ = TargetTy::CPU;
  DeviceTy device_ = DeviceTy::Generic;

public:
  Context() = default;
  ~Context() = default;
  Context(const TargetTy target, const DeviceTy device)
    : target_{target}, device_{device} {};

  inline TargetTy target() const { return target_; }
  inline DeviceTy device() const { return device_; }

  // static Context CPU() { return Context(TargetTy::CPU, DeviceTy::Generic); }
  // static Context GPU() { return Context(TargetTy::GPU, DeviceTy::Geforce); }
};

} // namespace dnn