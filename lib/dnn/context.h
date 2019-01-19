#pragma once
#include <memory>

namespace {

class Node;
using NodePtr = std::shared_ptr<Node>;

enum class TargetTy
{
  Generic,
  CPU,
  GPU,
};

enum class DeviceTy
{
  Generic,
  X86,
  Arm32,
  Arm64,
  Geforce,
  Vega,
  Radeon,
};

class Context
{
private:
  TargetTy target_;
  DeviceTy device_;

public:
  Context(const TargetTy target, const DeviceTy device)
    : target_{target}, device_{device} {};
  ~Context() = default;

  inline TargetTy target() const { return target_; }
  inline DeviceTy device() const { return device_; }

  static Context CPU() { return Context(TargetTy::CPU, DeviceTy::X86); }
  static Context GPU() { return Context(TargetTy::GPU, DeviceTy::Geforce); }
};

} // namespace