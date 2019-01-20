#pragma once
#include <vector>

namespace dnn {

using DEFAULT_DTYPE = float;
using Index = std::uint32_t;
using Shape = std::vector<Index>;
using Data = std::vector<DEFAULT_DTYPE>;

enum class NodeTy
{
  Placeholder,
  Add,
};

enum class TargetTy
{
  CPU,
  GPU,
  Others,
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

} // namespace dnn