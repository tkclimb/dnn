#pragma once
#include <vector>

namespace dnn {

using DEFAULT_DTYPE = float;
using Index = std::uint32_t;
using Shape = std::vector<Index>;
using Data = std::vector<DEFAULT_DTYPE>;

} // namespace dnn