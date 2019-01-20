#pragma once
#include "dnn/context.h"
#include "dnn/node.h"

namespace dnn {
namespace backend {

using ::dnn::Add;

template <typename NodeTy, DeviceTy Device>
void forward(NodeTy& add);

} // namespace backend
} // namespace dnn