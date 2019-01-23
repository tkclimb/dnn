#pragma once
#include "dnn/backend/macro.h"
#include "dnn/node.h"
#include "dnn/type.h"

namespace dnn {
namespace backend {
namespace X86 {

DEF_DEFAULT_BACKEND_FORWARD(X86, Placeholder)
DEF_DEFAULT_BACKEND_FORWARD(X86, Add)
DEF_DEFAULT_BACKEND_FORWARD(X86, Sub)
DEF_DEFAULT_BACKEND_FORWARD(X86, Mul)
DEF_DEFAULT_BACKEND_FORWARD(X86, Mutmul)

} // namespace X86
} // namespace backend
} // namespace dnn