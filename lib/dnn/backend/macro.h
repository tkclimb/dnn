#pragma once
#include "dnn/type.h"
#include "dnn/utils/checking.h"
#include "dnn/utils/support.h"

namespace dnn {
namespace backend {

#define TO_STR(S) #S
#define MESSAGE(DEV, F, N) TO_STR(is not implemented for backend::DEV::F(N*)...)

#define DEF_DEFAULT_BACKEND_FUNC(DEV, F, N) \
  template <DataTy T>                       \
  void F(N*)                                \
  {                                         \
    std::string msg = to_string(T);         \
    msg += " ";                             \
    msg += MESSAGE(DEV, F, N);              \
    EXCEPTION_STR(msg);                     \
  }

#define DEF_DEFAULT_BACKEND_FORWARD(DEV, N) \
  DEF_DEFAULT_BACKEND_FUNC(DEV, forward, N)

} // namespace backend
} // namespace dnn