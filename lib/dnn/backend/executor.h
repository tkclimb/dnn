#pragma once
#include "dnn/backend/generic.h"
#include "dnn/backend/x86.h"
#include "dnn/node.h"
#include "dnn/type.h"
#include "dnn/utils/checking.h"

namespace dnn {

class Executor
{
public:
  Executor() = default;
  ~Executor() = default;

  inline void dispatch_forward(Node& node)
  {
#define CALL(T) dispatch_forward_by_dtype<T>((T&)node);
    SWITCH_BY_NTYPE(node.ntype(), CALL)
#undef CALL
  }

  template <typename NodeT>
  inline void dispatch_forward_by_dtype(NodeT& node)
  {
#define CALL(T) dispatch_forward_by_devtype<NodeT, DataTy::T>(node);
    SWITCH_BY_DTYPE(node.dtype(), CALL)
#undef CALL
  }

  template <typename NodeT, DataTy DataT>
  inline void dispatch_forward_by_devtype(NodeT& node)
  {
#define CALL(T) backend::T::forward<DataT>(node);
    SWITCH_BY_DEVTYPE(node.devtype(), CALL)
#undef CALL
  }
};

} // namespace dnn