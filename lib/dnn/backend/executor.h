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
#define CALL(T) dispatch_forward_by_dataty<T>((T&)node);
    SWITCH_BY_NODETY(node.nodety(), CALL)
#undef CALL
  }

  template <typename NodeT>
  inline void dispatch_forward_by_dataty(NodeT& node)
  {
#define CALL(T) dispatch_forward_by_devty<NodeT, DataTy::T>(node);
    SWITCH_BY_DATATY(node.dataty(), CALL)
#undef CALL
  }

  template <typename NodeT, DataTy DataT>
  inline void dispatch_forward_by_devty(NodeT& node)
  {
#define CALL(T) backend::T::forward<DataT>(node);
    SWITCH_BY_DEVTY(node.devty(), CALL)
#undef CALL
  }
};

} // namespace dnn