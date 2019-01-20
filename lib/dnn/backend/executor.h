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

  inline void dispatch_forward(Node* node)
  {
#define CALL(T) f<T>((T*)node);
    SWITCH_BY_NTYPE(node->node_type(), CALL)
#undef CALL
  }

  template <typename NodeT>
  inline void f(NodeT* node)
  {
#define CALL(T) ff<NodeT, DataTy::T>(node);
    SWITCH_BY_DTYPE(node->dtype(), CALL)
#undef CALL
  }

  template <typename NodeT, DataTy DataT>
  inline void ff(NodeT* node)
  {
// #define CALL(T) backend::forward<NodeT, DevT, DataTy::T>(node);
#define CALL(T) backend::T::forward<DataT>(node);
    SWITCH_BY_DEVTYPE(node->device(), CALL)
#undef CALL
  }
};

} // namespace dnn