#include "dnn/node.h"
#include "dnn/backend/executor.h"
#include "dnn/context.h"
#include "dnn/utils/support.h"

namespace dnn {

static Executor executor = Executor();

#define DEF_NODE_FUNC(NAME)                                  \
  void NAME::forward() { executor.dispatch_forward(*this); } \
  void NAME::backward() {}

DEF_NODE_FUNC(Placeholder)
DEF_NODE_FUNC(Add)
DEF_NODE_FUNC(Sub)
DEF_NODE_FUNC(Mul)
DEF_NODE_FUNC(Matmul)

template <>
const Type& infer_type(const Add& node)
{
  return node.inputs()[0]->type();
}
template <>
const Type& infer_type(const Sub& node)
{
  return node.inputs()[0]->type();
}
template <>
const Type& infer_type(const Mul& node)
{
  return node.inputs()[0]->type();
}
template <>
const Type& infer_type(const Matmul& node)
{
  return node.inputs()[0]->type();
}

} // namespace dnn