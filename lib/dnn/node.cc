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
template <>
const Type& infer_type<Add>(NodePtr a, NodePtr)
{
  return a->type();
}

DEF_NODE_FUNC(Sub)
template <>
const Type& infer_type<Sub>(NodePtr a, NodePtr)
{
  return a->type();
}

DEF_NODE_FUNC(Mul)
template <>
const Type& infer_type<Mul>(NodePtr a, NodePtr)
{
  return a->type();
}

DEF_NODE_FUNC(Matmul)
template <>
const Type& infer_type<Matmul>(NodePtr a, NodePtr)
{
  return a->type();
}

} // namespace dnn