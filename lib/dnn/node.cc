#include "dnn/node.h"
#include "dnn/backend/executor.h"
#include "dnn/context.h"
#include "dnn/utils/support.h"

namespace dnn {

static Executor executor = Executor();

#define DEF_NODE_FUNC(NAME)                  \
  void NAME::forward(TensorArray in_tensors) \
  {                                          \
    in_tensors_ = in_tensors;                \
    executor.dispatch_forward(*this);        \
  } // namespace dnn

DEF_NODE_FUNC(Placeholder)
DEF_NODE_FUNC(Add)
DEF_NODE_FUNC(Sub)
DEF_NODE_FUNC(Mul)
DEF_NODE_FUNC(Matmul)

template <>
const Type& infer_type<Add>(TensorArray inputs)
{
  return inputs[0]->type();
}
template <>
const Type& infer_type<Sub>(TensorArray inputs)
{
  return inputs[0]->type();
}
template <>
const Type& infer_type<Mul>(TensorArray inputs)
{
  return inputs[0]->type();
}
template <>
const Type& infer_type<Matmul>(TensorArray inputs)
{
  return inputs[0]->type();
}

} // namespace dnn