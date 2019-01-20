#include "dnn/node.h"
#include "dnn/backend/executor.h"
#include "dnn/context.h"
#include "dnn/utils/support.h"

namespace dnn {

static Executor executor = Executor();

void Placeholder::forward() { executor.dispatch_forward(this); }
void Placeholder::backward() {}

void Add::forward() { executor.dispatch_forward(this); }
void Add::backward() {}
template <>
inline const Type& infer_type<Add>(NodePtr a, NodePtr)
{
  return a->type();
}

void Sub::forward() { executor.dispatch_forward(this); }
void Sub::backward() {}
template <>
inline const Type& infer_type<Sub>(NodePtr a, NodePtr)
{
  return a->type();
}

} // namespace dnn