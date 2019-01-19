#include "dnn/node.h"

namespace dnn {

void Placeholder::forward() {}
void Placeholder::backward() {}

Shape Add::infer_shape(NodePtr a, NodePtr b) const { return a->shape(); }
void Add::forward() {}
void Add::backward() {}

} // namespace dnn