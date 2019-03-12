#include "dnn/node.h"
#include "dnn/backend/executor.h"
#include "dnn/context.h"
#include "dnn/graph.h"
#include "dnn/utils/support.h"

namespace dnn {

static Executor executor = Executor();

Node::Node(const std::string& name, const NodeTy nodety, const Type& ty,
           Graph& g, Context& ctx)
  : name_{name}, nodety_{nodety}, ty_{ty}, graph_{&g}, ctx_{&ctx}
{
  alloc_tensor();
}

const NodeVec& Node::parent_nodes() const
{
  return graph_->get_parent_nodes(const_cast<const NodePtr>(this));
}

const NodePtr Node::input(const Index idx) const
{
  auto& inputs_ = parent_nodes();
  if (inputs_.size() <= idx) {
    EXCEPTION_STR("node(" + name() + ") have only " +
                  std::to_string(inputs_.size()) + " parent nodes, not " +
                  std::to_string(idx) + "...");
  }
  return inputs_[idx];
}

void Node::set_in_tensors(TensorArray tensors)
{
  if (tensors.size() != num_inputs()) {
    EXCEPTION_STR(
      "the number of given tensors(" + std::to_string(tensors.size()) +
      ") differs from the intended(" + std::to_string(num_inputs()) + ")")
  }
  for (size_t i = 0; i < tensors.size(); i++) {
    auto paren_node = tensors[i]->owner();
    if (paren_node != nullptr) {
      graph_->add_link(this, paren_node);
    }
  }
  for (const auto t : tensors) { ctx_->assign_in_tensor(this, t); }
}

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
  return node.parent_nodes()[0]->type();
}
template <>
const Type& infer_type(const Sub& node)
{
  return node.parent_nodes()[0]->type();
}
template <>
const Type& infer_type(const Mul& node)
{
  return node.parent_nodes()[0]->type();
}
template <>
const Type& infer_type(const Matmul& node)
{
  return node.parent_nodes()[0]->type();
}

} // namespace dnn