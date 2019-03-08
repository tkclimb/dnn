#pragma once
#include <memory>
#include <string>
#include <vector>
#include "dnn/context.h"
#include "dnn/graph.h"
#include "dnn/node.h"
#include "dnn/type.h"

namespace dnn {

class Functional
{
public:
  static NodePtr placeholder(const std::string& name, const Tensor& tensor,
                             Graph& graph, const Context& ctx)
  {
    auto nptr = std::make_shared<Placeholder>(tensor, ctx);
    graph.add_node(nptr, name);
    nptr->forward();
    return nptr;
  }

  static NodePtr add(const std::string& name, NodePtr a, NodePtr b,
                     Graph& graph, const Context& ctx)
  {
    auto type = infer_type<Add>(a, b);
    auto nptr = std::make_shared<Add>(a, b, type, ctx);
    graph.add_node(nptr, name);
    nptr->forward();
    return nptr;
  }

  static NodePtr mul(const std::string& name, NodePtr a, NodePtr b,
                     Graph& graph, const Context& ctx)
  {
    auto type = infer_type<Mul>(a, b);
    auto nptr = std::make_shared<Mul>(a, b, type, ctx);
    graph.add_node(nptr, name);
    nptr->forward();
    return nptr;
  }
};

} // namespace dnn
