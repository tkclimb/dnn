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
  static NodePtr placeholder(const std::string& name, const Type& ty,
                             Graph& graph, Context& ctx)
  {
    // auto nptr = std::make_shared<Placeholder>(name, ty, ctx);
    // graph.add_node(nptr, name);
    // return nptr;
  }

  static NodePtr add(const std::string& name, NodePtr a, NodePtr b,
                     Graph& graph, Context& ctx)
  {
    // auto type = infer_type<Add>(a, b);
    // auto nptr = std::make_shared<Add>(name, a, b, type, ctx);
    // graph.add_node(nptr, name);
    // return nptr;
  }

  static NodePtr mul(const std::string& name, NodePtr a, NodePtr b,
                     Graph& graph, Context& ctx)
  {
    // auto type = infer_type<Mul>(a, b);
    // auto nptr = std::make_shared<Mul>(name, a, b, type, ctx);
    // graph.add_node(nptr, name);
    // return nptr;
  }
};

} // namespace dnn
