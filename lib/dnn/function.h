#pragma once
#include <memory>
#include <string>
#include <vector>
#include "dnn/context.h"
#include "dnn/node.h"

namespace dnn {

class Functional
{
public:
  static NodePtr placeholder(const std::string& name, const Shape& shape,
                             const Data& data, const Context& ctx)
  {
    auto nptr = std::make_shared<Placeholder>(&data, shape, ctx);
    nptr->set_name(name);
    nptr->forward();
    return nptr;
  }

  static NodePtr add(const std::string& name, NodePtr a, NodePtr b,
                     const Context& ctx)
  {
    auto nptr = std::make_shared<Add>(a, b, ctx);
    nptr->set_name(name);
    nptr->forward();
    return nptr;
  }
};

} // namespace dnn
