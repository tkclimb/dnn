#pragma once
#include <memory>
#include <string>
#include <vector>
#include "dnn/context.h"
#include "dnn/node.h"

namespace dnn {
namespace functional {

NodePtr placeholder(const std::string& name, const Shape& shape,
                    const Data& data, const Context& ctx)
{
  auto nptr = std::make_shared<Placeholder>(name, shape, data, ctx);
  nptr->forward();
  return nptr;
}

NodePtr add(const std::string& name, NodePtr a, NodePtr b, const Context& ctx)
{
  auto nptr = std::make_shared<Add>(name, a, b, ctx);
  nptr->forward();
  return nptr;
}

} // namespace functional
} // namespace dnn
