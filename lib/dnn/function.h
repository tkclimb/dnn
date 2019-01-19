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
  return std::make_shared<Placeholder>(name, shape, data, ctx);
}

NodePtr add(const std::string& name, NodePtr a, NodePtr b, const Context& ctx)
{
  return std::make_shared<Add>(name, a, b, ctx);
}

} // namespace functional
} // namespace dnn
