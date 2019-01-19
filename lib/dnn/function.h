#pragma once
#include <memory>
#include <string>
#include <vector>
#include "dnn/node.h"

namespace dnn {
namespace functional {

NodePtr placeholder(const std::string& name, const Shape& shape,
                    const Data& data)
{
  return std::make_shared<Placeholder>(name, Tensor{shape, data});
}

NodePtr add(const std::string& name, NodePtr a, NodePtr b)
{
  return std::make_shared<Add>(name, a, b);
}

} // namespace functional
} // namespace dnn
