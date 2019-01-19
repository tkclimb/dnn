#pragma once

namespace dnn {

class Visitor
{
public:
  Visitor();
  ~Visitor();
};

template <typename T>
class Visitable
{
public:
  void accept(const Visitor*) const;
};

} // namespace dnn