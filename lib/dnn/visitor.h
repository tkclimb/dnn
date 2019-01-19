#pragma once
#include "dnn/node.h"

namespace dnn {

class Visitor
{
public:
  Visitor() = default;
  ~Visitor() = default;

  void visit(const Placeholder*);
  void visit(const Add*);
  virtual void visit_pre(const Placeholder*) = 0;
  virtual void visit_pre(const Add*) = 0;
  virtual void visit_post(const Placeholder*) = 0;
  virtual void visit_post(const Add*) = 0;
};

class PrintVisitor : public Visitor
{
public:
  PrintVisitor() = default;
  ~PrintVisitor() = default;

  virtual void visit_pre(const Placeholder*);
  virtual void visit_pre(const Add*);
  virtual void visit_post(const Placeholder*);
  virtual void visit_post(const Add*);
};

} // namespace dnn