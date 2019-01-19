#include "dnn/visitor.h"

namespace dnn {

template <>
void TensorNode<Placeholder>::accept(Visitor* v) const
{
  v->visit_pre((const Placeholder*)this);
  v->visit((const Placeholder*)this);
  v->visit_post((const Placeholder*)this);
}

template <>
void BinaryOpNode<Add>::accept(Visitor* v) const
{
  v->visit_pre((const Add*)this);
  v->visit((const Add*)this);
  v->visit_post((const Add*)this);
}

void Visitor::visit(const Placeholder*) {}
void Visitor::visit(const Add* x)
{
  x->input0()->accept(this);
  x->input1()->accept(this);
}

using std::cout;
using std::endl;

void PrintVisitor::visit_pre(const Placeholder*) {}
void PrintVisitor::visit_post(const Placeholder* x)
{
  cout << "Placeholder[" << x->name() << "]" << endl;
}

void PrintVisitor::visit_pre(const Add*) {}
void PrintVisitor::visit_post(const Add* x)
{
  cout << "Add[" << x->name() << "]" << endl;
}

} // namespace dnn