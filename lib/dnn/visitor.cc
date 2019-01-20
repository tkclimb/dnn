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

template <>
void BinaryOpNode<Sub>::accept(Visitor* v) const
{
  v->visit_pre((const Sub*)this);
  v->visit((const Sub*)this);
  v->visit_post((const Sub*)this);
}

void Visitor::visit(const Placeholder*) {}
void Visitor::visit(const Add* x)
{
  x->input0()->accept(this);
  x->input1()->accept(this);
}
void Visitor::visit(const Sub* x)
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

void PrintVisitor::visit_pre(const Sub*) {}
void PrintVisitor::visit_post(const Sub* x)
{
  cout << "Sub[" << x->name() << "]" << endl;
}

} // namespace dnn