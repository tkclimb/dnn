#include "dnn/visitor.h"

namespace dnn {

#define DEF_VISITABLE_NODE(TY, NAME)                        \
  template <>                                               \
  void TY<NAME>::accept(Visitor *v) const                   \
  {                                                         \
    v->visit_pre((const NAME *)this);                       \
    v->visit((const NAME *)this);                           \
    v->visit_post((const NAME *)this);                      \
  }                                                         \
  template <>                                               \
  void TY<NAME>::accept(Visitor *v, const VisitFunc &f_pre, \
                        const VisitFunc &f_post) const      \
  {                                                         \
    v->visit_pre((const NAME *)this, f_pre);                \
    v->visit((const NAME *)this, f_pre, f_post);            \
    v->visit_post((const NAME *)this, f_post);              \
  }

#define DEF_VISITABLE_TENSOR_NODE(NAME) \
  DEF_VISITABLE_NODE(TensorNode, NAME)  \
  void Visitor::visit(const NAME *) {}  \
  void Visitor::visit(const NAME *, const VisitFunc &, const VisitFunc &) {}

#define DEF_VISITABLE_BINARY_OP_NODE(NAME)                   \
  DEF_VISITABLE_NODE(BinaryOpNode, NAME)                     \
  void Visitor::visit(const NAME *x)                         \
  {                                                          \
    x->a()->accept(this);                                    \
    x->b()->accept(this);                                    \
  }                                                          \
  void Visitor::visit(const NAME *x, const VisitFunc &f_pre, \
                      const VisitFunc &f_post)               \
  {                                                          \
    x->a()->accept(this, f_pre, f_post);                     \
    x->b()->accept(this, f_pre, f_post);                     \
  }

#define DEF_VISIT_PRE_AND_POST(NAME)                                   \
  void Visitor::visit_pre(const NAME *)                                \
  {                                                                    \
    EXCEPTION(#NAME + " doesn't implement visit_pre...");              \
  }                                                                    \
  void Visitor::visit_post(const NAME *)                               \
  {                                                                    \
    EXCEPTION(#NAME + " doesn't implement visit_post...");             \
  }                                                                    \
  void Visitor::visit_pre(const NAME *x, const VisitFunc &f) { f(x); } \
  void Visitor::visit_post(const NAME *x, const VisitFunc &f) { f(x); }

#define DEF_VISIT_TENSOR_NODE(T) \
  DEF_VISITABLE_TENSOR_NODE(T)   \
  DEF_VISIT_PRE_AND_POST(T)

#define DEF_VISIT_BINARY_OP_NODE(T) \
  DEF_VISITABLE_BINARY_OP_NODE(T)   \
  DEF_VISIT_PRE_AND_POST(T)

DEFINED_NODETYS_BY_OPS(DEF_VISIT_TENSOR_NODE, _, DEF_VISIT_BINARY_OP_NODE)

// void PrintVisitor::visit_pre(const Placeholder *) {}
// void PrintVisitor::visit_post(const Placeholder *x)
// {
//   cout << "Placeholder[" << x->name() << "]" << endl;
// }

// void PrintVisitor::visit_pre(const Add *) {}
// void PrintVisitor::visit_post(const Add *x)
// {
//   cout << "Add[" << x->name() << "]" << endl;
// }

// void PrintVisitor::visit_pre(const Sub *) {}
// void PrintVisitor::visit_post(const Sub *x)
// {
//   cout << "Sub[" << x->name() << "]" << endl;
// }

// void PrintVisitor::visit_pre(const Sub *) {}
// void PrintVisitor::visit_post(const Sub *x)
// {
//   cout << "Sub[" << x->name() << "]" << endl;
// }

} // namespace dnn