#include "dnn/visitor.h"

namespace dnn {

#define DEF_VISITABLE_NODE(NAME)                                       \
  template <>                                                          \
  void VisitableNode<NAME>::accept(Visitor *v) const                   \
  {                                                                    \
    v->visit_pre((const NAME *)this);                                  \
    v->visit((const NAME *)this);                                      \
    v->visit_post((const NAME *)this);                                 \
  }                                                                    \
  template <>                                                          \
  void VisitableNode<NAME>::accept(Visitor *v, const VisitFunc &f_pre, \
                                   const VisitFunc &f_post) const      \
  {                                                                    \
    v->visit_pre((const NAME *)this, f_pre);                           \
    v->visit((const NAME *)this, f_pre, f_post);                       \
    v->visit_post((const NAME *)this, f_post);                         \
  }

#define DEF_VISITABLE_NOARG_NODE(NAME) \
  DEF_VISITABLE_NODE(NAME)             \
  void Visitor::visit(const NAME *) {} \
  void Visitor::visit(const NAME *, const VisitFunc &, const VisitFunc &) {}

#define DEF_VISITABLE_BINARY_OP_NODE(NAME)                             \
  DEF_VISITABLE_NODE(NAME)                                             \
  void Visitor::visit(const NAME *x)                                   \
  {                                                                    \
    if (x->in_tensors().size() != 2) {                                 \
      EXCEPTION_STR("node(" + x->name() + ") doesn't initialized..."); \
    }                                                                  \
    /* for (auto &x : x.in_tensors()) { x.accept(this); } */           \
  }                                                                    \
  void Visitor::visit(const NAME *x, const VisitFunc &f_pre,           \
                      const VisitFunc &f_post)                         \
  {                                                                    \
    if (x->in_tensors().size() != 2) {                                 \
      EXCEPTION_STR("node(" + x->name() + ") doesn't initialized..."); \
    }                                                                  \
    /* for (auto &next_node : x->inputs_) { x.accept(this); } */       \
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

#define DEF_VISIT_NOARG_NODE(NAME) \
  DEF_VISITABLE_NOARG_NODE(NAME)   \
  DEF_VISIT_PRE_AND_POST(NAME)

#define DEF_VISIT_BINARY_OP_NODE(NAME) \
  DEF_VISITABLE_BINARY_OP_NODE(NAME)   \
  DEF_VISIT_PRE_AND_POST(NAME)

DEF_VISIT_NOARG_NODE(Placeholder)
DEF_VISIT_BINARY_OP_NODE(Add)
DEF_VISIT_BINARY_OP_NODE(Sub)
DEF_VISIT_BINARY_OP_NODE(Mul)
DEF_VISIT_BINARY_OP_NODE(Matmul)

} // namespace dnn