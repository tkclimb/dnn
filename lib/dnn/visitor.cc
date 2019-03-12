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

#define DEF_VISIT(NAME)                                             \
  DEF_VISITABLE_NODE(NAME)                                          \
  void Visitor::visit(const NAME *x)                                \
  {                                                                 \
    for (auto &i : x->inputs()) { i->accept(this); }                \
  }                                                                 \
  void Visitor::visit(const NAME *x, const VisitFunc &f_pre,        \
                      const VisitFunc &f_post)                      \
  {                                                                 \
    for (auto &i : x->inputs()) { i->accept(this, f_pre, f_post); } \
  }

#define DEF_VISIT_PRE_AND_POST(NAME)                                   \
  void Visitor::visit_pre(const NAME *)                                \
  {                                                                    \
    EXCEPTION(#NAME " doesn't implement visit_pre...");                \
  }                                                                    \
  void Visitor::visit_post(const NAME *)                               \
  {                                                                    \
    EXCEPTION(#NAME " doesn't implement visit_post...");               \
  }                                                                    \
  void Visitor::visit_pre(const NAME *x, const VisitFunc &f) { f(x); } \
  void Visitor::visit_post(const NAME *x, const VisitFunc &f) { f(x); }

#define DEF_VISIT_FUNC_FOR(NAME) \
  DEF_VISIT(NAME)                \
  DEF_VISIT_PRE_AND_POST(NAME)

DEFINED_NODETYS(DEF_VISIT_FUNC_FOR)

} // namespace dnn