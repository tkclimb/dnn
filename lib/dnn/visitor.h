#pragma once
#include <functional>
#include "dnn/node.h"

namespace dnn {

#define DECL_VISIT(NAME)                                          \
  void visit(const NAME *);                                       \
  void visit(const NAME *, const VisitFunc &, const VisitFunc &); \
  virtual void visit_pre(const NAME *);                           \
  virtual void visit_post(const NAME *);                          \
  void visit_pre(const NAME *, const VisitFunc &);                \
  void visit_post(const NAME *, const VisitFunc &);

class Visitor
{
public:
  Visitor() = default;
  ~Visitor() = default;

  DECL_VISIT(Placeholder)
  DECL_VISIT(Add)
  DECL_VISIT(Sub)
};

#define DECL_VISIT_OVERRIDE(NAME)        \
  void visit_pre(const NAME *) override; \
  void visit_post(const NAME *) override;

#define DEF_CUSTOM_VISITOR(NAME)     \
  class NAME : public Visitor        \
  {                                  \
  public:                            \
    NAME() = default;                \
    ~NAME() = default;               \
                                     \
    DECL_VISIT_OVERRIDE(Placeholder) \
    DECL_VISIT_OVERRIDE(Add)         \
    DECL_VISIT_OVERRIDE(Sub)         \
  };

DEF_CUSTOM_VISITOR(PrintVisitor)

} // namespace dnn