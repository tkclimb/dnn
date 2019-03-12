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
  DEFINED_NODETYS(DECL_VISIT)
};

#define DECL_VISIT_OVERRIDE(NAME)        \
  void visit_pre(const NAME *) override; \
  void visit_post(const NAME *) override;

#define DEF_CUSTOM_VISITOR(NAME)         \
  class NAME : public Visitor            \
  {                                      \
  public:                                \
    NAME() = default;                    \
    ~NAME() = default;                   \
    DEFINED_NODETYS(DECL_VISIT_OVERRIDE) \
  };

} // namespace dnn