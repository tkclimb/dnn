#pragma once
#include <functional>
#include "dnn/node.h"

namespace dnn {

#define DECL_VISIT(T)                                          \
  void visit(const T *);                                       \
  void visit(const T *, const VisitFunc &, const VisitFunc &); \
  virtual void visit_pre(const T *);                           \
  virtual void visit_post(const T *);                          \
  void visit_pre(const T *, const VisitFunc &);                \
  void visit_post(const T *, const VisitFunc &);

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

// #define DEF_CUSTOM_VISITOR(NAME)     \
//   class NAME : public Visitor        \
//   {                                  \
//   public:                            \
//     NAME() = default;                \
//     ~NAME() = default;               \
//                                      \
//     DECL_VISIT_OVERRIDE(Placeholder) \
//     DECL_VISIT_OVERRIDE(Add)         \
//     DECL_VISIT_OVERRIDE(Sub)         \
//     DECL_VISIT_OVERRIDE(Mul)         \
//   };

// DEF_CUSTOM_VISITOR(PrintVisitor)

} // namespace dnn