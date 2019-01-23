#pragma once
#include "dnn/backend/macro.h"
#include "dnn/node.h"
#include "dnn/type.h"

namespace dnn {
namespace backend {
namespace Generic {

// DEF_DEFAULT_BACKEND_FORWARD(Generic, Placeholder)
template <DataTy T>
void forward(Placeholder& node)
{
  auto y = node.tensor().get_access<T>();
  auto x = node.ref->get_access<T>();
  for (Index i = 0; i < node.elems(); ++i) { y[i] = x[i]; }
}

// DEF_DEFAULT_BACKEND_FORWARD(Generic, Add)
template <DataTy T>
void forward(Add& node)
{
  auto a = node.a()->tensor().get_access<T>();
  auto b = node.b()->tensor().get_access<T>();
  auto c = node.tensor().get_access<T>();
  for (Index i = 0; i < node.elems(); ++i) { c[i] = a[i] + b[i]; }
}

// DEF_DEFAULT_BACKEND_FORWARD(Generic, Sub)
template <DataTy T>
void forward(Sub& node)
{
  auto a = node.a()->tensor().get_access<T>();
  auto b = node.b()->tensor().get_access<T>();
  auto c = node.tensor().get_access<T>();
  for (Index i = 0; i < node.elems(); ++i) { c[i] = a[i] - b[i]; }
}

// DEF_DEFAULT_BACKEND_FORWARD(Generic, Mul)
template <DataTy T>
void forward(Mul& node)
{
  auto a = node.a()->tensor().get_access<T>();
  auto b = node.b()->tensor().get_access<T>();
  auto c = node.tensor().get_access<T>();
  for (Index i = 0; i < node.elems(); ++i) { c[i] = a[i] * b[i]; }
}

// DEF_DEFAULT_BACKEND_FORWARD(Generic, Mul)
template <DataTy T>
void forward(Mutmul& node)
{
  // auto a = node.a()->tensor().get_access<T>();
  // auto b = node.b()->tensor().get_access<T>();
  // auto c = node.tensor().get_access<T>();
  // for (Index i = 0; i < node.elems(); ++i) { c[i] = a[i] * b[i]; }
}

} // namespace Generic
} // namespace backend
} // namespace dnn