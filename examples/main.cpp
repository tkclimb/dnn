#include "dnn/context.h"
#include "dnn/function.h"
#include "dnn/graph.h"
#include "dnn/node.h"
#include "dnn/type.h"
#include "dnn/utils/support.h"
#include "dnn/visitor.h"

using namespace dnn;

using F = Functional;

int main(int argc, char const* argv[])
{
  auto ctx = Context();
  Type ty{DataTy::F32, {2, 3}};
  Tensor t1(ty);
  Tensor t2(ty);

  auto acc1 = t1.get_access<DataTy::F32>();
  acc1 = {1, 2, 3, 4, 5, 6};
  auto acc2 = t2.get_access<DataTy::F32>();
  acc2 = {7, 8, 9, 10, 11, 12};

  auto p1 = F::placeholder("p1", t1, ctx);
  auto p2 = F::placeholder("p2", t2, ctx);
  // auto add = F::add("add1", p1, p2, ctx);
  print(p2->tensor());

  // auto v = new PrintVisitor();
  // add->accept(v);

  return 0;
}