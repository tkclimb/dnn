#include "dnn/context.h"
#include "dnn/graph.h"
#include "dnn/node.h"
#include "dnn/type.h"
#include "dnn/utils/support.h"
#include "dnn/visitor.h"

using namespace dnn;

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

  Graph g;

  auto& p1 = g.placeholder("p1", ty, g, ctx);
  auto& p2 = g.placeholder("p2", ty, g, ctx);
  auto& add = g.add("add1", ty, g, ctx);
  auto& mul = g.mul("mul1", ty, g, ctx);

  auto p1_t = p1({&t1});
  auto p2_t = p2({&t2});
  auto add_t = add({p1_t, p2_t});
  // auto add_t = mul({add, &t2});

  print(*add_t);
  // print(mul->tensor());

  // auto v = new PrintVisitor();
  // add->accept(v);

  return 0;
}