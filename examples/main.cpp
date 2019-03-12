#include <functional>

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
  auto t1 = make_tensor("t1", ty, ctx);
  auto t2 = make_tensor("t2", ty, ctx);
  auto t3 = make_tensor("t3", ty, ctx);

  auto acc1 = t1->get_access<DataTy::F32>();
  acc1 = {1, 2, 3, 4, 5, 6};
  auto acc2 = t2->get_access<DataTy::F32>();
  acc2 = {7, 8, 9, 10, 11, 12};
  auto acc3 = t3->get_access<DataTy::F32>();
  acc3 = {10, 10, 10, 10, 10, 10};

  Graph g;

  auto& add = g.add("add1", ty, ctx);
  auto& mul = g.mul("mul1", ty, ctx);

  auto add_t = add({t1, t2});
  auto mul_t = mul({add_t, t3});

  print(*mul_t);

  VisitFunc do_nothing = [](const Node* x) { ; };
  VisitFunc print_node_name = [](const Node* x) {
    std::cout << to_string(x->nodety()) << std::endl;
  };
  auto v = new Visitor();
  mul.accept(v, do_nothing, print_node_name);

  return 0;
}