#include "dnn/context.h"
#include "dnn/function.h"
#include "dnn/graph.h"
#include "dnn/node.h"
#include "dnn/tensor.h"
#include "dnn/visitor.h"

using namespace dnn;

namespace F = functional;

int main(int argc, char const* argv[])
{
  auto ctx = Context::CPU();
  Shape s{2, 3};
  Data d1{1, 2, 3, 4, 5, 6};
  Data d2{1, 2, 3, 4, 5, 6};

  auto p1 = F::placeholder("p1", s, d1, ctx);
  auto p2 = F::placeholder("p2", s, d2, ctx);
  auto add = F::add("add1", p1, p2, ctx);

  print_tensor(p1->tensor());
  // auto v = new PrintVisitort();
  // add->accept(v);

  // auto input = dnn::make_tensor(1, 28, 28, 1);

  return 0;
}