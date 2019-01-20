#include "dnn/context.h"
#include "dnn/function.h"
#include "dnn/graph.h"
#include "dnn/node.h"
#include "dnn/tensor.h"
#include "dnn/utils/support.h"
#include "dnn/visitor.h"

using namespace dnn;

using F = Functional;

int main(int argc, char const* argv[])
{
  auto ctx = Context();
  Shape s{2, 3};
  Data d1{1, 2, 3, 4, 5, 6};
  Data d2{1, 2, 3, 4, 5, 6};

  auto p1 = F::placeholder("p1", s, d1, ctx);
  auto p2 = F::placeholder("p2", s, d2, ctx);
  auto add = F::add("add1", p1, p2, ctx);

  // print(add->tensor());
  // auto v = new Visitor();
  auto v = new PrintVisitor();
  add->accept(v);

  // auto input = dnn::make_tensor(1, 28, 28, 1);

  return 0;
}