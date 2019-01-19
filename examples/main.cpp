#include "dnn/function.h"
#include "dnn/graph.h"
#include "dnn/node.h"
#include "dnn/tensor.h"

using namespace dnn;

namespace F = functional;

int main(int argc, char const* argv[])
{
  Shape s{2, 3};
  Data d1{1, 2, 3, 4, 5, 6};
  Data d2{1, 2, 3, 4, 5, 6};

  auto p1 = F::placeholder("p1", s, d1);
  auto p2 = F::placeholder("p2", s, d2);
  auto add = F::add("add1", p1, p2);

  // auto input = dnn::make_tensor(1, 28, 28, 1);

  return 0;
}