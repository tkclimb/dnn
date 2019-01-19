#include "dnn/tensor.h"

namespace dnn {

void print_tensor(const Tensor& tensor)
{
  std::cout << "Float ";
  print_tensor(tensor, 0, 0);
  std::cout << std::endl;
}

Index print_tensor(const Tensor& tensor, Index rank, Index idx)
{
  std::string indent = "";
  for (size_t i = 0; i < rank; i++) { indent += "  "; }

  auto cur_elems = tensor.shape()[rank];

  if (rank == tensor.rank() - 1) {
    std::string str = "";
    str += indent + "{ ";

    for (size_t i = 0; i < cur_elems; i++) {
      str += std::to_string(tensor[idx++]);
      if (i != cur_elems - 1) {
        str += ", ";
      }
    }

    str += " }\n";
    std::cout << str;
    return idx;
  } else {
    for (size_t i = 0; i < cur_elems; i++) {
      std::cout << indent + "{\n";

      idx = print_tensor(tensor, rank + 1, idx);

      std::string str = "";
      str += indent + "}";
      if (i != cur_elems - 1) {
        str += ",";
      }
      std::cout << str;
    }
    return idx;
  }
}

} // namespace dnn