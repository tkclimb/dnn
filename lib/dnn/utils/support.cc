#include "dnn/utils/support.h"
#include "dnn/tensor.h"
#include "dnn/type.h"
#include "dnn/utils/checking.h"

namespace dnn {

#define NODETYPE_STR(NODETY) #NODETY

#define RETURN_NODETYPE_STR(NODE)      \
  case NodeTy::NODE:                   \
    return NODETYPE_STR(NodeTy::NODE); \
    break;

std::string to_string(const NodeTy& ntype)
{
  switch (ntype) {
    RETURN_NODETYPE_STR(Placeholder)
    RETURN_NODETYPE_STR(Add)
    RETURN_NODETYPE_STR(Sub)
    default:
      EXCEPTION("This node type is not supported...")
      break;
  }
}
#undef NODETYPE_STR
#undef RETURN_NODETYPE_STR

std::string to_string(const HostTy& target)
{
  switch (target) {
    case HostTy::X86:
      return "HostTy::X86";
      break;
    case HostTy::Arm32:
      return "HostTy::Arm32";
      break;
    case HostTy::Arm64:
      return "HostTy::Arm64";
      break;
    default:
      EXCEPTION("This target type is not supported...")
      break;
  }
}

std::string to_string(const DeviceTy& device)
{
  switch (device) {
    case DeviceTy::Generic:
      return "DeviceTy::Generic";
      break;
    case DeviceTy::X86:
      return "DeviceTy::X86";
      break;
    case DeviceTy::Geforce:
      return "DeviceTy::Geforce";
      break;
    default:
      EXCEPTION("This device type is not supported...")
      break;
  }
}

void print(const DeviceTy& device)
{
  std::cout << to_string(device) << std::endl;
}

Index print(const Tensor& tensor, Index rank, Index idx)
{
  std::string indent = "";
  for (size_t i = 0; i < rank; ++i) { indent += "  "; }

  auto cur_elems = tensor.shape()[rank];

  if (rank == tensor.rank() - 1) {
    std::string str = "";
    str += indent + "{ ";

    for (size_t i = 0; i < cur_elems; ++i) {
      str += std::to_string(tensor[idx++]);
      if (i != cur_elems - 1) {
        str += ", ";
      }
    }

    str += " }\n";
    std::cout << str;
    return idx;
  } else {
    for (size_t i = 0; i < cur_elems; ++i) {
      std::cout << indent + "{\n";

      idx = print(tensor, rank + 1, idx);

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

void print(const Tensor& tensor)
{
  std::cout << "Float ";
  print(tensor, 0, 0);
  std::cout << std::endl;
}

} // namespace dnn