#include "dnn/utils/support.h"
#include "dnn/tensor/tensor.h"
#include "dnn/type.h"
#include "dnn/utils/checking.h"

namespace dnn {

std::string to_string(const DataTy& dataty)
{
#define DATATY_STR(DT) #DT;
#define RETURN_DATATY_STR(DT) return DATATY_STR(DataTy::DT);
  SWITCH_BY_DATATY(dataty, RETURN_DATATY_STR)
#undef DATATY_STR
#undef RETURN_DATATY_STR
}

std::string to_string(const NodeTy& nodety)
{
#define NODETY_STR(NT) #NT;
#define RETURN_NODETY_STR(NT) return NODETY_STR(NodeTy::NT);
  SWITCH_BY_NODETY(nodety, RETURN_NODETY_STR)
#undef NODETY_STR
#undef RETURN_NODETY_STR
}

std::string to_string(const HostTy& hostty)
{
#define HOSTTY_STR(HT) #HT;
#define RETURN_HOSTTY_STR(HT) return HOSTTY_STR(HostTy::HT);
  SWITCH_BY_HOSTTY(hostty, RETURN_HOSTTY_STR)
#undef HOSTTY_STR
#undef RETURN_HOSTTY_STR
}

std::string to_string(const DeviceTy& device)
{
#define DEVTY_STR(DEVT) #DEVT;
#define RETURN_DEVTY_STR(DEVT) return DEVTY_STR(DeviceTy::DEVT);
  SWITCH_BY_DEVTY(device, RETURN_DEVTY_STR)
#undef DEVTY_STR
#undef RETURN_DEVTY_STR
}

template <DataTy DT>
void print(const DeviceTy& device)
{
  std::cout << to_string(device) << std::endl;
}

template <DataTy DT>
Index print(const Tensor& tensor, Index rank, Index idx)
{
  std::string indent = "";
  auto acc = tensor.get_access<DT>();
  for (size_t i = 0; i < rank; ++i) { indent += "  "; }

  auto cur_elems = tensor.shape()[rank];

  if (rank == tensor.rank() - 1) {
    std::string str = "";
    str += indent + "{ ";

    for (size_t i = 0; i < cur_elems; ++i) {
      str += std::to_string(acc[idx++]);
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

      idx = print<DT>(tensor, rank + 1, idx);

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

#define PRINT_TENSOR(DT) print<DataTy::DT>(tensor, 0, 0);

void print(const Tensor& tensor)
{
  std::cout << to_string(tensor.dataty()) << " ";
  SWITCH_BY_DATATY(tensor.dataty(), PRINT_TENSOR)
  std::cout << std::endl;
}

#undef PRINT_TENSOR

} // namespace dnn