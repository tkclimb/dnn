#include "dnn/utils/support.h"
#include "dnn/tensor/tensor.h"
#include "dnn/type.h"
#include "dnn/utils/checking.h"

namespace dnn {

std::string to_string(const DataTy& dtype)
{
#define DTYPE_STR(DT) #DT;
#define RETURN_DTYPE_STR(DT) return DTYPE_STR(DataTy::DT);
  SWITCH_BY_DTYPE(dtype, RETURN_DTYPE_STR)
#undef DTYPE_STR
#undef RETURN_DTYPE_STR
}

std::string to_string(const NodeTy& ntype)
{
#define NTYPE_STR(NT) #NT;
#define RETURN_NTYPE_STR(NT) return NTYPE_STR(NodeTy::NT);
  SWITCH_BY_NTYPE(ntype, RETURN_NTYPE_STR)
#undef NTYPE_STR
#undef RETURN_NTYPE_STR
}

std::string to_string(const HostTy& htype)
{
#define HTYPE_STR(HT) #HT;
#define RETURN_HTYPE_STR(HT) return HTYPE_STR(HostTy::HT);
  SWITCH_BY_HTYPE(htype, RETURN_HTYPE_STR)
#undef HTYPE_STR
#undef RETURN_HTYPE_STR
}

std::string to_string(const DeviceTy& device)
{
#define DEVTYPE_STR(DEVT) #DEVT;
#define RETURN_DEVTYPE_STR(DEVT) return DEVTYPE_STR(DeviceTy::DEVT);
  SWITCH_BY_DEVTYPE(device, RETURN_DEVTYPE_STR)
#undef DEVTYPE_STR
#undef RETURN_DEVTYPE_STR
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
  std::cout << to_string(tensor.dtype()) << " ";
  SWITCH_BY_DTYPE(tensor.dtype(), PRINT_TENSOR)
  std::cout << std::endl;
}

#undef PRINT_TENSOR

} // namespace dnn