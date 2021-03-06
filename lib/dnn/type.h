#pragma once
#include <cinttypes>
#include <vector>
#include "dnn/arrayref.h"
#include "dnn/utils/support.h"

namespace dnn {

/// Register some aliases to save typing long charactors.
class Type;
using TypePtr = const Type*;

constexpr unsigned MAX_TENSOR_DIMS = 8;

namespace primitive {
using F32 = float;
using I32 = std::int32_t;
using Idx = std::uint32_t;
} // namespace primitive

/**
 * @brief User defined types
 *
 */
#define DEFINED_DATATYS(MACRO) \
  MACRO(F32)                   \
  MACRO(I32) MACRO(Idx)

#define DEFINED_NODETYS(MACRO) \
  MACRO(Placeholder)           \
  MACRO(Add)                   \
  MACRO(Sub)                   \
  MACRO(Mul)                   \
  MACRO(Matmul)

#define DEFINED_HOSTTYS(MACRO) MACRO(X86)

#define DEFINED_DEVTYS(MACRO) MACRO(Generic) MACRO(X86)

/* */

/// declare common type aliases
using DEFAULT_DATATY = primitive::F32;
using Index = primitive::Idx;
using Shape = std::vector<Index>;
using Data = std::vector<DEFAULT_DATATY>;

/// Classes that represents a specific data type usually used in DNN.
enum class DataTy : std::uint32_t
{
#define DEF(T) T,
  DEFINED_DATATYS(DEF)
#undef DEF
};
static constexpr DataTy DefaultDataTy = DataTy::F32;

#define CASE(NAME, T, MACRO) \
  case NAME::T: {            \
    MACRO(T)                 \
    break;                   \
  }

#define DEFAULT_EXCEPTION(T)                              \
  default: {                                              \
    EXCEPTION_STR(to_string(T) + " is not supported..."); \
  }

#define SWITCH_BY_DATATY(DT, MACRO) \
  switch (DT) {                     \
    CASE(DataTy, F32, MACRO)        \
    CASE(DataTy, I32, MACRO)        \
    CASE(DataTy, Idx, MACRO)        \
    DEFAULT_EXCEPTION(DT)           \
  }

enum class NodeTy : std::uint32_t
{
#define DEF(T) T,
  DEFINED_NODETYS(DEF)
#undef DEF
};

enum class HostTy : std::uint32_t
{
#define DEF(T) T,
  DEFINED_HOSTTYS(DEF)
#undef DEF
};

enum class DeviceTy : std::uint32_t
{
#define DEF(T) T,
  DEFINED_DEVTYS(DEF)
#undef DEF
};

#define SWITCH_BY_NODETY(T, MACRO)   \
  switch (T) {                       \
    CASE(NodeTy, Placeholder, MACRO) \
    CASE(NodeTy, Add, MACRO)         \
    CASE(NodeTy, Sub, MACRO)         \
    CASE(NodeTy, Mul, MACRO)         \
    CASE(NodeTy, Matmul, MACRO)      \
    DEFAULT_EXCEPTION(T)             \
  }

#define SWITCH_BY_HOSTTY(T, MACRO) \
  switch (T) {                     \
    CASE(HostTy, X86, MACRO)       \
    DEFAULT_EXCEPTION(T)           \
  }

#define SWITCH_BY_DEVTY(T, MACRO)  \
  switch (T) {                     \
    CASE(DeviceTy, Generic, MACRO) \
    CASE(DeviceTy, X86, MACRO)     \
    DEFAULT_EXCEPTION(T)           \
  }

/// Struct to deduce the data type from a given DataTy object
template <DataTy DT>
struct MetaDataTy
{
  using type = void;
};

#define DEF_META_DATATY(NAME)     \
  template <>                     \
  struct MetaDataTy<DataTy::NAME> \
  {                               \
    using type = primitive::NAME; \
  };
DEFINED_DATATYS(DEF_META_DATATY)
#undef DEF_META_DATATY

template <DataTy T>
using DeclDataTy = typename MetaDataTy<T>::type;

/// Symbols that represent some a specific order usually used in DNN.
enum class OrderTy : std::uint32_t
{
  NHWC,
  NCHW,
};
static constexpr OrderTy DefaultOrder = OrderTy::NHWC;

/// A class that represents type for ir.
/// Different from the usual type class implementation,
/// this has the dimension of the tensor as well as the data type.
class Type final
{
private:
  /// data type of tensor of this type.
  DataTy dataty_ = DefaultDataTy;
  /// shape of tensor of this type by default it's scalar
  Shape shape_ = {1};
  /// order of the tensor of this type by default it's NHWC
  OrderTy order_ = DefaultOrder;

public:
  /// Create default type as a scalar by default.
  Type() = default;
  ~Type() = default;
  /// Create the default type as a scalar.
  explicit Type(DataTy dataty) : dataty_{dataty} {}
  /// Create the given type with the given dimensions.
  Type(DataTy dataty, ArrayRef<Index> shape) : dataty_{dataty}, shape_{shape} {}

  Type operator=(const Type ty)
  {
    dataty_ = ty.dataty();
    shape_ = ty.shape();
    return *(this);
  }

  /// Get the data type
  DataTy dataty() const { return dataty_; }

  /// Get the data shape
  const Shape& shape() const { return shape_; }

  /// Get the number of elements in this tensor.
  inline Index elems() const
  {
    return std::accumulate(shape().begin(), shape().end(), 1,
                           std::multiplies<Index>());
  }

  /// Get the number of dimension
  Index ndims() const { return shape_.size(); }

  /// Return true the runtime dataty object matches this type object.
  bool is_type(DataTy dataty) const { return dataty_ == dataty; }

  /// Return true the templated parameter matches this type object.
  template <typename T>
  bool is_type() const
  {
    return IsType<T>(dataty_);
  }

  /// Return true by comparing templated parameter and each data type
  /// respectively.
  template <typename T>
  static bool IsType(DataTy dataty)
  {
#define RETURN_ISSAME(DT) return std::is_same<T, primitive::DT>::value;
    SWITCH_BY_DATATY(dataty, RETURN_ISSAME)
#undef RETURN_ISSAME
  }

  /// Internal functions that compare types
  /// @{
  inline bool is_same_dataty(const Type& x) const
  {
    return dataty_ == x.dataty_;
  }

  inline bool is_same_dims(const Type& x) const { return shape_ == x.shape_; }

  inline bool is_same(const Type& x) const
  {
    return is_same_dataty(x) && is_same_dims(x);
  }

  inline bool operator==(const Type& x) const { return is_same(x); }
  /// @}

  /// A function that generates scalar type with the given data type.
  template <typename T>
  static Type FromPrimitive(ArrayRef<Index> shape)
  {
    return Type(Type::GetDataTy<T>(), shape);
  }

  /// A function that generates scalar type with the given data type.
  template <typename T>
  static Type FromPrimitive()
  {
    return Type(Type::GetDataTy<T>(), {1});
  }

  template <typename T>
  static DataTy GetDataTy()
  {
    if (std::is_same<T, primitive::F32>::value)
      return DataTy::F32;
    else if (std::is_same<T, primitive::I32>::value)
      return DataTy::I32;
    else if (std::is_same<T, primitive::Idx>::value)
      return DataTy::Idx;
    else
      EXCEPTION("The given data type is not supported...");
  }

  static size_t SizeOfDataTy(DataTy dataty)
  {
#define RETURN_DATATY(DT) return sizeof(DeclDataTy<DataTy::DT>);
    SWITCH_BY_DATATY(dataty, RETURN_DATATY)
#undef RETURN_DATATY
  }
};

} // namespace dnn