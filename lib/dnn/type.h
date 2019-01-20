#pragma once
#include <cinttypes>
#include <vector>
#include "dnn/arrayref.h"

namespace dnn {

/// declare common primitive types
using DEFAULT_DTYPE = float;
using Index = std::uint32_t;
using Shape = std::vector<Index>;
using Data = std::vector<DEFAULT_DTYPE>;

/// Register some aliases to save typing long charactors.
class Type;
using TypePtr = const Type*;

constexpr unsigned MAX_TENSOR_DIMS = 8;

/// Classes that represents a specific data type usually used in DNN.
enum class DataTy : std::uint32_t
{
  F32,
  I32,
  Idx
};
static constexpr DataTy DefaultDataTy = DataTy::F32;

/// Struct to deduce the data type from a given DataTy object
template <DataTy DT>
struct MetaDataTy
{
  using type = void;
};

/// @{
template <>
struct MetaDataTy<DataTy::F32>
{
  using type = float;
};

template <>
struct MetaDataTy<DataTy::I32>
{
  using type = std::int32_t;
};

template <>
struct MetaDataTy<DataTy::Idx>
{
  using type = std::size_t;
};

/// @}

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
  DataTy dtype_ = DefaultDataTy;
  /// shape of tensor of this type by default it's scalar
  Shape shape_ = {1};
  /// order of the tensor of this type by default it's NHWC
  OrderTy order_ = DefaultOrder;

public:
  /// Create default type as a scalar by default.
  Type() = default;
  ~Type() = default;
  /// Create the default type as a scalar.
  explicit Type(DataTy dtype) : dtype_{dtype} {}
  /// Create the given type with the given dimensions.
  Type(DataTy dtype, ArrayRef<Index> shape) : dtype_{dtype}, shape_{shape} {}

  Type operator=(const Type ty)
  {
    dtype_ = ty.dtype();
    shape_ = ty.shape();
    return *(this);
  }

  /// Get the dimension
  DataTy dtype() const { return dtype_; }

  /// Get the data dimension
  const Shape& shape() const { return shape_; }

  /// Return true the runtime dtype object matches this type object.
  bool is_type(DataTy dtype) const { return dtype_ == dtype; }

  /// Return true the templated parameter matches this type object.
  template <typename T>
  bool is_type() const
  {
    return IsType<T>(dtype_);
  }

  /// Return true by comparing templated parameter and each data type
  /// respectively.
  template <typename T>
  static bool IsType(DataTy dtype)
  {
    switch (dtype) {
      case DataTy::F32:
        return std::is_same<T, float>::value;
      case DataTy::I32:
        return std::is_same<T, std::int32_t>::value;
      case DataTy::Idx:
        return std::is_same<T, Index>::value;
      default:
        EXCEPTION("The given data type is not supported...");
    }
  }

  /// Internal functions that compare types
  /// @{
  inline bool is_same_dtype(const Type& x) const { return dtype_ == x.dtype_; }

  inline bool is_same_dims(const Type& x) const { return shape_ == x.shape_; }
  inline bool is_same(const Type& x) const
  {
    return is_same_dtype(x) && is_same_dims(x);
  }
  inline bool operator==(const Type& x) const { return is_same(x); }
  /// @}

  /// A function that generates scalar type with the given data type.
  template <typename T>
  static Type FromPrimitive(ArrayRef<size_t> dims)
  {
    return Type(Type::GetDataTy<T>(), dims);
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
    if (std::is_same<T, float>::value)
      return DataTy::F32;
    else if (std::is_same<T, std::int32_t>::value)
      return DataTy::I32;
    else if (std::is_same<T, Index>::value)
      return DataTy::Idx;
    else
      EXCEPTION("The given data type is not supported...");
  }

  static size_t SizeOfDataTy(DataTy dtype)
  {
    switch (dtype) {
      case DataTy::F32:
        return sizeof(DeclDataTy<DataTy::F32>);
      case DataTy::I32:
        return sizeof(DeclDataTy<DataTy::I32>);
      case DataTy::Idx:
        return sizeof(DeclDataTy<DataTy::Idx>);
      default:
        EXCEPTION("The given data type is not supported...");
    }
  }
};

enum class NodeTy : std::uint32_t
{
  Placeholder,
  Add,
  Sub,
  Mul,
  Div,
};

enum class HostTy : std::uint32_t
{
  X86,
  Arm32,
  Arm64,
};

enum class DeviceTy : std::uint32_t
{
  Generic,
  X86,
  Arm32,
  Arm64,
  Geforce,
  Vega,
  Radeon,
};

} // namespace dnn