/**
 * Copyright (c) 2018 by Contributers
 * @brief This contains Datatype implementation.
 *
 * @file tensor.h
 * @author tkclimb
 * @date 2018-10-16
 */
#ifndef DNN_TENSOR_TENSOR_H
#define DNN_TENSOR_TENSOR_H

#include "dnn/arrayref.h"
#include "dnn/name_manager.h"
#include "dnn/tensor/tensor_format.h"
#include "dnn/tensor/tensor_storage.h"
#include "dnn/type.h"
#include "dnn/utils/checking.h"

#include <memory>
#include <numeric>
#include <string>

namespace dnn {

template <typename T>
class TensorAccessor;

/**
 * @brief This class represents tensor structure. In fact this class just
 * contains essential information only like shape, name and format. This
 * doesn't have actual data inside, so that the we can easilly move the data
 * to or from others.
 *
 */
class Tensor final
{
public:
  template <typename T>
  friend class TensorAccessor;

private:
  /// Tensor's name.
  std::string name_;

  /// Owner node of this tensor

  /// Type of tensor's data.
  Type type_;

  /// Format of tensor.
  TensorFormat fmt_;

  /// Data storage of tensor that can contain variant data types.
  TensorStorage storage_{type_};

public:
  ~Tensor() = default;
  Tensor(Tensor&) = delete;
  Tensor(Tensor&&) = delete;
  Tensor(const Tensor&) = delete;
  Tensor(const Tensor&&) = delete;

  /// Create a scalar with value 0 without any given name
  Tensor() : name_{NameManager::MakeUnique("Tensor")}, type_{} {};

  /// Create a scalar with the given type
  explicit Tensor(const Type type)
    : name_{NameManager::MakeUnique("Tensor")}, type_{type} {};

  /// Create a scalar with 0 and the given name
  explicit Tensor(const std::string& name) : name_{name}, type_{} {}

  /// Create a tensor with the given shape
  explicit Tensor(DataTy dataty, ArrayRef<Index> shape)
    : name_{NameManager::MakeUnique("Tensor")}, type_{dataty, shape} {};

  /// Create a tensor with the given name, dataty and shape.
  explicit Tensor(const std::string& name, const DataTy dataty,
                  ArrayRef<Index> shape)
    : name_{name}, type_{dataty, shape} {};

  /// Get the name of this tensor.
  inline const std::string& name() const { return name_; }

  /// Get the type of this tensor.
  inline const Type& type() const { return type_; }

  /// Get the data type of this tensor.
  inline DataTy dataty() const { return type_.dataty(); }

  /// Get the shape of this tensor.
  inline const Shape& shape() const { return type_.shape(); }

  /// Get the number of elements in this tensor.
  inline Index elems() const { return type_.elems(); }

  /// Get the rank of this tensor.
  inline Index rank() const { return type_.ndims(); }

  /// Return accessor with itself.
  template <DataTy DT>
  auto get_access() &;

  /// Return const accessor with itself.
  template <DataTy DT>
  auto get_access() const&;
};

} // namespace dnn

/// TODO: Fix this way to include.
/// This must be included in this location.
#include "dnn/tensor/tensor_accessor.h"

namespace dnn {

template <DataTy DT>
auto Tensor::get_access() &
{
  if (type_.is_type(DT)) {
    return TensorAccessor<DeclDataTy<DT>>(this);
  } else {
    EXCEPTION("The given type differs from the type intended...");
  }
};

template <DataTy DT>
auto Tensor::get_access() const&
{
  if (type_.is_type(DT)) {
    return TensorAccessor<DeclDataTy<DT>>(const_cast<Tensor*>(this));
  } else {
    EXCEPTION("The given type differs from the type intended...");
  }
};

} // namespace dnn

#endif
