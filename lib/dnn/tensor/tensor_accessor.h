/**
 * Copyright (c) 2018 by Contributers
 * @brief This contains Datatype implementation.
 *
 * @file tensor.h
 * @author tkclimb
 * @date 2018-10-16
 */
#ifndef DNN_TENSOR_TENSOR_ACCESSOR_H
#define DNN_TENSOR_TENSOR_ACCESSOR_H

#include "dnn/arrayref.h"
#include "dnn/tensor/tensor.h"
#include "dnn/tensor/tensor_storage.h"
#include "dnn/utils/checking.h"

#include <memory>
#include <numeric>
#include <string>

namespace dnn {

/**
 * @brief This class helps user to access the data in tensor, which
 * validate the index, shape, required type or etc.
 *
 * @tparam T type of the element in the tensor
 */
template <typename T>
class TensorAccessor final
{
private:
  /// Pointer to a tensor
  Tensor* tensor_;

  /// The number of extents in each shape maybe used with index assess.
  Shape extents_;

public:
  /// Create a scaler tensor storage.
  TensorAccessor() = delete;
  ~TensorAccessor() = default;
  TensorAccessor(TensorAccessor&) = delete;
  TensorAccessor(const TensorAccessor&) = delete;
  TensorAccessor(const TensorAccessor&&) = delete;

  TensorAccessor(TensorAccessor&& other)
    : tensor_{other.tensor_}, extents_{other.extents_}
  {}

  /// Create empty tensor with the given shape.
  explicit TensorAccessor(Tensor* tensor)
    : tensor_{tensor}, extents_{derive_extents(tensor_->shape())}
  {}

  /// Insert the given data into itself.
  TensorAccessor& operator=(ArrayRef<T> data)
  {
    validate_num_elems(data);
    tensor_->storage_.copy(data);
    return *this;
  }

  /// Init means just allocates the memory, not clear at all.
  void init() { tensor_->storage_.init<T>(elems()); }

  /// Get contained data from storage
  inline T* data() { return tensor_->storage_.data<T>(); }

  /// Get the data element at the given index.
  inline T& operator[](const size_t idx)
  {
    validate_idx(idx);
    return data()[idx];
  }

  /// Get the data element at the given indices.
  inline T& at(ArrayRef<Index> idxs)
  {
    size_t idx =
      std::inner_product(extents_.begin(), extents_.end(), idxs.begin(), 0);
    return (*this)[idx];
  }

  /// Get the data element with index via variadic templates from storage
  template <typename... DIMS>
  inline T& at(DIMS... indices)
  {
    ArrayRef<Index> idxs{indices...};
    return at(idxs);
  }

  /// Get the shape of this tensor
  inline const Shape& shape() const { return tensor_->shape(); }

  /// Get number of elements in this tensor.
  inline Index elems() const { return tensor_->elems(); }

private:
  /// Check the given shape' size is valid or not.
  void validate_shape_size(ArrayRef<Index> shape) const
  {
    ASSERT_STR(shape.size() == tensor_->shape().size(),
               "The number of given shape(" + std::to_string(shape.size()) +
                 ") differs from the expected(" +
                 std::to_string(tensor_->shape().size()) + ")...")
  }

  /// Check the given shape is valid or not.
  void validate_shape(ArrayRef<Index> shape) const
  {
    validate_shape_size(shape);
    bool extents_valid = true;
    for (int i = 0; i < tensor_->shape().size(); ++i) {
      extents_valid &= tensor_->shape()[i] == shape[i];
    }

    ASSERT(extents_valid,
           "The given shape' extents differs from the expected... ");
  }

  /// Check the given shape when reshape.
  void validate_reshaped_shape(ArrayRef<Index> shape) const
  {
    size_t num_given_data_elems =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    ASSERT(num_given_data_elems == elems(),
           "The given data size differs from the expected...");
  }

  /// Check the given indices' size and extents are valid or not.
  void validate_idx(const size_t idx) const
  {
    ASSERT(idx <= elems(), "The given index excesses the expected... ");
  }

  /// Check the given indices' size and extents are valid or not.
  void validate_idxs(ArrayRef<Index> idxs) const
  {
    bool size_valid = idxs.size() == tensor_->shape().size();
    bool extents_valid = true;
    for (int i = 0; i < tensor_->shape().size(); ++i) {
      extents_valid &= idxs[i] < tensor_->shape()[i];
    }

    ASSERT(size_valid && extents_valid,
           "The given indices size or the extent differ from the expected... ");
  }

  void validate_num_elems(ArrayRef<T> data)
  {
    ASSERT(data.size() == elems(),
           "The given data size differs from the expected...");
  }

  /// Derive each extent, which would be used in ctor.
  Shape derive_extents(ArrayRef<Index> shape) const
  {
    validate_shape_size(shape);
    Shape ext(shape.size());
    size_t current_ext = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      ext[i] = current_ext;
      current_ext *= shape[i];
    }
    return ext;
  }
};

} // namespace dnn

#endif
