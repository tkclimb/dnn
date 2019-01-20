/**
 * Copyright (c) 2018 by Contributers
 * @brief This contains TensorStorage implementation.
 *
 * @file tensor_storage.h
 * @author tkclimb
 * @date 2018-10-26
 */
#ifndef DNN_TENSOR_TENSOR_STORAGE_H
#define DNN_TENSOR_TENSOR_STORAGE_H

#include "dnn/arrayref.h"
#include "dnn/type.h"
#include "dnn/utils/checking.h"

#include <cstdlib>
#include <memory>
#include <numeric>

namespace dnn {

class TensorStorage final
{
public:
  /// Pointer to the contained data, this could be any type.
  void* data_ = nullptr;

  /// The number of contained elems.
  Index elems_ = 0;

  /// The data type.
  DataTy dtype_ = DefaultDataTy;

public:
  TensorStorage() = default;
  ~TensorStorage() { free(); };
  TensorStorage(TensorStorage&) = delete;
  TensorStorage(TensorStorage&&) = delete;
  TensorStorage(const TensorStorage&) = delete;
  TensorStorage(const TensorStorage&&) = delete;

  /// Get low pointer for optimized computation.
  /// This should be handled carefully.
  template <typename T>
  T* data()
  {
    if (Type::IsType<T>(dtype_)) {
      if (!is_init()) {
        EXCEPTION("This hasn't been set any data yet...");
      }
      return reinterpret_cast<T*>(data_);
    } else {
      EXCEPTION("The given type differs from the currently contained...");
    }
  }

  /// init the memory space
  template <typename T>
  void init(Index elems)
  {
    free();
    elems_ = elems;
    size_t nbytes = elems * sizeof(T);
    T* allocated = reinterpret_cast<T*>(std::malloc(nbytes));
    data_ = reinterpret_cast<void*>(allocated);
    dtype_ = Type::GetDataTy<T>();
  }

  /// Copy the given data as a specific type.
  template <typename T>
  TensorStorage& copy(ArrayRef<T> data)
  {
    free();
    elems_ = data.size();
    size_t nbytes = elems() * sizeof(T);
    T* allocated = reinterpret_cast<T*>(std::malloc(nbytes));
    std::memcpy(allocated, data.data(), nbytes);

    data_ = reinterpret_cast<void*>(allocated);
    dtype_ = Type::GetDataTy<T>();
    return *this;
  }

  /// Get the number of elems.
  inline Index elems() const { return elems_; }

  /// Get the array size.
  inline Index sizeof_data() const { return Type::SizeOfDataTy(dtype_); }

private:
  /// Check it's initialized yet or not.
  inline bool is_init() const { return data_ != nullptr; }

  /// Check its data is initialized or not.
  const void free() const { std::free(data_); }
};

} // namespace dnn

#endif