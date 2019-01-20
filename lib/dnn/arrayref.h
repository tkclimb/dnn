/**
 * Copyright (c) 2018 by Contributers
 * @brief This contains ArrayRef implementation.
 * This implementation is simple version of llvm::ArrayRef.
 * http://llvm.org/doxygen/ArrayRef_8h_source.html
 *
 * @file arrayref.h
 * @author tkclimb
 * @date 2018-10-26
 */
#ifndef DNN__ARRAYREF_H
#define DNN__ARRAYREF_H

#include "dnn/utils/checking.h"

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

namespace dnn {

template <typename T>
class ArrayRef
{
public:
  using iterator = const T *;
  using const_iterator = const T *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using size_type = size_t;

private:
  /// The head pointer of the array
  const T *data_ = nullptr;

  /// The number of elements.
  size_type length_ = 0;

public:
  /// Construct an empty array
  ArrayRef() = default;

  /// Destructor an empty array
  ~ArrayRef() = default;

  /// Construct an array from a single element.
  ArrayRef(const T &elem) : data_(&elem), length_(1) {}

  /// Construct an array from a pointer and length.
  ArrayRef(const T *data, size_t length) : data_(data), length_(length) {}

  /// Construct an array from a range.
  ArrayRef(const T *begin, const T *end) : data_(begin), length_(end - begin) {}

  /// Construct an array from a std::vector.
  template <typename A>
  ArrayRef(const std::vector<T, A> &vec)
    : data_(vec.data()), length_(vec.size())
  {}

  /// Construct an array from a std::array
  template <size_t N>
  constexpr ArrayRef(const std::array<T, N> &ary)
    : data_(ary.data()), length_(N)
  {}

  /// Construct an array from a C array.
  template <size_t N>
  constexpr ArrayRef(const T (&ary)[N]) : data_(ary), length_(N)
  {}

  /// Construct an array from a std::initializer_list.
  ArrayRef(const std::initializer_list<T> &list)
    : data_(list.begin() == list.end() ? (T *)nullptr : list.begin())
    , length_(list.size())
  {}

  /// Construct an ArrayRef<const T*> from ArrayRef<T*>.
  /// This uses SFINAE ensure that only ArrayRefs of pointers can be
  /// converted.
  template <typename U>
  ArrayRef(
    const ArrayRef<U *> &ary,
    typename std::enable_if<
      std::is_convertible<U *const *, T const *>::value>::type * = nullptr)
    : data_(ary.data()), length_(ary.size())
  {}

  /// Construct an ArrayRef<const T*> from std::vector<T*>.
  /// This uses SFINAE to ensure that only vectors of pointers can be converted.
  template <typename U, typename A>
  ArrayRef(const std::vector<U *, A> &vec,
           typename std::enable_if<
             std::is_convertible<U *const *, T const *>::value>::type * = 0)
    : data_(vec.data()), length_(vec.size())
  {}

  iterator begin() const { return data_; }
  iterator end() const { return data_ + length_; }

  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  reverse_iterator rend() const { return reverse_iterator(begin()); }

  /// Check if the array is empty.
  bool empty() const { return length_ == 0; }

  const T *data() const { return data_; }

  /// Get the array size.
  size_t size() const { return length_; }

  /// Get the first element.
  const T &front() const
  {
    ASSERT(!empty(), "This array is empty...");
    return data_[0];
  }

  /// Get the last element.
  const T &back() const
  {
    ASSERT(!empty(), "This array is empty...");
    return data_[length_ - 1];
  }

  // Allocate copy in Allocator and return ArrayRef<T> to it.
  template <typename Allocator>
  ArrayRef<T> copy(Allocator &alloc)
  {
    T *buff = alloc.template Allocate<T>(length_);
    std::uninitialized_copy(begin(), end(), buff);
    return ArrayRef<T>(buff, length_);
  }

  /// Check for element-wise equality.
  bool equals(ArrayRef<T> rhs) const
  {
    if (length_ != rhs.length_)
      return false;
    return std::equal(begin(), end(), rhs.begin());
  }

  /// Get an element with bracket accessor.
  const T &operator[](size_t idx) const
  {
    ASSERT(idx < length_, "The given index excess its data size...");
    return data_[idx];
  }

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type &
  operator=(U &&Temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type &
  operator=(std::initializer_list<U>) = delete;

  /// Converet to a std::vector
  operator std::vector<T>() const
  {
    return std::vector<T>(data_, data_ + length_);
  }
};

} // namespace dnn

#endif