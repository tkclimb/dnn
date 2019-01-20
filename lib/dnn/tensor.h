#pragma once
#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "dnn/type.h"

namespace dnn {

static const constexpr size_t MAX_DIMS = 4;

class Tensor
{
  using T = DEFAULT_DTYPE;

private:
  Shape shape_ = {1};
  Shape extents_ = {1};
  T* data_ = nullptr;

public:
  Tensor(const Shape& shape)
    : shape_{shape}
    , extents_{derive_extents(shape)}
    , data_{reinterpret_cast<T*>(std::malloc(elems() * sizeof(T)))}
  {}

  Tensor(const Shape& shape, const Data& data) : Tensor(shape)
  {
    copy_from_vec(data);
  }

  virtual ~Tensor() { free(); };

  Tensor(Tensor&) = default;
  Tensor(const Tensor&) = default;
  Tensor(Tensor&&) = default;
  Tensor(const Tensor&&) = delete;

  Tensor& copy_from_vec(const Data& data)
  {
    size_t given_elems = data.size();
    assert(given_elems == elems());
    free();
    size_t nbytes = given_elems * sizeof(T);
    T* allocated = reinterpret_cast<T*>(std::malloc(nbytes));
    std::memcpy(allocated, data.data(), nbytes);
    data_ = allocated;
    return *this;
  }

  T* data() { return data_; }

  inline Shape shape() const { return shape_; }

  inline Index rank() const { return shape_.size(); }

  inline Index elems() const { return calc_elems(shape_); }

  inline T operator[](const Index idx) const noexcept { return data_[idx]; }

  inline T at(const Shape& idxs) const noexcept
  {
    size_t idx =
      std::inner_product(extents_.begin(), extents_.end(), idxs.begin(), 0);
    return (*this)[idx];
  }

private:
  const void free() const { std::free(data_); }

  inline const Index calc_elems(const Shape& shape) const noexcept
  {
    return std::accumulate(shape.begin(), shape.end(), 1,
                           std::multiplies<Index>());
  }

  Shape derive_extents(const Shape& shape) const
  {
    Shape ext(shape.size());
    size_t cur_ext = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      ext[i] = cur_ext;
      cur_ext *= shape[i];
    }
    return ext;
  }
};

} // namespace dnn
