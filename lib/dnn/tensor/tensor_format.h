#ifndef DNN_TENSOR_TENSORFORMAT_H
#define DNN_TENSOR_TENSORFORMAT_H
#include <cstdint>

namespace dnn {

enum class TensorMode : uint8_t
{
  Dense,
  Quantized,
};

enum class MajorType : uint8_t
{
  Row,
  Column,
};

struct TensorFormat final
{
  TensorFormat() : mode{TensorMode::Dense}, major{MajorType::Row} {}
  TensorMode mode;
  MajorType major;
};

} // namespace dnn

#endif