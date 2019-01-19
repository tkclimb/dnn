#pragma once
#include <cassert>
#include <exception>
#include <stdexcept>

#include "dnn/utils/logger.h"

namespace dnn {

/// Check the given x. if it's false, show some massage
#define CHECK(x)                                                           \
  if (!(x)) {                                                              \
    ::dnn::Logger::Message("CHECK", "Failed: " #x, __FILE__, __FUNCTION__, \
                           __LINE__);                                      \
  }

#define CHECK_LT(a, b) CHECK((a) < (b))
#define CHECK_GT(a, b) CHECK((a) > (b))
#define CHECK_LE(a, b) CHECK((a) <= (b))
#define CHECK_GE(a, b) CHECK((a) >= (b))
#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_NE(a, b) CHECK((a) != (b))

#define CHECK_NOTNULL(x)                                                 \
  if ((x) != NULL && (x) != nullptr) {                                   \
  ::dnn::Logger::Message("CHECK", "Failed: " #x, __FILE__, __FUNCTION__, \
                         __LINE__)

/// Check the given x. if it's false, show some massage
#define ASSERT(x, msg)                                                        \
  if (!(x)) {                                                                 \
    ::dnn::Logger::Message("ASSERT", #msg, __FILE__, __FUNCTION__, __LINE__); \
    assert(x);                                                                \
  }

#define EXCEPTION(msg)                                                         \
  ::dnn::Logger::Message("EXCEPTION", #msg, __FILE__, __FUNCTION__, __LINE__); \
  assert(false);

/// Almost ASSERT but with string message
#define ASSERT_STR(x, msg)                                                   \
  if (!(x)) {                                                                \
    ::dnn::Logger::Message("ASSERT", msg, __FILE__, __FUNCTION__, __LINE__); \
    assert(x);                                                               \
  }

/// Almost EXCEPTION but with string message
#define EXCEPTION_STR(msg)                                                    \
  ::dnn::Logger::Message("EXCEPTION", msg, __FILE__, __FUNCTION__, __LINE__); \
  assert(false);

} // namespace dnn