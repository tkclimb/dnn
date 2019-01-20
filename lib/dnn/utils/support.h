#include <iostream>
#include "dnn/type.h"

namespace dnn {

/// to_string
std::string to_string(const NodeTy& ntype);
std::string to_string(const TargetTy& target);
std::string to_string(const DeviceTy& device);

class Tensor;

/// print
void print(const Tensor&);
void print(const DeviceTy&);

} // namespace dnn