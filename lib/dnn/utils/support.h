#include <cinttypes>
#include <iostream>

namespace dnn {

enum class DataTy : std::uint32_t;
enum class NodeTy : std::uint32_t;
enum class HostTy : std::uint32_t;
enum class DeviceTy : std::uint32_t;

/// define to_string
#define TO_STRING(T) to_string(T);

std::string to_string(const DataTy& dataty);
std::string to_string(const NodeTy& nodety);
std::string to_string(const HostTy& hostty);
std::string to_string(const DeviceTy& devty);

class Tensor;

/// print
void print(const Tensor&);
void print(const DeviceTy&);

} // namespace dnn