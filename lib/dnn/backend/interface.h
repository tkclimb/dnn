#pragma once
#include "dnn/node.h"
#include "dnn/type.h"
#include "dnn/utils/checking.h"

namespace dnn {

class Backend
{
public:
  Backend() = default;
  ~Backend() = default;

  template <typename Node, DeviceTy Device>
  void forward(Node&);

  template <typename Node>
  void dispatch_forward(Node& node)
  {
    switch (node.device()) {
      case DeviceTy::Generic:
        forward<Node, DeviceTy::Generic>(node);
        break;
      default:
        EXCEPTION("This device is not supported....");
    }
  }
};

} // namespace dnn