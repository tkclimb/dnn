#pragma once
#include <memory>
#include <string>
#include <vector>
#include "dnn/context.h"
#include "dnn/name_manager.h"
#include "dnn/tensor/tensor.h"
#include "dnn/type.h"
#include "dnn/utils/checking.h"
#include "dnn/utils/support.h"

namespace dnn {

class Node;
using NodePtr = std::shared_ptr<Node>;
using NodeVec = std::vector<NodePtr>;
class Visitor;
using VisitFunc = std::function<void(const Node*)>;

class Node
{
protected:
  NodeTy nodety_;
  NodeVec inputs_;
  Tensor tensor_;
  Tensor grad_;
  Context ctx_;
  std::string name_;

public:
  friend class Backend;

  Node(const NodeTy nodety, NodeVec& inputs, const Type& type,
       const Context& ctx)
    : nodety_{nodety}
    , inputs_{inputs}
    , tensor_{type}
    , grad_{type}
    , ctx_{ctx}
    , name_{NameManager::MakeUnique(to_string(nodety))}
  {}

  Node(const NodeTy nodety, NodeVec&& inputs, const Type& type,
       const Context& ctx)
    : nodety_{nodety}
    , inputs_{inputs}
    , tensor_{type}
    , grad_{type}
    , ctx_{ctx}
    , name_{NameManager::MakeUnique(to_string(nodety))}
  {}

  virtual ~Node() = default;

  /// getter functions.
  inline NodeTy nodety() const { return nodety_; };
  inline const std::string& name() const { return name_; }
  inline const Tensor& tensor() const { return tensor_; }
  inline Tensor& tensor() { return tensor_; }

  inline const Type& type() const { return tensor_.type(); }
  inline DataTy dataty() const { return tensor_.dataty(); }
  inline Shape shape() const { return tensor_.shape(); }
  inline Index elems() const { return tensor_.elems(); }

  /// setter functions.
  inline void set_name(const std::string& name) { name_ = name; }

  inline HostTy hostty() const { return ctx_.hostty(); }
  inline DeviceTy devty() const { return ctx_.devty(); }

  virtual void forward() = 0;
  virtual void backward() = 0;

  virtual void accept(Visitor*) const = 0;
  virtual void accept(Visitor*, const VisitFunc&, const VisitFunc&) const = 0;
};

template <typename T>
class VisitableNode : public Node
{
public:
  using Node::Node;
  void accept(Visitor*) const;
  void accept(Visitor*, const VisitFunc&, const VisitFunc&) const;
};

#define DEFINE_BASIC_NODE_FUNCS \
  void forward() override;      \
  void backward() override;

class Placeholder : public VisitableNode<Placeholder>
{
private:
  bool init_ = false;

public:
  const Tensor* ref;

  Placeholder(const Tensor& tensor, const Context& ctx)
    : VisitableNode(NodeTy::Placeholder, {}, tensor.type(), ctx), ref{&tensor}
  {}
  DEFINE_BASIC_NODE_FUNCS
};

#define DEFINE_ARITHMETIC_OP(NAME)                                   \
  class NAME : public VisitableNode<NAME>                            \
  {                                                                  \
  public:                                                            \
    NAME(NodePtr a, NodePtr b, const Type& type, const Context& ctx) \
      : VisitableNode{NodeTy::NAME, {a, b}, type, ctx}               \
    {}                                                               \
    inline NodePtr a() { return inputs_[0]; }                        \
    inline NodePtr b() { return inputs_[1]; }                        \
    inline const NodePtr a() const { return inputs_[0]; }            \
    inline const NodePtr b() const { return inputs_[1]; }            \
    DEFINE_BASIC_NODE_FUNCS                                          \
  };

DEFINE_ARITHMETIC_OP(Add)
DEFINE_ARITHMETIC_OP(Sub)
DEFINE_ARITHMETIC_OP(Mul)
DEFINE_ARITHMETIC_OP(Matmul)

template <typename T>
const Type& infer_type(NodePtr);
template <typename T>
const Type& infer_type(NodePtr, NodePtr);

} // namespace dnn
