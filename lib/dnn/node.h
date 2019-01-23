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
  NodeTy ntype_;
  NodeVec inputs_;
  Tensor tensor_;
  Tensor grad_;
  Context ctx_;
  std::string name_;

public:
  friend class Backend;

  Node(const NodeTy ntype, NodeVec& inputs, const Type& type,
       const Context& ctx)
    : ntype_{ntype}
    , inputs_{inputs}
    , tensor_{type}
    , grad_{type}
    , ctx_{ctx}
    , name_{NameManager::MakeUnique(to_string(ntype))}
  {}

  Node(const NodeTy ntype, NodeVec&& inputs, const Type& type,
       const Context& ctx)
    : ntype_{ntype}
    , inputs_{inputs}
    , tensor_{type}
    , grad_{type}
    , ctx_{ctx}
    , name_{NameManager::MakeUnique(to_string(ntype))}
  {}

  virtual ~Node() = default;

  /// getter functions.
  inline NodeTy ntype() const { return ntype_; };
  inline const std::string& name() const { return name_; }
  inline const Tensor& tensor() const { return tensor_; }
  inline Tensor& tensor() { return tensor_; }

  inline const Type& type() const { return tensor_.type(); }
  inline DataTy dtype() const { return tensor_.dtype(); }
  inline Shape shape() const { return tensor_.shape(); }
  inline Index elems() const { return tensor_.elems(); }

  /// setter functions.
  inline void set_name(const std::string& name) { name_ = name; }

  inline HostTy target() const { return ctx_.target(); }
  inline DeviceTy devtype() const { return ctx_.devtype(); }

  virtual void forward() = 0;
  virtual void backward() = 0;

  virtual void accept(Visitor*) const = 0;
  virtual void accept(Visitor*, const VisitFunc&, const VisitFunc&) const = 0;
};

template <typename T>
class TensorNode : public Node
{
private:
  bool init_ = false;

public:
  friend class Backend;

public:
  const Tensor* ref;

  TensorNode(const Tensor& tensor, const Context& ctx)
    : Node(T::NType, {}, tensor.type(), ctx), ref{&tensor}
  {}

  virtual void forward() = 0;
  virtual void backward() = 0;

  void accept(Visitor*) const;
  void accept(Visitor*, const VisitFunc&, const VisitFunc&) const;
};

template <typename T>
class UnaryOpNode : public Node
{
public:
  UnaryOpNode(NodePtr a, const Type& type, const Context& ctx)
    : Node{T::NType, {a}, type, ctx}
  {}

  virtual void forward() = 0;
  virtual void backward() = 0;
  inline NodePtr a() { return inputs_[0]; }
  inline const NodePtr a() const { return inputs_[0]; }

  void accept(Visitor*) const;
  void accept(Visitor*, const VisitFunc&, const VisitFunc&) const;
};

template <typename T>
class BinaryOpNode : public Node
{
public:
  BinaryOpNode(NodePtr a, NodePtr b, const Type& type, const Context& ctx)
    : Node{T::NType, {a, b}, type, ctx}
  {}

  virtual void forward() = 0;
  virtual void backward() = 0;
  inline NodePtr a() { return inputs_[0]; }
  inline NodePtr b() { return inputs_[1]; }
  inline const NodePtr a() const { return inputs_[0]; }
  inline const NodePtr b() const { return inputs_[1]; }

  void accept(Visitor*) const;
  void accept(Visitor*, const VisitFunc&, const VisitFunc&) const;
};

#define DEF_TENSOR_NODE(NAME)                           \
  class NAME final : public TensorNode<NAME>            \
  {                                                     \
  public:                                               \
    static constexpr const NodeTy NType = NodeTy::NAME; \
    static constexpr const char* Name = #NAME;          \
    using TensorNode::TensorNode;                       \
    void forward();                                     \
    void backward();                                    \
  };

#define DEF_UNARY_OP_NODE(NAME)                         \
  class NAME final : public UnaryOpNode                 \
  {                                                     \
  public:                                               \
    static constexpr const NodeTy NType = NodeTy::NAME; \
    static constexpr const char* Name = #NAME;          \
                                                        \
    using UnaryOpNode::UnaryOpNode;                     \
    void forward();                                     \
    void backward();                                    \
  };

#define DEF_BINARY_OP_NODE(NAME)                        \
  class NAME final : public BinaryOpNode<NAME>          \
  {                                                     \
  public:                                               \
    static constexpr const NodeTy NType = NodeTy::NAME; \
    static constexpr const char* Name = #NAME;          \
                                                        \
    using BinaryOpNode::BinaryOpNode;                   \
    void forward();                                     \
    void backward();                                    \
  };

DEFINED_NTYPES_BY_OPS(DEF_TENSOR_NODE, _, DEF_BINARY_OP_NODE)

template <typename T>
const Type& infer_type(NodePtr);
template <typename T>
const Type& infer_type(NodePtr, NodePtr);

} // namespace dnn
