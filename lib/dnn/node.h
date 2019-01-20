#pragma once
#include <memory>
#include <string>
#include <vector>
#include "dnn/context.h"
#include "dnn/tensor.h"
#include "dnn/type.h"
#include "dnn/utils/checking.h"
#include "dnn/utils/support.h"

namespace dnn {

class Node;
using NodePtr = std::shared_ptr<Node>;
using NodeVec = std::vector<NodePtr>;
class Visitor;
using VisitFunc = std::function<void(const Node*)>;

class Placeholder;
class Add;
class Sub;

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

  Node(const NodeTy ntype, NodeVec& inputs, const Shape& shape,
       const Context& ctx)
    : ntype_{ntype}
    , inputs_{inputs}
    , tensor_{shape}
    , grad_{shape}
    , ctx_{ctx}
    , name_{to_string(ntype)}
  {}

  Node(const NodeTy ntype, NodeVec&& inputs, const Shape& shape,
       const Context& ctx)
    : ntype_{ntype}
    , inputs_{inputs}
    , tensor_{shape}
    , grad_{shape}
    , ctx_{ctx}
    , name_{to_string(ntype)}
  {}

  virtual ~Node() = default;

  /// getter functions.
  inline NodeTy node_type() const { return ntype_; };
  inline const std::string& name() const { return name_; }
  inline const Tensor& tensor() const { return tensor_; }
  inline Shape shape() const { return tensor_.shape(); }
  inline Index elems() const { return tensor_.elems(); }
  inline DEFAULT_DTYPE* data() { return tensor_.data(); }

  /// setter functions.
  inline void set_name(const std::string& name) { name_ = name; }

  inline TargetTy target() const { return ctx_.target(); }
  inline DeviceTy device() const { return ctx_.device(); }

  virtual void forward() = 0;
  virtual void backward() = 0;

  virtual void accept(Visitor*) const = 0;
  virtual void accept(Visitor*, const VisitFunc&, const VisitFunc&) const = 0;
};

template <typename T>
class TensorNode : public Node
{
public:
  friend class Backend;

protected:
  const Data* data_;

public:
  TensorNode(const Data* data, const Shape& shape, const Context& ctx)
    : Node(T::NType, {}, shape, ctx), data_{data}
  {}

  virtual void forward() = 0;
  virtual void backward() = 0;

  void accept(Visitor*) const;
  void accept(Visitor*, const VisitFunc&, const VisitFunc&) const;
};

template <typename T>
class UnaryOpNode : public Node
{
protected:
  virtual Shape infer_shape(NodePtr a) const;

public:
  UnaryOpNode(NodePtr a, const Context& ctx)
    : Node{T::NType, {a}, infer_shape(a), ctx}
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
protected:
  Shape infer_shape(NodePtr a, NodePtr b) const;

public:
  BinaryOpNode(NodePtr a, NodePtr b, const Context& ctx)
    : Node{T::NType, {a, b}, infer_shape(a, b), ctx}
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
                                                        \
  private:                                              \
    Shape infer_shape(NodePtr a) const;                 \
  };                                                    \
  using NAME##Ptr = std::shared_ptr<NAME>;

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
  };                                                    \
  using NAME##Ptr = std::shared_ptr<NAME>;

DEF_TENSOR_NODE(Placeholder)
DEF_BINARY_OP_NODE(Add)
DEF_BINARY_OP_NODE(Sub)

} // namespace dnn
