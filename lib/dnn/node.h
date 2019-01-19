#pragma once
#include <memory>
#include <string>
#include <vector>
#include "dnn/context.h"
#include "dnn/tensor.h"
#include "dnn/type.h"
#include "dnn/utils/checking.h"

namespace dnn {

class Node;
using NodePtr = std::shared_ptr<Node>;
using NodeVec = std::vector<NodePtr>;
class Visitor;

class Node
{
protected:
  NodeTy ntype_;
  std::string name_;
  NodeVec inputs_;
  Tensor tensor_;
  Tensor grad_;
  Context ctx_;

public:
  Node(const NodeTy ntype, const std::string& name, const Shape& shape,
       const Data& data, const Context& ctx)
    : ntype_{ntype}, name_{name}, tensor_{shape, data}, grad_{shape}, ctx_{ctx}
  {}

  Node(const NodeTy ntype, const std::string& name, NodeVec& inputs,
       const Shape& shape, const Context& ctx)
    : ntype_{ntype}
    , name_{name}
    , inputs_{inputs}
    , tensor_{shape}
    , grad_{shape}
    , ctx_{ctx}
  {}

  Node(const NodeTy ntype, const std::string& name, NodeVec&& inputs,
       const Shape& shape, const Context& ctx)
    : ntype_{ntype}
    , name_{name}
    , inputs_{inputs}
    , tensor_{shape}
    , grad_{shape}
    , ctx_{ctx}
  {}

  virtual ~Node() = default;

  inline NodeTy node_type() const { return ntype_; };
  inline const std::string& name() const { return name_; }
  inline const Tensor& tensor() const { return tensor_; }
  inline Shape shape() const { return tensor_.shape(); }
  inline Index elems() const { return tensor_.elems(); }
  inline DEFAULT_DTYPE* data() { return tensor_.data(); }

  inline TargetTy target() const { return ctx_.target(); }
  inline DeviceTy device() const { return ctx_.device(); }

  virtual void forward() = 0;
  virtual void backward() = 0;

  virtual void accept(Visitor*) const = 0;
};

template <typename T>
class TensorNode : public Node
{
public:
  TensorNode(const std::string& name, const Shape& shape, const Data& data,
             const Context& ctx)
    : Node(T::NType, name, shape, data, ctx)
  {}

  virtual void forward() = 0;
  virtual void backward() = 0;

  void accept(Visitor*) const;
};

template <typename T>
class UnaryOpNode : public Node
{
protected:
  virtual Shape infer_shape(NodePtr a) const;

public:
  UnaryOpNode(const std::string& name, NodePtr a, const Context& ctx)
    : Node{T::NType, name, {a}, infer_shape(a), ctx}
  {}

  virtual void forward() = 0;
  virtual void backward() = 0;
  inline NodePtr input0() { return inputs_[0]; }
  inline const NodePtr input0() const { return inputs_[0]; }

  void accept(Visitor*) const;
};

template <typename T>
class BinaryOpNode : public Node
{
protected:
  Shape infer_shape(NodePtr a, NodePtr b) const;

public:
  BinaryOpNode(const std::string& name, NodePtr a, NodePtr b,
               const Context& ctx)
    : Node{T::NType, name, {a, b}, infer_shape(a, b), ctx}
  {}

  virtual void forward() = 0;
  virtual void backward() = 0;
  inline NodePtr input0() { return inputs_[0]; }
  inline NodePtr input1() { return inputs_[1]; }
  inline const NodePtr input0() const { return inputs_[0]; }
  inline const NodePtr input1() const { return inputs_[1]; }

  void accept(Visitor*) const;
};

#define DEF_TENSOR_NODE(NAME)                           \
  class NAME final : public TensorNode<NAME>            \
  {                                                     \
  public:                                               \
    static constexpr const NodeTy NType = NodeTy::NAME; \
    static constexpr const char* Name = #NAME;          \
                                                        \
    using TensorNode::TensorNode;                       \
    void forward();                                     \
    void backward();                                    \
  };                                                    \
  using NAME##Ptr = std::shared_ptr<NAME>;

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

} // namespace dnn
