#pragma once
#include <memory>
#include <string>
#include <vector>
#include "dnn/tensor.h"
#include "dnn/utils/checking.h"
#include "dnn/visitor.h"

namespace dnn {

class Node;
using NodePtr = std::shared_ptr<Node>;
using NodeVec = std::vector<NodePtr>;

class Node
{
protected:
  std::string name_;
  NodeVec inputs_;
  Tensor tensor_;
  Tensor grad_;

public:
  Node(const std::string& name, Tensor tensor)
    : name_{name}, tensor_{tensor}, grad_{shape()}
  {}

  Node(const std::string& name, NodeVec& inputs, const Shape& shape)
    : name_{name}, inputs_{inputs}, tensor_{shape}, grad_{shape}
  {}

  Node(const std::string& name, NodeVec&& inputs, const Shape& shape)
    : name_{name}, inputs_{inputs}, tensor_{shape}, grad_{shape}
  {}

  virtual ~Node() = default;

  virtual void forward() = 0;
  virtual void backward() = 0;

  Shape shape() const { return tensor_.shape(); }
};

class TensorNode : public Node
{
public:
  using Node::Node;

  virtual void forward() = 0;
  virtual void backward() = 0;
};

class UnaryOpNode : public Node
{
protected:
  virtual Shape infer_shape(NodePtr a) const
  {
    EXCEPTION("This must be override...");
  };

public:
  UnaryOpNode(const std::string& name, NodePtr a)
    : Node{name, {a}, infer_shape(a)}
  {}

  virtual void forward() = 0;
  virtual void backward() = 0;
};

class BinaryOpNode : public Node
{
protected:
  virtual Shape infer_shape(NodePtr a, NodePtr b) const
  {
    EXCEPTION("This must be override...");
  };

public:
  BinaryOpNode(const std::string& name, NodePtr a, NodePtr b)
    : Node{name, {a, b}, infer_shape(a, b)}
  {}

  virtual void forward() = 0;
  virtual void backward() = 0;
};

#define DEF_TENSOR_NODE(NAME)                                  \
  class NAME final : public TensorNode, public Visitable<NAME> \
  {                                                            \
    static constexpr const char* Name = #NAME;                 \
                                                               \
  public:                                                      \
    using TensorNode::TensorNode;                              \
    void forward();                                            \
    void backward();                                           \
  };

#define DEF_UNARY_OP_NODE(NAME)                                 \
  class NAME final : public UnaryOpNode, public Visitable<NAME> \
  {                                                             \
    static constexpr const char* Name = #NAME;                  \
                                                                \
  public:                                                       \
    using UnaryOpNode::UnaryOpNode;                             \
    void forward();                                             \
    void backward();                                            \
                                                                \
  private:                                                      \
    Shape infer_shape(NodePtr a) const;                         \
  };

#define DEF_BINARY_OP_NODE(NAME)                                 \
  class NAME final : public BinaryOpNode, public Visitable<NAME> \
  {                                                              \
    static constexpr const char* Name = #NAME;                   \
                                                                 \
  public:                                                        \
    using BinaryOpNode::BinaryOpNode;                            \
    void forward();                                              \
    void backward();                                             \
                                                                 \
  private:                                                       \
    Shape infer_shape(NodePtr a, NodePtr b) const;               \
  };

DEF_TENSOR_NODE(Placeholder)
DEF_BINARY_OP_NODE(Add)

} // namespace dnn
