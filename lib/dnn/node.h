#pragma once
#include <memory>
#include <string>
#include <vector>
#include "dnn/arrayref.h"
#include "dnn/context.h"
#include "dnn/name_manager.h"
#include "dnn/tensor/tensor.h"
#include "dnn/type.h"
#include "dnn/utils/checking.h"
#include "dnn/utils/support.h"

namespace dnn {

class Node;
class Graph;

using NodePtr = Node*;
using NodeVec = std::vector<NodePtr>;
using NodeArray = ArrayRef<NodePtr>;
using TensorArray = ArrayRef<Tensor*>;
using TensorVec = std::vector<Tensor*>;
class Visitor;
using VisitFunc = std::function<void(const Node*)>;

class Node
{
protected:
  std::string name_ = "Node";
  NodeTy nodety_;
  Type ty_;
  Graph* graph_ = nullptr;
  Context* ctx_ = nullptr;

public:
  friend class Backend;

  Node(const std::string& name, const NodeTy nodety, const Type& ty, Graph& g,
       Context& ctx);
  virtual ~Node() = default;

  /// getter functions.
  inline const std::string& name() const { return name_; }
  inline Tensor* tensor() { return ctx_->get_tensor(this); }
  inline const Tensor* tensor() const { return ctx_->get_tensor(this); }
  inline const Tensor* in_tensor(const Index idx) const
  {
    return ctx_->get_in_tensor(this, idx);
  }
  const NodeVec& inputs() const;
  const NodePtr input(const Index idx) const;
  inline const Type& type() const { return ty_; }
  inline NodeTy nodety() const { return nodety_; };
  inline DataTy dataty() const { return tensor()->dataty(); }
  inline Shape shape() const { return tensor()->shape(); }
  inline Index elems() const { return tensor()->elems(); }
  inline HostTy hostty() const { return ctx_->hostty(); }
  inline DeviceTy devty() const { return ctx_->devty(); }

  /// setter functions.
  inline void set_name(const std::string& name) { name_ = name; }
  void set_in_tensors(TensorArray);

  /// virtual functions.
  virtual void forward() = 0;
  virtual void backward() = 0;
  virtual Tensor* operator()(TensorArray tensors) = 0;
  virtual void accept(Visitor*) const = 0;
  virtual void accept(Visitor*, const VisitFunc&, const VisitFunc&) const = 0;
  virtual Index num_inputs() = 0;

private:
  void alloc_tensor() { ctx_->alloc_tensor(this); }
};

template <typename T>
class VisitableNode : public Node
{
public:
  using Node::Node;
  virtual void forward() override = 0;
  virtual void backward() override = 0;
  virtual Index num_inputs() override = 0;
  void accept(Visitor*) const override;
  void accept(Visitor*, const VisitFunc&, const VisitFunc&) const override;

  Tensor* operator()(TensorArray tensors) override
  {
    set_in_tensors(tensors);
    forward(); /// must call forward after set input tensors
    return tensor();
  }
};

#define DEFINE_BASIC_NODE_FUNCS(NAME, NUM_INPUTS)                         \
  class NAME : public VisitableNode<NAME>                                 \
  {                                                                       \
  public:                                                                 \
    NAME(const std::string& name, const Type& ty, Graph& g, Context& ctx) \
      : VisitableNode(name, NodeTy::NAME, ty, g, ctx)                     \
    {}                                                                    \
    void forward() override;                                              \
    void backward() override;                                             \
    Index num_inputs() override { return NUM_INPUTS; }                    \
  };

DEFINE_BASIC_NODE_FUNCS(Placeholder, 1)
DEFINE_BASIC_NODE_FUNCS(Add, 2)
DEFINE_BASIC_NODE_FUNCS(Sub, 2)
DEFINE_BASIC_NODE_FUNCS(Mul, 2)
DEFINE_BASIC_NODE_FUNCS(Matmul, 2)

template <typename T>
const Type& infer_type(const T& node);

} // namespace dnn
