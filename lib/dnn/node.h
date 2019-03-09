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
using NodePtr = std::shared_ptr<Node>;
using NodeVec = std::vector<NodePtr>;
using TensorVec = std::vector<Tensor*>;
using TensorArray = ArrayRef<Tensor*>;
class Visitor;
using VisitFunc = std::function<void(const Node*)>;

class Node
{
protected:
  std::string name_ = "Node";
  NodeTy nodety_;
  Type ty_;
  Context* ctx_ = nullptr;
  TensorVec in_tensors_ = {};

public:
  friend class Backend;

  Node(const std::string& name, const NodeTy nodety, const Type& ty,
       Context& ctx)
    : name_{name}, nodety_{nodety}, ty_{ty}, ctx_{&ctx}
  {
    alloc_tensor();
  }

  // Node(const NodeTy nodety, NodeVec& in_tensors, const Type& ty, Context&
  // ctx)
  //   : Node(NameManager::MakeUnique(to_string(nodety)), nodety, in_tensors,
  //   ty, ctx)
  // {}

  virtual ~Node() = default;

  /// getter functions.
  inline NodeTy nodety() const { return nodety_; };
  inline const std::string& name() const { return name_; }
  inline Tensor& tensor() { return *(ctx_->get_tensor(name_)); }
  inline const Tensor& tensor() const { return *(ctx_->get_tensor(name_)); }
  void set_tensor(Tensor& tensor) { return ctx_->set_tensor(name_, tensor); }

  inline TensorVec& in_tensors() { return in_tensors_; }
  inline Tensor* in_tensor(Index idx) { return in_tensors_[idx]; }
  inline const TensorVec& in_tensors() const { return in_tensors_; }
  inline const Tensor* in_tensor(Index idx) const { return in_tensors_[idx]; }

  inline const Type& type() const { return ty_; }
  inline DataTy dataty() const { return tensor().dataty(); }
  inline Shape shape() const { return tensor().shape(); }
  inline Index elems() const { return tensor().elems(); }

  /// setter functions.
  inline void set_name(const std::string& name) { name_ = name; }

  inline HostTy hostty() const { return ctx_->hostty(); }
  inline DeviceTy devty() const { return ctx_->devty(); }

  virtual void forward(TensorArray inputs) = 0;
  virtual void backward() = 0;

  virtual void accept(Visitor*) const = 0;
  virtual void accept(Visitor*, const VisitFunc&, const VisitFunc&) const = 0;

private:
  void alloc_tensor() { ctx_->alloc_tensor(name_, ty_); }
};

template <typename T>
class VisitableNode : public Node
{
public:
  using Node::Node;
  virtual void forward(TensorArray inputs) = 0;
  virtual void backward() = 0;
  void accept(Visitor*) const;
  void accept(Visitor*, const VisitFunc&, const VisitFunc&) const;
};

#define DEFINE_BASIC_NODE_FUNCS(NAME)                           \
  class NAME : public VisitableNode<NAME>                       \
  {                                                             \
  public:                                                       \
    NAME(const std::string& name, const Type& ty, Context& ctx) \
      : VisitableNode(name, NodeTy::NAME, ty, ctx)              \
    {}                                                          \
    void forward(TensorArray inputs) override;                  \
    void backward() override;                                   \
  };

DEFINE_BASIC_NODE_FUNCS(Placeholder)
DEFINE_BASIC_NODE_FUNCS(Add)
DEFINE_BASIC_NODE_FUNCS(Sub)
DEFINE_BASIC_NODE_FUNCS(Mul)
DEFINE_BASIC_NODE_FUNCS(Matmul)

template <typename T>
const Type& infer_type(TensorArray inputs);
template <typename T>
const Type& infer_type(TensorArray inputs);

} // namespace dnn
