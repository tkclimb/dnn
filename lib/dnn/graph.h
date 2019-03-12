#pragma once
#include <map>
#include <memory>
#include <vector>
#include "dnn/context.h"
#include "dnn/node.h"
#include "dnn/type.h"

namespace dnn {

class Graph
{
  using Symbol = std::string;
  // using SymbolMap = std::map<std::string, Symbol>;
  using NodeVec = std::vector<NodePtr>;
  using NodeMap = std::map<Symbol, NodePtr>;
  using NodeVecMap = std::map<Symbol, std::vector<NodePtr>>;

private:
  NodeMap node_map_;
  NodeVecMap link_map_;
  // SymbolMap symbol_map_;
  Index num_nodes_ = 0;

public:
  Graph() = default;
  ~Graph() = default;

  void add_node(const NodePtr, const Symbol&);
  void add_link(const NodePtr, const NodePtr);
  const NodeVec& get_paren_nodes(const NodePtr);
  Node& placeholder(const std::string&, const Type&, Context&);
  Node& add(const Symbol&, const Type&, Context&);
  Node& mul(const Symbol&, const Type&, Context&);
  // Symbol get_unique_symbol();
};

} // namespace dnn