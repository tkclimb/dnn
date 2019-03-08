#pragma once
#include <map>
#include "dnn/context.h"
#include "dnn/node.h"
#include "dnn/type.h"

namespace dnn {

class Graph
{
  using Symbol = primitive::Idx;
  using NodeMap = std::map<Symbol, NodePtr>;
  using SymbolMap = std::map<std::string, Symbol>;

private:
  NodeMap node_map_;
  SymbolMap symbol_map_;
  primitive::Idx num_nodes_ = 0;

public:
  Graph() = default;
  ~Graph() = default;

  NodePtr add_node(const NodePtr& nptr, const std::string& name);
  Symbol get_unique_symbol();
};

} // namespace dnn