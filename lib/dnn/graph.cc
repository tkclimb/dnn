#include <string>

#include "dnn/graph.h"
#include "dnn/utils/checking.h"

namespace dnn {

void Graph::add_node(const NodePtr node, const Symbol& sym)
{
  node->set_name(sym);
  if (node_map_.count(sym)) {
    EXCEPTION_STR("node(" + sym + ") already exists in this graph...");
  } else {
    // auto sym = get_unique_symbol();
    node_map_[sym] = const_cast<NodePtr>(node);
    link_map_[sym] = {}; /// initialize the link with empty vector
  }
}

void Graph::add_link(NodePtr child_node, NodePtr paren_node)
{
  const auto& child_sym = child_node->name();
  const auto& paren_sym = paren_node->name();
  if (!node_map_.count(child_sym)) {
    EXCEPTION_STR("node(" + child_sym + ") doesn't exist in this graph...");
  } else if (!node_map_.count(paren_sym)) {
    EXCEPTION_STR("node(" + paren_sym + ") doesn't exist in this graph...");
  } else {
    link_map_[child_sym].push_back(paren_node);
  }
}

const NodeVec& Graph::get_paren_nodes(const NodePtr child_node)
{
  auto child_sym = child_node->name();
  if (!link_map_.count(child_sym)) {
    EXCEPTION_STR("node(" + child_sym +
                  ") doesn't connect to any parent nodes...");
  } else {
    return link_map_[child_sym];
  }
  // auto paren_sym = paren_syms[idx];
  // if (!node_map_.count(paren_sym)) {
  //   EXCEPTION_STR("node(" + paren_sym + ") doesn't exist in this
  //   graph...");
  // } else {
  //   return node_map_[paren_sym];
  // }
}

Node& Graph::placeholder(const std::string& sym, const Type& ty, Context& ctx)
{
  auto node = new Placeholder(sym, ty, *this, ctx);
  add_node(node, sym);
  return *node;
}

Node& Graph::add(const std::string& sym, const Type& ty, Context& ctx)
{
  auto node = new Add(sym, ty, *this, ctx);
  add_node(node, sym);
  return *node;
}

Node& Graph::mul(const std::string& sym, const Type& ty, Context& ctx)
{
  auto node = new Mul(sym, ty, *this, ctx);
  add_node(node, sym);
  return *node;
}

// Graph::Symbol Graph::get_unique_symbol()
// {
//   auto current_id = num_nodes_;
//   ++num_nodes_;
//   return current_id;
// }

} // namespace dnn