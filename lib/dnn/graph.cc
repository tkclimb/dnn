#include <string>

#include "dnn/graph.h"
#include "dnn/utils/checking.h"

namespace dnn {

void Graph::add_node(NodePtr nptr, const std::string& name)
{
  nptr->set_name(name);

  if (symbol_map_.count(name)) {
    EXCEPTION_STR("node(" + name + ") already exists in this graph...");
  } else {
    auto sym = get_unique_symbol();
    node_map_[sym] = nptr;
    symbol_map_[name] = sym;
  }
}

Node& Graph::placeholder(const std::string& name, const Type& ty, Graph& graph,
                         Context& ctx)
{
  auto nptr = new Placeholder(name, ty, ctx);
  graph.add_node(nptr, name);
  return *nptr;
}

Node& Graph::add(const std::string& name, const Type& ty, Graph& graph,
                 Context& ctx)
{
  auto nptr = new Add(name, ty, ctx);
  graph.add_node(nptr, name);
  return *nptr;
}

Node& Graph::mul(const std::string& name, const Type& ty, Graph& graph,
                 Context& ctx)
{
  auto nptr = new Mul(name, ty, ctx);
  graph.add_node(nptr, name);
  return *nptr;
}

Graph::Symbol Graph::get_unique_symbol()
{
  auto current_id = num_nodes_;
  ++num_nodes_;
  return current_id;
}

} // namespace dnn