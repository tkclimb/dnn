#include <string>

#include "dnn/graph.h"
#include "dnn/utils/checking.h"

namespace dnn {

NodePtr Graph::add_node(const NodePtr& nptr, const std::string& name)
{
  nptr->set_name(name);
  auto unique_name = nptr->name();

  if (symbol_map_.count(unique_name)) {
    EXCEPTION_STR("node(" + unique_name + ") already exists in this graph...");
  } else {
    auto sym = get_unique_symbol();
    node_map_[sym] = nptr;
    symbol_map_[unique_name] = sym;
  }
  return nptr;
}

Graph::Symbol Graph::get_unique_symbol()
{
  auto current_id = num_nodes_;
  ++num_nodes_;
  return current_id;
}

} // namespace dnn