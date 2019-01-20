/**
 * Copyright (c) 2018 by Contributers
 * @brief This contains NameManager implementation
 *
 * @file name_manager.h
 * @author tkclimb
 * @data 2018-10-18
 */
#ifndef DNN_CORE_NAMEMANAGER_H
#define DNN_CORE_NAMEMANAGER_H

#include <map>
#include <sstream>
#include <vector>

namespace dnn {

class NameManager
{
  using NameCounter = std::map<std::string, int>;

public:
  /// return uniq name by putting index after the name
  static std::string MakeUnique(const std::string& name)
  {
    NameCounter& name_ctr = GetNameCtr();

    std::stringstream os;
    os << name;
    if (CheckUnique(name)) {
      name_ctr[name] = 1;
    } else {
      name_ctr[name]++;
    }
    os << "_" << name_ctr[name];
    return os.str();
  }

  /// check whether the given name is uniq or not
  static bool CheckUnique(const std::string& name)
  {
    NameCounter& name_ctr = GetNameCtr();
    return name_ctr.count(name) <= 0;
  }

  static NameCounter& GetNameCtr()
  {
    static NameCounter name_ctr;
    return name_ctr;
  }

private:
  /// ctors and dtor are defined in private context because it's singleton
  /// @{
  NameManager() = delete;
  ~NameManager() = delete;
  NameManager(NameManager&) = delete;
  NameManager(NameManager&&) = delete;
  NameManager(const NameManager&) = delete;
  NameManager(const NameManager&&) = delete;
  /// @}
};

} // namespace dnn

#endif