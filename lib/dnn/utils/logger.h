#pragma once
#include <iostream>
#include <string>

namespace dnn {

using std::endl;
using std::ostream;
using std::string;

struct Logger
{
  static ostream& Debug(const string& msg, ostream& os = std::cout)
  {
    return Message("DEBUG", msg, os);
  }

  static ostream& Info(const string& msg, ostream& os = std::cout)
  {
    return Message("INFO", msg, os);
  };

  static ostream& Fatal(const string& msg, ostream& os = std::cout)
  {
    return Message("FATAL", msg, os);
  }

  static ostream& Message(const string& tag, const string& msg, ostream& os)
  {
    os << "[" + tag + "] " + msg << std::endl;
    return os;
  }

  static ostream& Message(const string& tag, const string& msg,
                          const string& file_name, const string& func_name,
                          const unsigned line_num)
  {
    return Message(tag, msg, std::cout) << "file: " << file_name << "\n"
                                        << "func: " << func_name << "\n"
                                        << "line: " << line_num << endl;
  }
};

} // namespace dnn