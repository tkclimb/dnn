cmake_minimum_required(VERSION 3.7)
project(dnn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUirED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS "CMAKE_DEBUG_POSTFIX: ${CMAKE_DEBUG_POSTFIX}")
message(STATUS "ENVirONMENT VARIABLES:")
message(STATUS "    DYLD_LIBRARY_PATH: $ENV{DYLD_LIBRARY_PATH}")

option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_DOCS "Build documents" OFF)
message(STATUS "BUILD_TESTS: ${BUILD_TESTS}")
message(STATUS "BUILD_EXAMPLES: ${BUILD_EXAMPLES}")
message(STATUS "BUILD_DOCS: ${BUILD_DOCS}")

option(USE_SANITIZER "Use Sanitizer" OFF)
option(USE_CLANG_TIDY "Use clang-tidy" OFF)
message(STATUS "USE_SANITIZER: ${USE_SANITIZER}")
message(STATUS "USE_CLANG_TIDY: ${USE_CLANG_TIDY}")

# Turn off in-combinable options
if (BUILD_TESTS) 
  if(USE_SANITIZER)
    message(FATAL_ERROR "USE_SANITIZER option is not available with BUILD_TESTS...")
  endif()

  if(USE_CLANG_TIDY)
    message(FATAL_ERROR "USE_CLANG_TIDY option is not available with BUILD_TESTS...")
  endif()
endif()

get_filename_component(DNN_ROOT_DIR . ABSOLUTE)
get_filename_component(DNN_LIB_DIR lib ABSOLUTE)
get_filename_component(DNN_THIRDPARTY_DIR thirdparty ABSOLUTE)
get_filename_component(DNN_TESTS_DIR tests ABSOLUTE)
get_filename_component(DNN_EXAMPLES_DIR examples ABSOLUTE)
get_filename_component(DNN_LIB_DIR lib ABSOLUTE)
get_filename_component(DNN_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR} ABSOLUTE)
get_filename_component(DNN_BUILD_LIB_DIR ${DNN_BUILD_DIR}/lib ABSOLUTE)
get_filename_component(DNN_BUILD_DIR ${DNN_BUILD_DNN_DIR}/lib/dnn ABSOLUTE)
get_filename_component(DNN_BUILD_THIRDPARTY_DIR ${DNN_BUILD_DIR}/THIRDParty ABSOLUTE)
get_filename_component(DNN_BUILD_TESTS_DIR ${DNN_BUILD_DIR}/tests ABSOLUTE)


# Build thirdparty first
add_subdirectory(${DNN_THIRDPARTY_DIR})
get_filename_component(DNN_EIGEN_DIR ${DNN_THIRDPARTY_DIR}/eigen ABSOLUTE)

list(APPEND CMAKE_MODULE_PATH ${DNN_ROOT_DIR}/cmake)
include(default_config)
include(document_config)
include(sanitizer_config)

if(USE_CLANG_TIDY)
  include(clang_tidy_config) # Config clang-tidy
endif()

include_directories(BEFORE
  ${DNN_LIB_DIR}
  ${DNN_EIGEN_DIR}
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${DNN_BUILD_LIB_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${DNN_BUILD_LIB_DIR})
link_directories(${DNN_BUILD_LIB_DIR})

# Build DNN library
add_subdirectory(${DNN_LIB_DIR})

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(${DNN_TESTS_DIR})
endif()
