find_program(
  CLANG_TIDY_FOUND
  NAMES "clang-tidy"
  PATHS "$ENV{LLVM_DIR}"
  PATH_SUFFIXES "../../bin"
  DOC "Path to clang-tidy executable"
)

if(NOT CLANG_TIDY_FOUND)
  message(STATUS "clang-tidy is not found.")
else()
  message(STATUS "clang-tidy is found: ${CLANG_TIDY_FOUND}")
  set(CLANG_TIDY_CMD "${CLANG_TIDY_FOUND} -checks=*,google-readability-casting,-clang-analyzer-alpha.*,-header-filter=${INCLUDE_DIR}")
  message(STATUS "clang-tidy command: ${CLANG_TIDY_CMD}")
endif()
