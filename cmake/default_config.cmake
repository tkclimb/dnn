if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "CMAKE_BUILD_TYPE is not set, set Debug by default")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default: Debug)" FORCE)
endif()

# for all
add_compile_options(-Wall -fno-exceptions -fno-rtti -Wno-nonportable-include-path)
# only when CMAKE_BUILD_TYPE is "Debug"
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-Wno-unused-variable -Wno-unused-private-field -fno-omit-frame-pointer -O0)
endif()

#if(MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
#  message(FATAL_ERROR "Currently MSVC is not supported" )
#else()
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wnon-virtual-dtor -fno-exceptions -fno-rtti")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-exceptions")
  # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -Og")

  # if((CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64
  #     OR CMAKE_SYSTEM_PROCESSOR STREQUAL armv7)
  #     AND CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  #   set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")
  #   set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ffast-math")
  #   set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -ffast-math")

  # else()
  #   set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -ffast-math")
  #   set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native -ffast-math")
  #   set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -march=native -ffast-math")
  # endif()
#endif()
