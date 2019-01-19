set(USE_SANITIZER "" CACHE STRING
    "Define the sanitizer to build binaries and tests.")

if(USE_SANITIZER)
  set_property(GLOBAL PROPERTY -fno-omit-frame-pointer -O1 -fsanitize=address -fsanitize=leak)
  link_libraries(-fsanitize=address -fsanitize=leak)
endif()

# if(USE_SANITIZER)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
# 
#   if(CMAKE_BUILD_TYPE MATCHES "Debug")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
#   elseif(NOT CMAKE_BUILD_TYPE MATCHES "Debug" AND
#          NOT CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gline-tables-only")
#   endif()
# 
#   if(USE_SANITIZER STREQUAL "Address")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
# 
#   elseif(USE_SANITIZER MATCHES "Memory(WithOrigins)?")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory")
#     if(USE_SANITIZER STREQUAL "MemoryWithOrigins")
#       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize-memory-track-origins") 
#     endif()
# 
#   elseif(USE_SANITIZER STREQUAL "Undefined")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sanitize-recover=all")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sanitize=vptr,function") 
# 
#   elseif(USE_SANITIZER STREQUAL "Thread")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
# 
#   elseif(USE_SANITIZER STREQUAL "Address;Undefined" OR
#          USE_SANITIZER STREQUAL "Undefined;Address")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sanitize=vptr,function")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sanitize-recover=all")
# 
#   elseif(USE_SANITIZER STREQUAL "Leaks")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
# 
#   else()
#     message(FATAL_ERROR "USE_SANITIZER: ${USE_SANITIZER} is not supported")
#   endif()
# endif()

