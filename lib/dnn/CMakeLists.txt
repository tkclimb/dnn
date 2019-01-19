add_subdirectory(backend)

add_library(dnn node.cc visitor.cc tensor.cc)
target_link_libraries(dnn dnn_backend)