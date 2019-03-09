add_subdirectory(backend)

add_library(dnn node.cc 
                context.cc
                graph.cc
                visitor.cc
                tensor/tensor.cc
                utils/support.cc)
target_link_libraries(dnn dnn_backend)