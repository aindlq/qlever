# Example custom-operation extension: a `SERVICE`-triggered outer-bound join.
add_library(extension_exampleOperation OBJECT ExampleOperation.cpp)
qlever_target_link_libraries(extension_exampleOperation parser sparqlExpressions
        engine index global)
set_property(GLOBAL APPEND PROPERTY QLEVER_EXTENSION_OBJLIBS
        extension_exampleOperation)
