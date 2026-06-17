# The built-in vector-search SERVICE, as a self-contained plugin.
#
# `vectorStorage` is a plain static lib with the on-disk store + builder + reader
# (depends only on util/global + header-only usearch); it is also linked by the
# storage unit test. The service itself is an OBJECT library so its
# self-registration (parser, planner, and index-extension hooks) survives the
# link; it is linked into the binaries by `src/services/CMakeLists.txt`.

add_library(vectorStorage
        VectorIndex.cpp
        VectorIndexBuilder.cpp
        VectorInputReader.cpp)
qlever_target_link_libraries(vectorStorage util global)

add_library(service_vectorSearch OBJECT
        VectorSearch.cpp
        VectorSearchJoin.cpp
        VectorSearchQuery.cpp
        VectorSearchService.cpp
        VectorIndexExtension.cpp
        EmbeddingClient.cpp)
# `vectorStorage` and `http` are PUBLIC so they propagate to the binaries that
# link this object library (the binaries don't otherwise link them).
target_link_libraries(service_vectorSearch PUBLIC vectorStorage http)
target_link_libraries(service_vectorSearch PRIVATE engine index parser util global)
if (QLEVER_WITH_LLAMACPP)
    target_link_libraries(service_vectorSearch PRIVATE llama)
endif ()

# Advertise this object library to `src/services/CMakeLists.txt`.
set_property(GLOBAL APPEND PROPERTY QLEVER_SERVICE_OBJLIBS service_vectorSearch)
