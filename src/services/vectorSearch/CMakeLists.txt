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
# usearch is header-only; scope its include directory and configuration to the
# targets that actually use it (PUBLIC keeps the macros ODR-consistent in every
# TU that might include usearch headers through us).
target_include_directories(vectorStorage SYSTEM PUBLIC
        "${usearch_SOURCE_DIR}/include")
target_compile_definitions(vectorStorage PUBLIC
        USEARCH_USE_FP16LIB=0 USEARCH_USE_OPENMP=0)
if (QLEVER_VECTOR_SEARCH_SIMD)
    # SIMD distance kernels with runtime CPU dispatch (also used by the exact
    # brute-force scan, which shares usearch's `metric_punned_t`). usearch
    # consumes NumKong in its dynamic-dispatch mode, which needs NumKong's C
    # dispatch tables compiled once; we build them as a small static library
    # (upstream only offers a shared one).
    file(GLOB NUMKONG_DISPATCH_SOURCES
         "${numkong_SOURCE_DIR}/c/numkong.c"
         "${numkong_SOURCE_DIR}/c/dispatch_*.c")
    add_library(numkongDispatch STATIC ${NUMKONG_DISPATCH_SOURCES})
    target_include_directories(numkongDispatch SYSTEM PUBLIC
            "${numkong_SOURCE_DIR}/include")
    target_compile_definitions(numkongDispatch PUBLIC NK_DYNAMIC_DISPATCH=1)
    if (NOT MSVC)
        # Third-party C sources; not subject to our warning level.
        target_compile_options(numkongDispatch PRIVATE -w)
    endif ()
    # Plain signature to match `qlever_target_link_libraries` above.
    qlever_target_link_libraries(vectorStorage numkongDispatch)
    target_compile_definitions(vectorStorage PUBLIC USEARCH_USE_NUMKONG=1)
else ()
    target_compile_definitions(vectorStorage PUBLIC USEARCH_USE_NUMKONG=0)
endif ()

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
