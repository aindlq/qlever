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
    #
    # CRUCIAL: NumKong's per-ISA kernels are compiled ONLY where an explicit
    # `NK_TARGET_<ISA>=1` define is passed (its headers otherwise auto-detect
    # from the baseline `-march`, which would silently yield serial-only
    # kernels). The kernels carry per-function `target` attributes, so no
    # global `-march` change is needed and the binary stays portable -- the
    # best implementation is selected per CPU at run time. The probes below
    # mirror NumKong's own `cmake/nk_x86_isa_probes.cmake` (compile pass).
    include(CheckSourceCompiles)
    set(NUMKONG_ISA_DEFS "")
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
        set(_nkProbes
            "HASWELL|-mavx2 -mfma -mf16c|x86_haswell.c"
            "SKYLAKE|-mavx512f -mavx512bw -mavx512dq -mavx512vl|x86_skylake.c"
            "ICELAKE|-mavx512vnni -mavx512vl|x86_icelake.c"
            "GENOA|-mavx512bf16 -mavx512vl|x86_genoa.c"
            "SAPPHIRE|-mavx512fp16 -mavx512vl|x86_sapphire.c"
            "SAPPHIREAMX|-mamx-tile -mamx-int8|x86_sapphireamx.c"
            "GRANITEAMX|-mamx-tile -mamx-fp16|x86_graniteamx.c"
            "DIAMOND|-mavx10.2-512|x86_diamond.c"
            "TURIN|-mavx512vp2intersect|x86_turin.c"
            "ALDER|-mavxvnni|x86_alder.c"
            "SIERRA|-mavxvnniint8|x86_sierra.c")
        set(_nkSavedFlags "${CMAKE_REQUIRED_FLAGS}")
        foreach (_nkProbe IN LISTS _nkProbes)
            string(REPLACE "|" ";" _nkParts "${_nkProbe}")
            list(GET _nkParts 0 _nkIsa)
            list(GET _nkParts 1 _nkFlags)
            list(GET _nkParts 2 _nkFile)
            file(READ "${numkong_SOURCE_DIR}/probes/${_nkFile}" _nkSource)
            set(CMAKE_REQUIRED_FLAGS "${_nkFlags}")
            check_source_compiles(C "${_nkSource}" nkCanCompile${_nkIsa})
            if (nkCanCompile${_nkIsa})
                list(APPEND NUMKONG_ISA_DEFS "NK_TARGET_${_nkIsa}=1")
            else ()
                list(APPEND NUMKONG_ISA_DEFS "NK_TARGET_${_nkIsa}=0")
            endif ()
        endforeach ()
        set(CMAKE_REQUIRED_FLAGS "${_nkSavedFlags}")
    else ()
        message(STATUS "QLever vector search: no NumKong ISA probes for "
                "${CMAKE_SYSTEM_PROCESSOR}; SIMD kernels fall back to the "
                "header auto-detection of the baseline architecture.")
    endif ()

    file(GLOB NUMKONG_DISPATCH_SOURCES
         "${numkong_SOURCE_DIR}/c/numkong.c"
         "${numkong_SOURCE_DIR}/c/dispatch_*.c")
    add_library(numkongDispatch STATIC ${NUMKONG_DISPATCH_SOURCES})
    target_include_directories(numkongDispatch SYSTEM PUBLIC
            "${numkong_SOURCE_DIR}/include")
    # PUBLIC: the consuming C++ TUs (usearch headers) must see the same
    # configuration as the dispatch tables.
    target_compile_definitions(numkongDispatch PUBLIC NK_DYNAMIC_DISPATCH=1
            ${NUMKONG_ISA_DEFS})
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

if (QLEVER_VECTOR_SEARCH_PARQUET)
    # Prefer upstream Arrow's own CMake config (targets Parquet::parquet_*);
    # fall back to a conan-generated "Arrow" config, whose arrow::arrow target
    # bundles the Parquet component.
    find_package(Parquet CONFIG QUIET)
    if (TARGET Parquet::parquet_shared)
        set(_qleverParquetTarget Parquet::parquet_shared)
    elseif (TARGET Parquet::parquet_static)
        set(_qleverParquetTarget Parquet::parquet_static)
    else ()
        find_package(Arrow CONFIG REQUIRED)
        if (TARGET arrow::arrow)
            set(_qleverParquetTarget arrow::arrow)
        else ()
            message(FATAL_ERROR "Found Arrow, but no usable Parquet target; "
                    "install Arrow with the Parquet component enabled.")
        endif ()
    endif ()
    message(STATUS "QLever vector search: Parquet ingest enabled via "
            "${_qleverParquetTarget}")
    qlever_target_link_libraries(vectorStorage ${_qleverParquetTarget})
    target_compile_definitions(vectorStorage PUBLIC QLEVER_WITH_PARQUET=1)
endif ()

add_library(service_vectorSearch OBJECT
        VectorSearch.cpp
        VectorSearchAmong.cpp
        VectorSearchJoin.cpp
        VectorSearchQuery.cpp
        VectorSearchService.cpp
        VectorQueryPoint.cpp
        VectorDistanceExpression.cpp
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
