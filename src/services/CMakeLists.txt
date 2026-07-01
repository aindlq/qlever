# Custom magic-SERVICE plugins. Each subdirectory is a self-contained service
# (parser + planner registration + operations) that wires itself in via the
# runtime registries -- with NO edits to the core parser/planner/variant. To add
# a service, add its folder plus one `add_subdirectory` line here.
#
# Services are built as OBJECT libraries and linked directly into the QLever
# binaries below, so their self-registration static initializers are always
# included in the link (object-library objects are never stripped, unlike static
# archive members). This `add_subdirectory(src/services)` must appear AFTER the
# binaries are defined in the top-level CMakeLists.

if (QLEVER_WITH_VECTOR_SEARCH)
    add_subdirectory(vectorSearch)
endif ()

# Link every registered service object library into the QLever binaries --
# including the `LibQLeverExample` embedder: any binary that parses queries
# needs the registrations, otherwise a magic-SERVICE IRI silently degrades to a
# federated HTTP SERVICE request. (External libqlever embedders likewise have
# to link the service object libraries themselves.)
get_property(serviceObjLibs GLOBAL PROPERTY QLEVER_SERVICE_OBJLIBS)
foreach (target qlever-server qlever-index LibQLeverExample)
    if (TARGET ${target})
        foreach (objLib ${serviceObjLibs})
            # Plain signature to match the binaries' existing plain-signature
            # links (via `qlever_target_link_libraries`).
            target_link_libraries(${target} ${objLib})
        endforeach ()
    endif ()
endforeach ()
