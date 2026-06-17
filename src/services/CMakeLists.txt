# Custom magic-SERVICE plugins. Each subdirectory with a `CMakeLists.txt` is a
# self-contained service (parser + planner registration + operations) that wires
# itself in via the runtime registries -- with NO edits to the core
# parser/planner/variant or to this file. To add a service, drop a folder here.
#
# Services are built as OBJECT libraries and linked directly into the QLever
# binaries below, so their self-registration static initializers are always
# included in the link (object-library objects are never stripped, unlike static
# archive members). This `add_subdirectory(src/services)` must appear AFTER the
# binaries are defined in the top-level CMakeLists.

file(GLOB serviceDirs CONFIGURE_DEPENDS LIST_DIRECTORIES true
     "${CMAKE_CURRENT_SOURCE_DIR}/*")
foreach (dir ${serviceDirs})
    if (IS_DIRECTORY "${dir}" AND EXISTS "${dir}/CMakeLists.txt")
        add_subdirectory("${dir}")
    endif ()
endforeach ()

# Link every registered service object library into the QLever binaries.
get_property(serviceObjLibs GLOBAL PROPERTY QLEVER_SERVICE_OBJLIBS)
foreach (target qlever-server qlever-index)
    if (TARGET ${target})
        foreach (objLib ${serviceObjLibs})
            # Plain signature to match the binaries' existing plain-signature
            # links (via `qlever_target_link_libraries`).
            target_link_libraries(${target} ${objLib})
        endforeach ()
    endif ()
endforeach ()
