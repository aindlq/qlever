# Windows-only platform support layer (this directory is only added on
# Windows, see the top-level CMakeLists.txt).
#
# `qleverWindowsInit` contains process-wide runtime fixes that every QLever
# executable must link:
#   - ProcessInit.cpp: binary-mode default for CRT file opens and a precise
#     replacement for winpthreads' `nanosleep64` (see the file for details).
#     The replacement is activated via the `--wrap` linker flag below.
#   - LongPathAware.rc: embeds a manifest that declares the executables
#     long-path aware.
# The INTERFACE link options are inherited by every target that links this
# library; `link_libraries(qleverWindowsInit)` in the top-level
# CMakeLists.txt establishes that for all QLever targets.
enable_language(RC)
# Let the resource compiler find the manifest referenced by the .rc file.
set(CMAKE_RC_FLAGS "${CMAKE_RC_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}")
add_library(qleverWindowsInit OBJECT ProcessInit.cpp LongPathAware.rc)
target_link_options(qleverWindowsInit INTERFACE
        # Route `std::this_thread::sleep_for/sleep_until` (which compile to
        # calls to winpthreads' `nanosleep64`) through the precise sleep in
        # ProcessInit.cpp.
        -Wl,--wrap=nanosleep64
        # 16 MiB main-thread stack: MinGW's 2 MiB default is too small for
        # QLever's recursion-heavy query parsing and planning (for
        # comparison, the Linux default is 8 MiB).
        -Wl,--stack,16777216)
