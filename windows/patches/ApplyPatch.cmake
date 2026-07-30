# Apply a git patch to a (FetchContent) source tree in an idempotent way: a
# second run on an already-patched tree is a successful no-op. This matters
# because CMake re-runs the patch step of a fetched dependency under various
# circumstances (e.g. when the declared content details change).
#
# Usage: cmake -P ApplyPatch.cmake <patch-file> <directory>
#
# <directory> is interpreted relative to the current working directory
# (FetchContent runs patch commands in the root of the fetched source tree):
# pass `.` for the tree itself, or a subdirectory for patches that target a
# git submodule (`git -C <dir> apply` resolves the paths inside the patch
# against that submodule).
if (NOT CMAKE_ARGC EQUAL 5)
    message(FATAL_ERROR "Usage: cmake -P ApplyPatch.cmake <patch-file> <directory>")
endif ()
set(PATCH_FILE "${CMAKE_ARGV3}")
set(TARGET_DIR "${CMAKE_ARGV4}")

execute_process(
        COMMAND git -C "${TARGET_DIR}" apply --check "${PATCH_FILE}"
        RESULT_VARIABLE CAN_APPLY
        ERROR_QUIET
)
if (CAN_APPLY EQUAL 0)
    execute_process(
            COMMAND git -C "${TARGET_DIR}" apply "${PATCH_FILE}"
            RESULT_VARIABLE APPLY_RESULT
    )
    if (NOT APPLY_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to apply ${PATCH_FILE} in ${TARGET_DIR}")
    endif ()
    message(STATUS "Applied ${PATCH_FILE}")
else ()
    # If the patch is already applied, reverse-applying it would succeed.
    execute_process(
            COMMAND git -C "${TARGET_DIR}" apply --reverse --check "${PATCH_FILE}"
            RESULT_VARIABLE ALREADY_APPLIED
            ERROR_QUIET
    )
    if (ALREADY_APPLIED EQUAL 0)
        message(STATUS "Already applied: ${PATCH_FILE}")
    else ()
        message(FATAL_ERROR "Patch ${PATCH_FILE} does not apply in ${TARGET_DIR} "
                "(neither cleanly nor as already-applied). The pinned GIT_TAG of "
                "the dependency probably changed; regenerate the patch.")
    endif ()
endif ()
