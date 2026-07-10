// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_POSIXPOSITIONEDREADER_H
#define QLEVER_SRC_UTIL_SYS_POSIXPOSITIONEDREADER_H

#include <sys/types.h>
#include <unistd.h>

#include <cstddef>
#include <string>

namespace ad_utility::posix {

// POSIX positioned reads for `File`, served by `pread`.
class PositionedReader {
 public:
  // Read up to `count` bytes at `offset` via `pread` on `fd`. `filename` is
  // unused on POSIX. Returns the number of bytes read, 0 at EOF, -1 on error.
  ssize_t readAtOffset(int fd, const std::string& filename, void* buffer,
                       size_t count, off_t offset) const {
    (void)filename;
    return ::pread(fd, buffer, count, offset);
  }

  // No positioned-read state to release.
  void close() {}
};

}  // namespace ad_utility::posix

#endif  // QLEVER_SRC_UTIL_SYS_POSIXPOSITIONEDREADER_H
