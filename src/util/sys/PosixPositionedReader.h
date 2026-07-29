// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_POSIXPOSITIONEDREADER_H
#define QLEVER_SRC_UTIL_SYS_POSIXPOSITIONEDREADER_H

#include <sys/types.h>
#include <unistd.h>

#include <cstddef>

namespace ad_utility::posix {

class PositionedReader {
 public:
  ssize_t readAtOffset(int fd, void* buffer, size_t count, off_t offset) const {
    return ::pread(fd, buffer, count, offset);
  }

  void close() {}
};

}  // namespace ad_utility::posix

#endif  // QLEVER_SRC_UTIL_SYS_POSIXPOSITIONEDREADER_H
