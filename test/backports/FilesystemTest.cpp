// Copyright 2026 The QLever Authors

#include <gtest/gtest.h>

#include "backports/filesystem.h"

TEST(PathFilename, UsesStandardFilesystemSemantics) {
  using Path = ql::filesystem::path;

  EXPECT_EQ(ql::pathFilename(Path{"directory/file"}), Path{"file"});
  EXPECT_TRUE(ql::pathFilename(Path{"directory/"}).empty());
  EXPECT_TRUE(ql::pathFilename(Path{"/"}).empty());

  Path withPreferredSeparator{"directory"};
  withPreferredSeparator += Path::preferred_separator;
  EXPECT_TRUE(ql::pathFilename(withPreferredSeparator).empty());
}
