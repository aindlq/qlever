// Copyright 2011, University of Freiburg, Chair of Algorithms and Data
// Structures.
// Author: Björn Buchhold <buchholb>

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "util/File.h"

namespace ad_utility {
TEST(File, move) {
  std::string filename = "testFileMove.tmp";
  File file1(filename, "w");
  ASSERT_TRUE(file1.isOpen());
  file1.write("aaa", 3);
  EXPECT_EQ(file1.name(), "testFileMove.tmp");

  File file2;
  ASSERT_TRUE(file1.isOpen());
  ASSERT_FALSE(file2.isOpen());
  file2 = std::move(file1);
  ASSERT_FALSE(file1.isOpen());
  ASSERT_TRUE(file2.isOpen());

  file2.write("bbb", 3);
  File file3(std::move(file2));
  ASSERT_FALSE(file2.isOpen());
  ASSERT_TRUE(file3.isOpen());
  file3.write("ccc", 3);
  file3.close();

  File fileRead(filename, "r");
  ASSERT_TRUE(fileRead.isOpen());
  std::string s;
  s.resize(2);
  auto numBytes = fileRead.read(s.data(), 2);
  ASSERT_EQ(numBytes, 2u);
  ASSERT_EQ(s, "aa");

  File fileRead2;
  fileRead2 = std::move(fileRead);
  s.resize(5);
  numBytes = fileRead2.read(s.data(), 5);
  ASSERT_EQ(numBytes, 5u);
  ASSERT_EQ(s, "abbbc");

  File fileRead3{std::move(fileRead2)};
  s.resize(2);
  numBytes = fileRead3.read(s.data(), 2);
  ASSERT_EQ(numBytes, 2u);
  ASSERT_EQ(s, "cc");

  ASSERT_EQ(0u, fileRead3.read(s.data(), 9));
  ad_utility::deleteFile(filename);
}
}  // namespace ad_utility

TEST(File, makeFilestream) {
  std::string filename = "makeFilstreamTest.dat";
  ad_utility::makeOfstream(filename) << "helloAgain\n";
  std::string s;
  auto reader = ad_utility::makeIfstream(filename);
  ASSERT_TRUE(reader.is_open());
  ASSERT_TRUE(std::getline(reader, s));
  ASSERT_EQ("helloAgain", s);
  ASSERT_FALSE(std::getline(reader, s));

  // Throw on nonexisting file
  ASSERT_THROW(ad_utility::makeIfstream("nonExisting1620349.datxyz"),
               std::runtime_error);
}

// `ad_utility::File` must do binary I/O (openWithPosixSharing passes
// `_O_BINARY`). In text mode, writing would insert `\r` before each `\n` and
// reading would stop at byte 0x1A (ctrl-Z = EOF), silently corrupting binary
// data. On POSIX this is a no-op; on Windows it guards the `_O_BINARY` flag.
TEST(File, BinaryModePreservesSpecialBytes) {
  std::string filename = "testFileBinary.tmp";
  const std::array<unsigned char, 10> bytes = {0x00, 0x0D, 0x0A, 0x1A, 0xFF,
                                               0x0A, 0x0D, 0x1A, 0x41, 0x00};
  {
    ad_utility::File w(filename, "w");
    w.write(bytes.data(), bytes.size());
    w.close();
  }
  // Text mode would inflate the file (each `\n` -> `\r\n`).
  EXPECT_EQ(std::filesystem::file_size(filename), bytes.size());
  ad_utility::File r(filename, "r");
  std::array<unsigned char, 10> got{};
  EXPECT_EQ(r.read(got.data(), got.size(), 0),
            static_cast<ssize_t>(bytes.size()));  // not truncated at 0x1A
  EXPECT_EQ(got, bytes);                          // byte-identical round trip
  r.close();
  ad_utility::deleteFile(filename);
}

// The process-wide binary default (`_set_fmode(_O_BINARY)` in
// `ProcessInit.cpp`): raw `fopen()`/`open()` without `"b"`/`_O_BINARY` must
// still be binary, because dependencies (e.g. libspatialjoin) open binary
// files that way. Guards the global fmode on Windows; a no-op on POSIX.
TEST(File, RawFopenDefaultsToBinaryMode) {
  std::string filename = "testFileFmode.tmp";
  const unsigned char bytes[8] = {0x0A, 0x0D, 0x1A, 0x00,
                                  0xFF, 0x0A, 0x1A, 0x42};
  FILE* w = std::fopen(filename.c_str(), "w");  // NB: no "b"
  ASSERT_NE(w, nullptr);
  ASSERT_EQ(std::fwrite(bytes, 1, sizeof(bytes), w), sizeof(bytes));
  std::fclose(w);
  EXPECT_EQ(std::filesystem::file_size(filename), sizeof(bytes));  // no `\r`
  FILE* r = std::fopen(filename.c_str(), "r");                     // NB: no "b"
  ASSERT_NE(r, nullptr);
  unsigned char got[8] = {};
  EXPECT_EQ(std::fread(got, 1, sizeof(got), r),
            sizeof(got));  // not cut at 0x1A
  std::fclose(r);
  EXPECT_EQ(std::memcmp(got, bytes, sizeof(bytes)), 0);
  ad_utility::deleteFile(filename);
}

// POSIX unlink-while-open semantics (`FILE_SHARE_DELETE` in
// `openWithPosixSharing`): an open file must be deletable — QLever's
// index-rebuild logic relies on it. On Windows this fails without
// `FILE_SHARE_DELETE`; on POSIX it always works.
TEST(File, OpenFileCanBeDeletedWhileOpen) {
  std::string filename = "testFileShareDelete.tmp";
  ad_utility::File f(filename, "w");
  f.write("xyz", 3);
  std::error_code ec;
  const bool removed = std::filesystem::remove(filename, ec);
  EXPECT_TRUE(removed) << "could not delete an open file (FILE_SHARE_DELETE?): "
                       << ec.message();
  f.close();
}
