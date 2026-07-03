// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_WINDOWS_WINDOWSFILEMAPPING_H
#define QLEVER_SRC_UTIL_WINDOWS_WINDOWSFILEMAPPING_H

#ifdef _WIN32

#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <cstdint>
#include <string>
#include <utility>

#include "util/Exception.h"
#include "util/ResetWhenMoved.h"
#include "util/windows/FileCompat.h"  // for the `truncate` shim

namespace ad_utility::windows {

// The Windows file-mapping primitive used by `MmapVector` (the override; the
// default is `ad_utility::posix::FileMapping`). Boost.Interprocess replaces
// POSIX `mmap`, but this class owns the backing file handle itself so it can
// pass FILE_SHARE_DELETE (Boost's own `file_mapping` does not), keeping a
// mapped file deletable and rewritable like a POSIX unlink-while-mapped. See
// `util/FileMapping.h` for the compile-time seam selecting between the two.
class FileMapping {
 private:
  // A default-constructed region is an empty placeholder that maps nothing.
  boost::interprocess::mapped_region mapping_;
  // The FILE_SHARE_DELETE handle backing `mapping_`, kept open for the
  // mapping's lifetime and closed in `unmap`.
  ResetWhenMoved<void*, nullptr> fileHandle_;

 public:
  // Windows cannot resize a file that still has an active mapping
  // (ERROR_USER_MAPPED_FILE), so the `MmapVector` must unmap before it
  // truncates.
  static constexpr bool kMustUnmapBeforeResize = true;

  FileMapping() = default;
  // Moving transfers the region and its backing handle; the moved-from object
  // is left empty (its `unmap` becomes a no-op).
  FileMapping(FileMapping&&) noexcept = default;
  FileMapping& operator=(FileMapping&&) noexcept = default;
  FileMapping(const FileMapping&) = delete;
  FileMapping& operator=(const FileMapping&) = delete;

  // Map `bytesize` bytes of `filename` and return the mapped address. We open
  // the backing file ourselves with FILE_SHARE_DELETE so it stays deletable
  // and rewritable while mapped; `read_write` corresponds to `MAP_SHARED` with
  // `PROT_READ | PROT_WRITE` (updates are written back to the file).
  void* map(const std::string& filename, size_t bytesize, bool writable) {
    namespace bip = boost::interprocess;
    DWORD access = writable ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ;
    HANDLE handle =
        CreateFileA(filename.c_str(), access,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    AD_CONTRACT_CHECK(handle != INVALID_HANDLE_VALUE);
    auto mode = writable ? bip::read_write : bip::read_only;
    // `mapped_region` accepts any "MemoryMappable" exposing
    // get_mapping_handle() and get_mode(); wrap our own share-delete handle in
    // one. The concrete Boost handle type is internal, so deduce it via
    // `decltype`.
    auto mappingHandle =
        bip::ipcdetail::mapping_handle_from_file_handle(handle);
    struct HandleMappable {
      decltype(mappingHandle) handle_;
      decltype(mode) mode_;
      decltype(mappingHandle) get_mapping_handle() const { return handle_; }
      decltype(mode) get_mode() const { return mode_; }
    } mappable{mappingHandle, mode};
    mapping_ = bip::mapped_region{mappable, mode, 0, bytesize};
    fileHandle_ = handle;
    return mapping_.get_address();
  }

  // Release the mapping. Replacing the region by an empty one unmaps the view
  // and flushes written data; then close the FILE_SHARE_DELETE handle.
  void unmap(void* /*ptr*/, size_t /*bytesize*/) {
    mapping_ = boost::interprocess::mapped_region{};
    if (fileHandle_ != nullptr) {
      CloseHandle(static_cast<HANDLE>(static_cast<void*>(fileHandle_)));
      fileHandle_ = nullptr;
    }
  }

  // In-place remap is Linux-only; `MmapVector` never calls this on Windows.
  void* remap(void* /*ptr*/, size_t /*oldBytesize*/, size_t /*newBytesize*/) {
    AD_FAIL();
  }

  // Like `madvise`, only a hint; Boost.Interprocess forwards it where supported
  // and returns `false` (ignored here) otherwise.
  void advise(void* /*ptr*/, size_t /*bytesize*/, AccessPattern pattern) {
    using Region = boost::interprocess::mapped_region;
    switch (pattern) {
      case AccessPattern::Sequential:
        mapping_.advise(Region::advice_sequential);
        break;
      case AccessPattern::Random:
        mapping_.advise(Region::advice_random);
        break;
      default:
        mapping_.advise(Region::advice_normal);
        break;
    }
  }

  // Mapped views must be aligned to this granularity (64 KiB on Windows, which
  // is Boost.Interprocess' notion of the "page size").
  static size_t pageSize() {
    return boost::interprocess::mapped_region::get_page_size();
  }

  // Resize the backing file by path (see the `truncate` shim in
  // `FileCompat.h`). Returns 0 on success.
  static int resizeFile(const char* path, int64_t length) {
    return ::truncate(path, length);
  }
};

}  // namespace ad_utility::windows

#endif  // _WIN32
#endif  // QLEVER_SRC_UTIL_WINDOWS_WINDOWSFILEMAPPING_H
