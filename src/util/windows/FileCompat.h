// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_WINDOWS_FILECOMPAT_H
#define QLEVER_SRC_UTIL_WINDOWS_FILECOMPAT_H

#ifdef _WIN32

#include <fcntl.h>
#include <io.h>
#include <sys/types.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <string>

#include "util/windows/WindowsUndefs.h"

// POSIX-flavored file primitives for Windows. Everything in this header
// exists to give `ad_utility::File` (and the `MmapVector` machinery) the
// POSIX file semantics that QLever relies on: shared-delete opens, binary
// mode, positioned reads that don't disturb the file position, and
// truncation by path.

namespace ad_utility::windows {

// Delete a file with POSIX `unlink` semantics: the name is removed from the
// directory *immediately*, so a new file can be created under the same name
// right away, even while the old file still has open handles or an active
// (share-delete) memory mapping. Existing handles/mappings keep operating on
// the now-nameless file, exactly like Linux `unlink`.
//
// Why this is needed: QLever rewrites index/metadata files in place while an
// older version may still be memory-mapped (e.g. a materialized view's
// `.index.spo.meta`, mapped read-only by a loaded `MmapVectorView`). A *plain*
// Windows delete of such a file (`DeleteFileA`, or `std::filesystem::remove`,
// or `SetFileInformationByHandle(FileDispositionInfo)`) only marks it
// DELETE-PENDING: the directory entry survives until the last section/handle
// closes, and every attempt to (re)create a file with that name in the
// meantime fails with STATUS_DELETE_PENDING -> ERROR_ACCESS_DENIED. That is the
// root cause of the "Could not open file ... for writing" failures when a
// mapped view is rewritten. A POSIX-semantics delete frees the name at once and
// removes the race entirely.
//
// Prerequisite (already satisfied in QLever): every handle/mapping of the file
// must have been opened with FILE_SHARE_DELETE - otherwise this open (which
// requests DELETE access) fails with a sharing violation, and the delete itself
// would return STATUS_CANNOT_DELETE. QLever's positioned-read handles/mappings
// (`util/windows/WindowsPositionedReader.h`), `openWithPosixSharing` and
// `MmapVector`'s share-delete map all use FILE_SHARE_DELETE, so this holds for
// every file QLever may rewrite.
//
// Availability: POSIX-semantics deletes require Windows 10 1607+/Server 2016+
// on NTFS. On older systems / other file systems `SetFileInformationByHandle`
// returns `false` and we fall back to a plain `DeleteFileA` (restoring the
// previous, delete-pending behavior - no worse than before).
// `FileDispositionInfoEx` and `FILE_DISPOSITION_INFO_EX` are only declared by
// the SDK/mingw-w64 headers when `NTDDI_VERSION >= 0x0A000002`, which this
// toolchain does not guarantee, so we declare the (ABI-stable, documented) info
// class and struct locally. `SetFileInformationByHandle` itself is available
// since Windows Vista
// (`_WIN32_WINNT >= 0x0600`, satisfied by the winsock2/bcrypt/Asio build).
//
// Returns true if the name is gone afterwards (deleted, or never existed).
inline bool posixDelete(const char* filename) {
  // The `DELETE` (0x00010000) standard access right. The `DELETE` macro from
  // <winnt.h> is `#undef`-ed project-wide by `WindowsUndefs.h` (it collides
  // with the SPARQL `DELETE` token), so the numeric value is spelled out.
  constexpr DWORD deleteAccessRight = 0x00010000L;
  HANDLE handle =
      CreateFileA(filename, deleteAccessRight,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    const DWORD lastError = GetLastError();
    // Nothing to delete (already gone) counts as success, matching the
    // "overwrite/truncate" intent of the callers.
    return lastError == ERROR_FILE_NOT_FOUND ||
           lastError == ERROR_PATH_NOT_FOUND;
  }

  // Locally-declared equivalents of the Win10-era SDK definitions (see comment
  // above). Values from ntddk.h / minwinbase.h and are part of the stable ABI.
  struct FileDispositionInfoExData {
    DWORD Flags;
  };
  constexpr auto fileDispositionInfoExClass =
      static_cast<FILE_INFO_BY_HANDLE_CLASS>(21);  // FileDispositionInfoEx
  constexpr DWORD flagDelete = 0x00000001;  // FILE_DISPOSITION_FLAG_DELETE
  constexpr DWORD flagPosixSemantics = 0x00000002;  // ..._FLAG_POSIX_SEMANTICS
  constexpr DWORD flagIgnoreReadonly =
      0x00000010;  // ..._FLAG_IGNORE_READONLY_ATTRIBUTE
  FileDispositionInfoExData info{flagDelete | flagPosixSemantics |
                                 flagIgnoreReadonly};
  bool posixOk = SetFileInformationByHandle(handle, fileDispositionInfoExClass,
                                            &info, sizeof(info));
  if (!posixOk) {
    // `FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE` was only added in
    // Windows 10 1809 (RS5); on 1607-1803 the combined call fails with
    // ERROR_INVALID_PARAMETER even though POSIX-semantics delete itself is
    // supported. Retry without the ignore-readonly flag so those systems still
    // get the POSIX unlink (matching the availability the comment above
    // claims).
    info.Flags = flagDelete | flagPosixSemantics;
    posixOk = SetFileInformationByHandle(handle, fileDispositionInfoExClass,
                                         &info, sizeof(info));
  }
  CloseHandle(handle);
  if (posixOk) {
    // Name is already unlinked from the namespace at this point.
    return true;
  }

  // Fallback for pre-1607 Windows or non-NTFS volumes: a plain delete. If the
  // file is still mapped this only marks it delete-pending (the pre-existing
  // behavior), but on a supported system we never reach here.
  return DeleteFileA(filename) != 0 || GetLastError() == ERROR_FILE_NOT_FOUND;
}

// Open a file with POSIX-like sharing semantics. QLever (and its tests) rely
// on POSIX behavior where open files can still be renamed and removed
// (`unlink` semantics). The CRT's `fopen` never passes FILE_SHARE_DELETE, so
// we open via `CreateFileA` with full sharing and wrap the handle in a
// `FILE*`. All files are opened in binary mode (text mode would corrupt
// binary data via line-ending translation).
inline FILE* openWithPosixSharing(const char* filename, const char* mode) {
  std::string modeString{mode};
  bool plus = modeString.find('+') != std::string::npos;
  char primary = modeString.empty() ? 'r' : modeString[0];
  DWORD access = 0;
  DWORD creation = 0;
  int osfFlags = _O_BINARY;
  const char* fdopenMode = nullptr;
  switch (primary) {
    case 'r':
      access = plus ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ;
      creation = OPEN_EXISTING;
      if (!plus) osfFlags |= _O_RDONLY;
      fdopenMode = plus ? "r+b" : "rb";
      break;
    case 'w':
      access = plus ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_WRITE;
      creation = CREATE_ALWAYS;
      fdopenMode = plus ? "w+b" : "wb";
      // POSIX rewrite semantics: unlink the old file first instead of
      // truncating it in place. Truncation fails on Windows when the file
      // still has active memory mappings (e.g. an older version of the same
      // index file is still loaded). A POSIX-semantics `unlink` (see
      // `posixDelete`) detaches the name *immediately* and leaves existing
      // mappings working on the old contents - exactly what happens on Linux -
      // so the `CreateFileA` below can recreate the name right away. A plain
      // `DeleteFileA` here would only mark the file delete-pending while it is
      // mapped, and the recreate would then fail with ERROR_ACCESS_DENIED.
      posixDelete(filename);
      break;
    case 'a':
      access = plus ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_WRITE;
      creation = OPEN_ALWAYS;
      osfFlags |= _O_APPEND;
      fdopenMode = plus ? "a+b" : "ab";
      break;
    default:
      errno = EINVAL;
      return nullptr;
  }
  HANDLE handle = CreateFileA(
      filename, access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      nullptr, creation, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    DWORD lastError = GetLastError();
    errno =
        (lastError == ERROR_FILE_NOT_FOUND || lastError == ERROR_PATH_NOT_FOUND)
            ? ENOENT
        : (lastError == ERROR_ACCESS_DENIED) ? EACCES
                                             : EIO;
    return nullptr;
  }
  int fd = _open_osfhandle((intptr_t)handle, osfFlags);
  if (fd == -1) {
    CloseHandle(handle);
    errno = EIO;
    return nullptr;
  }
  FILE* file = _fdopen(fd, fdopenMode);
  if (file == nullptr) {
    _close(fd);
    errno = EIO;
    return nullptr;
  }
  if (primary == 'a') {
    fseek(file, 0, SEEK_END);
  }
  return file;
}

}  // namespace ad_utility::windows

// Emulation of POSIX `truncate` (resize a file by path), which does not
// exist on Windows. Deliberately placed in the global namespace under the
// POSIX name so that call sites stay portable.
inline int truncate(const char* path, int64_t length) {
  // Short-circuit when the size wouldn't change. Windows cannot resize a
  // file that has active memory mappings (even from the same process), but a
  // no-op "resize" doesn't need to. This matters because multiple
  // `MmapVector`s may map the same fixed-size metadata file concurrently.
  WIN32_FILE_ATTRIBUTE_DATA attributes;
  if (GetFileAttributesExA(path, GetFileExInfoStandard, &attributes)) {
    int64_t currentSize = static_cast<int64_t>(
        (static_cast<uint64_t>(attributes.nFileSizeHigh) << 32) |
        attributes.nFileSizeLow);
    if (currentSize == length) {
      return 0;
    }
  }
  int fd = _open(path, _O_RDWR | _O_BINARY);
  if (fd < 0) return -1;
  int err = _chsize_s(fd, length);
  _close(fd);
  if (err != 0) {
    errno = err;
    return -1;
  }
  return 0;
}

#endif  // _WIN32

#endif  // QLEVER_SRC_UTIL_WINDOWS_FILECOMPAT_H
