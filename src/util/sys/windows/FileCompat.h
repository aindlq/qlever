// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_WINDOWS_FILECOMPAT_H
#define QLEVER_SRC_UTIL_SYS_WINDOWS_FILECOMPAT_H

#ifdef _WIN32

#include <fcntl.h>
#include <io.h>
#include <sys/types.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

#include "util/sys/windows/WindowsUndefs.h"

// POSIX-flavored file primitives for Windows. They give `ad_utility::File` the
// shared-delete and binary-open semantics that QLever relies on.

namespace ad_utility::windows {

// Delete a file with POSIX `unlink` semantics: the name is removed from the
// directory *immediately*, so a new file can be created under the same name
// right away, even while the old file still has open handles. Existing handles
// keep operating on the now-nameless file, exactly like Linux `unlink`.
//
// Why this is needed: QLever replaces index/metadata files while older handles
// can still be open. A plain Windows delete may leave the name delete-pending,
// so recreating it fails with ERROR_ACCESS_DENIED. A POSIX-semantics delete
// frees the name immediately.
//
// Prerequisite: every handle must have been opened with FILE_SHARE_DELETE.
// QLever's positioned-read handles and `openWithPosixSharing` do so.
//
// Availability: POSIX-semantics deletes require Windows 10 1607+/Server 2016+
// on NTFS. On older systems / other file systems `SetFileInformationByHandle`
// returns `false` and we fall back to a plain `DeleteFileW` (restoring the
// previous, delete-pending behavior - no worse than before).
// `FileDispositionInfoEx` and `FILE_DISPOSITION_INFO_EX` are only declared by
// the SDK/mingw-w64 headers when `NTDDI_VERSION >= 0x0A000002`, which this
// toolchain does not guarantee, so we declare the (ABI-stable, documented) info
// class and struct locally. `SetFileInformationByHandle` itself is available
// since Windows Vista
// (`_WIN32_WINNT >= 0x0600`, satisfied by the winsock2/bcrypt/Asio build).
//
// Returns true if the name is gone afterwards (deleted, or never existed).
inline bool posixDelete(const std::filesystem::path& filename) {
  // The `DELETE` (0x00010000) standard access right. The `DELETE` macro from
  // <winnt.h> is `#undef`-ed project-wide by `WindowsUndefs.h` (it collides
  // with the SPARQL `DELETE` token), so the numeric value is spelled out.
  constexpr DWORD deleteAccessRight = 0x00010000L;
  HANDLE handle =
      CreateFileW(filename.c_str(), deleteAccessRight,
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

  // Fallback for pre-1607 Windows or non-NTFS volumes: a plain delete.
  return DeleteFileW(filename.c_str()) != 0 ||
         GetLastError() == ERROR_FILE_NOT_FOUND;
}

// Open a file with POSIX-like sharing semantics. QLever (and its tests) rely
// on POSIX behavior where open files can still be renamed and removed
// (`unlink` semantics). The CRT's `fopen` never passes FILE_SHARE_DELETE, so
// we open via `CreateFileW` with full sharing and wrap the handle in a
// `FILE*`. All files are opened in binary mode (text mode would corrupt
// binary data via line-ending translation).
inline FILE* openWithPosixSharing(const std::filesystem::path& filename,
                                  const char* mode) {
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
      // POSIX rewrite semantics: unlink the old file first so the name can be
      // recreated immediately even while an older handle remains open.
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
  HANDLE handle =
      CreateFileW(filename.c_str(), access,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
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

inline FILE* openWithPosixSharing(const char* filename, const char* mode) {
  return openWithPosixSharing(std::filesystem::path{filename}, mode);
}

}  // namespace ad_utility::windows

#endif  // _WIN32

#endif  // QLEVER_SRC_UTIL_SYS_WINDOWS_FILECOMPAT_H
