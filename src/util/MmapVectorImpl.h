// Copyright 2018, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Johannes Kalmbach <johannes.kalmbach@gmail.com>

#ifndef QLEVER_SRC_UTIL_MMAPVECTOR_IMPL_H
#define QLEVER_SRC_UTIL_MMAPVECTOR_IMPL_H

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdio>
#include <utility>

#include "util/MmapVector.h"

namespace ad_utility {
// definition of static constants
template <class T>
constexpr size_t MmapVector<T>::MinCapacity;

template <class T>
constexpr float MmapVector<T>::ResizeFactor;

// __________________________________________________________________________
template <class T>
void MmapVector<T>::writeMetaDataToEnd() {
  // Truncate away any previous trailer, then append the new one below.
  if (FileMapping::resizeFile(_filename.c_str(), _bytesize)) {
    throw TruncateException(_filename, _bytesize, errno);
  }

  // Open the file for updating (mode "r+" does not truncate it), seek to the
  // end of the array space, and write the metadata trailer there.
  File file{_filename, "r+"};
  file.seek(static_cast<off_t>(_bytesize), SEEK_SET);
  MmapVectorMetaData{_size, _capacity, _bytesize}.writeToFile(file);
}

// __________________________________________________________________________
template <class T>
void MmapVector<T>::readMetaDataFromEnd() {
  File file{_filename, "r"};
  const auto metaData = MmapVectorMetaData::readFromFile(file);
  _size = metaData.size_;
  _capacity = metaData.capacity_;
  _bytesize = metaData.bytesize_;
}

// ________________________________________________________________
template <class T>
void MmapVector<T>::mapForReading() {
  _ptr = static_cast<T*>(
      fileMapping_.map(_filename, _bytesize, /*writable=*/false));
  advise(_pattern);
}

// ________________________________________________________________
template <class T>
void MmapVector<T>::mapForWriting() {
  _ptr = static_cast<T*>(
      fileMapping_.map(_filename, _bytesize, /*writable=*/true));
  advise(_pattern);
}
// ________________________________________________________________
template <class T>
void MmapVector<T>::remapLinux(size_t oldBytesize) {
  // Only ever called on Linux (see `adaptCapacity`).
  _ptr = static_cast<T*>(
      fileMapping_.remap(static_cast<void*>(_ptr), oldBytesize, _bytesize));
  advise(_pattern);
}

// __________________________________________________________________
template <class T>
VecInfo MmapVector<T>::convertArraySizeToFileSize(size_t targetSize) const {
  // calculate the smallest multiple of pagesize that can fit all the bytes
  // needed for the array.
  size_t pagesize = FileMapping::pageSize();
  size_t bytesize = targetSize * sizeof(T);
  // align size to pagesize. This might be |pagesize| bigger than necessary in
  // the case of an exact match, but this waste of some kb should not be bad
  bytesize = ((bytesize / pagesize) + 1) * pagesize;
  // the number of elements which can actually be fit in bytesize bytes
  size_t capacity = bytesize / sizeof(T);
  VecInfo res;
  res._capacity = capacity;
  res._bytesize = bytesize;
  return res;
}

// _________________________________________________________________
template <class T>
void MmapVector<T>::adaptCapacity(size_t newCapacity) {
  throwIfUninitialized();
  auto oldBytesize = _bytesize;
  auto realSize = convertArraySizeToFileSize(newCapacity);
  _capacity = realSize._capacity;
  _bytesize = realSize._bytesize;
  // writeMetaData will also adapt the filesize according to
  // _capacity and _bytesize

#ifdef __linux__
  writeMetaDataToEnd();
  remapLinux(oldBytesize);
#else
  // The `unmap()` function requires that the `_bytesize` hasn't changed since
  // the last call to `mapForWriting()`, so we have to restore the old
  // `_bytesize` temporarily. Otherwise we get a subtle bug that triggers the
  // address sanitizers on all macOS machines and segfaults on M1 machines.
  _bytesize = oldBytesize;
  unmap();
  _bytesize = realSize._bytesize;
  writeMetaDataToEnd();
  // renew the mapping because the file has changed
  mapForWriting();
#endif
}

// _________________________________________________________________
template <class T>
void MmapVector<T>::resize(size_t newSize) {
  _size = newSize;
  if (newSize > capacity()) {
    adaptCapacity(newSize);
  }
}

// _________________________________________________________________
template <class T>
void MmapVector<T>::push_back(T&& el) {
  if (_size == _capacity) {
    adaptCapacity(_capacity * ResizeFactor);
    AD_CONTRACT_CHECK(_capacity > _size);
  }
  _ptr[_size] = std::move(el);
  ++_size;
}

// _________________________________________________________________
template <class T>
void MmapVector<T>::push_back(const T& el) {
  if (_size == _capacity) {
    adaptCapacity(_capacity * ResizeFactor);
    AD_CONTRACT_CHECK(_capacity > _size);
  }
  _ptr[_size] = el;
  ++_size;
}

// ________________________________________________________________
template <class T>
void MmapVector<T>::open(size_t size, const T& defaultValue,
                         std::string filename, AccessPattern pattern) {
  open(size, filename, pattern);
  advise(AccessPattern::Sequential);
  for (size_t i = 0; i < _size; ++i) {
    _ptr[i] = defaultValue;
  }
  advise(_pattern);
}

// ________________________________________________________________
template <class T>
template <class It>
void MmapVector<T>::open(It begin, It end, const std::string& filename,
                         AccessPattern pattern) {
  open(end - begin, filename, pattern);
  advise(AccessPattern::Sequential);
  size_t i = 0;
  for (auto it = begin; it != end; ++it) {
    _ptr[i] = *it;
    ++i;
  }
  advise(pattern);
}

// _______________________________________________________________________
template <class T>
void MmapVector<T>::open(size_t size, std::string filename,
                         AccessPattern pattern) {
  unmap();
  _size = size;
  _filename = std::move(filename);
  _pattern = pattern;

  // open the file in case it does not exist yet
  // (data will be overwritten anyway)
  { auto ofs = ad_utility::makeOfstream(_filename); }
  auto info = convertArraySizeToFileSize(std::max(size, MinCapacity));
  _bytesize = info._bytesize;
  _capacity = info._capacity;
  // also sets fileSize correctly
  writeMetaDataToEnd();
  mapForWriting();
}

// _____________________________________________________________
template <class T>
void MmapVector<T>::open(std::string filename, ReuseTag,
                         AccessPattern pattern) {
  unmap();
  _filename = std::move(filename);
  _pattern = pattern;
  readMetaDataFromEnd();
  mapForWriting();
}

// ________________________________________________________________
template <class T>
void MmapVector<T>::close() {
  // we need the correct size to make the file persistent
  if (_ptr != nullptr) {
    fileMapping_.persistAndUnmap([this] { writeMetaDataToEnd(); },
                                 [this] { unmap(); });
  }
  _filename = "";
  _size = 0;
  _ptr = nullptr;
  _bytesize = 0;
  _capacity = 0;
}

// _______________________________________________________________
template <class T>
MmapVector<T>::~MmapVector() {
  std::string message = absl::StrCat(
      "Error while unmapping a file with name \"", _filename, "\"");
  ad_utility::terminateIfThrows([this]() { this->close(); }, message);
}

// ________________________________________________________________
template <class T>
void MmapVector<T>::unmap() {
  if (_ptr != nullptr) {
    fileMapping_.unmap(static_cast<void*>(_ptr), _bytesize);
    _ptr = nullptr;
  }
}

// ________________________________________________________________
template <class T>
MmapVector<T>::MmapVector(MmapVector<T>&& other) noexcept = default;

// ________________________________________________________________
template <class T>
MmapVector<T>& MmapVector<T>::operator=(MmapVector<T>&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  // if this vector already has a mapping, close it correctly
  close();
  _ptr = std::move(other._ptr);
  _size = std::move(other._size);
  _capacity = std::move(other._capacity);
  _bytesize = std::move(other._bytesize);
  _filename = std::move(other._filename);
  _pattern = std::move(other._pattern);
  fileMapping_ = std::move(other.fileMapping_);
  return *this;
}

// ________________________________________________________________
template <class T>
void MmapVector<T>::advise(AccessPattern pattern) {
  fileMapping_.advise(static_cast<void*>(_ptr), _bytesize, pattern);
}

// ________________________________________________________________
// there is much code duplication with these operations to their equivalents in
// MmapVector. But since we have chosen the "greedy" template constructors,
// this is necessary.
template <class T>
MmapVectorView<T>::MmapVectorView(MmapVectorView<T>&& other) noexcept = default;

// ________________________________________________________________
template <class T>
MmapVectorView<T>& MmapVectorView<T>::operator=(
    MmapVectorView<T>&& other) noexcept {
  // if this vector already has a mapping, close it correctly
  close();
  this->_ptr = std::move(other._ptr);
  this->_size = std::move(other._size);
  this->_capacity = std::move(other._capacity);
  this->_bytesize = std::move(other._bytesize);
  this->_filename = std::move(other._filename);
  this->_pattern = std::move(other._pattern);
  this->fileMapping_ = std::move(other.fileMapping_);
  return *this;
}

template <class T>
void MmapVectorView<T>::close() {
  // we need the correct size to make the file persistent
  if (this->_ptr != nullptr) {
    MmapVector<T>::unmap();
  }
  this->_filename = "";
  this->_size = 0;
  this->_ptr = nullptr;
  this->_bytesize = 0;
  this->_capacity = 0;
}

}  // namespace ad_utility

#endif  // header guard
