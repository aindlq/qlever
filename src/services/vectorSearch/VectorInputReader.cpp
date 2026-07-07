// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorInputReader.h"

#include <array>
#include <charconv>
#include <cstring>

#include "services/vectorSearch/VectorIndexFormat.h"
#include "util/Exception.h"

namespace qlever::vector {

namespace {
// Parse a decimal unsigned integer, throwing a descriptive error on garbage.
uint64_t parseUnsigned(std::string_view s, std::string_view what) {
  uint64_t value = 0;
  auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
  if (ec != std::errc{} || ptr != s.data() + s.size()) {
    AD_THROW("Could not parse the " + std::string{what} +
             " of the .npy header as a number: \"" + std::string{s} + "\"");
  }
  return value;
}

// Parse the `'shape': (N, D)` tuple out of a NumPy header dictionary string.
std::pair<uint64_t, uint32_t> parseShape(const std::string& header) {
  auto pos = header.find("'shape'");
  if (pos == std::string::npos) {
    AD_THROW("The .npy header has no shape field.");
  }
  pos = header.find('(', pos);
  auto end =
      pos == std::string::npos ? std::string::npos : header.find(')', pos);
  if (end == std::string::npos) {
    AD_THROW("The shape in the .npy header is malformed.");
  }
  std::string tuple = header.substr(pos + 1, end - pos - 1);
  // `tuple` is like "200, 16" (possibly with a trailing comma for 1-D).
  uint64_t n = 0;
  uint32_t d = 0;
  int field = 0;
  std::string cur;
  auto flush = [&]() {
    // Trim whitespace; skip empty fields (e.g. after a trailing comma).
    auto first = cur.find_first_not_of(" \t");
    if (first == std::string::npos) return;
    auto last = cur.find_last_not_of(" \t");
    std::string_view token{cur.data() + first, last - first + 1};
    if (field == 0) {
      n = parseUnsigned(token, "number of rows");
    } else if (field == 1) {
      d = static_cast<uint32_t>(parseUnsigned(token, "dimension"));
    }
    ++field;
    cur.clear();
  };
  for (char c : tuple) {
    if (c == ',') {
      flush();
    } else {
      cur += c;
    }
  }
  flush();
  if (field != 2) {
    AD_THROW("The .npy input must be a 2-D array of shape (N, D); got shape (" +
             tuple + ")");
  }
  return {n, d};
}
}  // namespace

// ____________________________________________________________________________
NpyVectorInputReader::NpyVectorInputReader(const std::string& npyPath,
                                           const std::string& irisPath)
    : npy_{npyPath, std::ios::binary}, iris_{irisPath} {
  if (!npy_.is_open()) {
    AD_THROW("Could not open the .npy file " + npyPath);
  }
  if (!iris_.is_open()) {
    AD_THROW("Could not open the IRI list file " + irisPath);
  }

  // Magic string + version.
  std::array<char, 8> prelude{};
  npy_.read(prelude.data(), prelude.size());
  if (npy_.gcount() != 8 || prelude[0] != '\x93' || prelude[1] != 'N' ||
      prelude[2] != 'U' || prelude[3] != 'M' || prelude[4] != 'P' ||
      prelude[5] != 'Y') {
    AD_THROW("Not a valid .npy file: " + npyPath);
  }
  uint8_t major = static_cast<uint8_t>(prelude[6]);
  if (major < 1 || major > 3) {
    AD_THROW("Unsupported .npy format version " + std::to_string(major) +
             " in " + npyPath);
  }

  // Header length: 2 bytes (v1) or 4 bytes (v2+), little-endian.
  uint32_t headerLen = 0;
  if (major == 1) {
    std::array<uint8_t, 2> b{};
    npy_.read(reinterpret_cast<char*>(b.data()), 2);
    if (npy_.gcount() != 2) {
      AD_THROW("Unexpected end of file in the .npy header of " + npyPath);
    }
    headerLen =
        static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8);
  } else {
    std::array<uint8_t, 4> b{};
    npy_.read(reinterpret_cast<char*>(b.data()), 4);
    if (npy_.gcount() != 4) {
      AD_THROW("Unexpected end of file in the .npy header of " + npyPath);
    }
    headerLen = static_cast<uint32_t>(b[0]) |
                (static_cast<uint32_t>(b[1]) << 8) |
                (static_cast<uint32_t>(b[2]) << 16) |
                (static_cast<uint32_t>(b[3]) << 24);
  }
  // A .npy header is a short ASCII dict; cap it so a corrupt length field
  // cannot request a multi-GB allocation before the (bounded) read.
  if (headerLen > (1u << 20)) {
    AD_THROW("The .npy header of " + npyPath + " is implausibly large (" +
             std::to_string(headerLen) + " bytes).");
  }
  std::string header(headerLen, '\0');
  npy_.read(header.data(), headerLen);
  if (npy_.gcount() != static_cast<std::streamsize>(headerLen)) {
    AD_THROW("Unexpected end of file in the .npy header of " + npyPath);
  }

  // We only support little-endian C-order matrices of float32 ('<f4') or --
  // for bfloat16 matrices written via `ml_dtypes` (numpy has no native bf16
  // type code) -- the opaque 2-byte void '<V2', whose value bytes are the
  // little-endian bf16 bit pattern (the top 16 bits of the fp32). Match the
  // `descr` VALUE exactly (a substring check would accept a structured dtype
  // that merely contains such a field and then silently misread the row
  // stride).
  auto descrIs = [&header](std::string_view value) {
    size_t pos = header.find("'descr'");
    if (pos == std::string::npos) return false;
    pos = header.find(':', pos);
    if (pos == std::string::npos) return false;
    pos = header.find_first_not_of(" ", pos + 1);
    if (pos == std::string::npos) return false;
    for (std::string_view quote : {"'", "\""}) {
      std::string quoted =
          std::string{quote} + std::string{value} + std::string{quote};
      if (header.compare(pos, quoted.size(), quoted) == 0) return true;
    }
    return false;
  };
  if (descrIs("<f4")) {
    bytesPerScalar_ = sizeof(float);
  } else if (descrIs("<V2")) {
    bytesPerScalar_ = 2;
  } else {
    AD_THROW(
        ".npy input must be little-endian float32 ('<f4') or bfloat16 "
        "('<V2', as written by ml_dtypes.bfloat16). Header: " +
        header.substr(0, 200));
  }
  if (header.find("'fortran_order': False") == std::string::npos) {
    AD_THROW(".npy input must be C-order (fortran_order: False).");
  }

  auto [n, d] = parseShape(header);
  numRows_ = n;
  dimensions_ = d;
  if (dimensions_ == 0) {
    AD_THROW("The .npy input " + npyPath + " has zero dimension.");
  }
  if (dimensions_ > MAX_VECTOR_DIMENSIONS) {
    AD_THROW("The .npy input " + npyPath +
             " declares an implausible dimension (" +
             std::to_string(dimensions_) + ").");
  }
  dataOffset_ = static_cast<uint64_t>(npy_.tellg());
}

// ____________________________________________________________________________
bool NpyVectorInputReader::next(std::string& iri, std::vector<float>& vector) {
  if (rowsRead_ >= numRows_) {
    // Detect a longer-than-expected IRI list once at the end -- a silent
    // mismatch would mean vectors were attached to the wrong entities.
    std::string extra;
    while (std::getline(iris_, extra)) {
      if (extra.find_first_not_of(" \t\r") != std::string::npos) {
        AD_THROW(
            "The IRI list has more (non-empty) lines than the .npy input "
            "has rows (" +
            std::to_string(numRows_) + ").");
      }
    }
    return false;
  }
  // IRI line.
  if (!std::getline(iris_, iri)) {
    AD_THROW("The IRI list has fewer lines (" + std::to_string(rowsRead_) +
             ") than the .npy input has rows (" + std::to_string(numRows_) +
             ").");
  }
  // Trim surrounding whitespace (also handles the '\r' of CRLF inputs). The
  // line is either a bare IRI or an iriref `<...>`; yield the bare form, the
  // build pass re-brackets it for resolution against the vocabulary.
  constexpr std::string_view whitespace = " \t\r\n";
  size_t first = iri.find_first_not_of(whitespace);
  if (first == std::string::npos) {
    AD_THROW("Line " + std::to_string(rowsRead_ + 1) +
             " of the IRI list is empty, but the .npy input has a row for "
             "it.");
  }
  size_t last = iri.find_last_not_of(whitespace);
  iri = iri.substr(first, last - first + 1);
  if (iri.size() >= 2 && iri.front() == '<' && iri.back() == '>') {
    iri = iri.substr(1, iri.size() - 2);
  }
  // The D scalars of this row (`bytesPerScalar_` bytes each).
  const auto rowBytes =
      static_cast<std::streamsize>(dimensions_) * bytesPerScalar_;
  vector.resize(dimensions_);
  const bool isF32 = bytesPerScalar_ == sizeof(float);
  if (isF32) {
    npy_.read(reinterpret_cast<char*>(vector.data()), rowBytes);
  } else {
    rawRow_.resize(static_cast<size_t>(rowBytes));
    npy_.read(rawRow_.data(), rowBytes);
  }
  if (npy_.gcount() != rowBytes) {
    AD_THROW("Unexpected end of data in the .npy input at row " +
             std::to_string(rowsRead_) + " of " + std::to_string(numRows_) +
             ".");
  }
  if (!isF32) {
    // Decode each little-endian bf16 bit pattern: a bf16 is exactly the top
    // 16 bits of the corresponding fp32, so widening is lossless.
    for (uint32_t j = 0; j < dimensions_; ++j) {
      const auto bits = static_cast<uint16_t>(
          static_cast<uint8_t>(rawRow_[2 * j]) |
          (static_cast<uint8_t>(rawRow_[2 * j + 1]) << 8));
      const uint32_t widened = static_cast<uint32_t>(bits) << 16;
      float value;
      std::memcpy(&value, &widened, sizeof(value));
      vector[j] = value;
    }
  }
  ++rowsRead_;
  return true;
}

}  // namespace qlever::vector
