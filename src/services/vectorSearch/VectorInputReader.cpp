// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorInputReader.h"

#include <array>
#include <charconv>

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
  std::string header(headerLen, '\0');
  npy_.read(header.data(), headerLen);
  if (npy_.gcount() != static_cast<std::streamsize>(headerLen)) {
    AD_THROW("Unexpected end of file in the .npy header of " + npyPath);
  }

  // We only support little-endian float32, C-order. (The raw little-endian
  // float read below matches QLever's general little-endian assumption.)
  if (header.find("'<f4'") == std::string::npos &&
      header.find("\"<f4\"") == std::string::npos) {
    AD_THROW(".npy input must be little-endian float32 ('<f4'). Header: " +
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
  // Trim a trailing '\r' (CRLF inputs) and surrounding whitespace.
  while (!iri.empty() && (iri.back() == '\r' || iri.back() == ' ' ||
                          iri.back() == '\t' || iri.back() == '\n')) {
    iri.pop_back();
  }
  // The D float32 values of this row.
  vector.resize(dimensions_);
  npy_.read(reinterpret_cast<char*>(vector.data()),
            static_cast<std::streamsize>(dimensions_ * sizeof(float)));
  if (npy_.gcount() !=
      static_cast<std::streamsize>(dimensions_ * sizeof(float))) {
    AD_THROW("Unexpected end of data in the .npy input at row " +
             std::to_string(rowsRead_) + " of " + std::to_string(numRows_) +
             ".");
  }
  ++rowsRead_;
  return true;
}

}  // namespace qlever::vector
