// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#include "services/vectorSearch/VectorInputReader.h"

#include <array>
#include <stdexcept>

#include "util/Exception.h"

namespace qlever::vector {

namespace {
// Parse the `'shape': (N, D)` tuple out of a NumPy header dictionary string.
std::pair<uint64_t, uint32_t> parseShape(const std::string& header) {
  auto pos = header.find("'shape'");
  AD_CONTRACT_CHECK(pos != std::string::npos, ".npy header has no shape field");
  pos = header.find('(', pos);
  AD_CONTRACT_CHECK(pos != std::string::npos, ".npy shape is malformed");
  auto end = header.find(')', pos);
  AD_CONTRACT_CHECK(end != std::string::npos, ".npy shape is malformed");
  std::string tuple = header.substr(pos + 1, end - pos - 1);
  // tuple is like "200, 16" (possibly with a trailing comma for 1-D).
  uint64_t n = 0;
  uint32_t d = 0;
  int field = 0;
  std::string cur;
  auto flush = [&]() {
    if (cur.find_first_not_of(" \t") == std::string::npos) return;
    if (field == 0) {
      n = std::stoull(cur);
    } else if (field == 1) {
      d = static_cast<uint32_t>(std::stoul(cur));
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
  AD_CONTRACT_CHECK(field == 2,
                    ".npy must be a 2-D array of shape (N, D); got: ", tuple);
  return {n, d};
}
}  // namespace

NpyVectorInputReader::NpyVectorInputReader(const std::string& npyPath,
                                           const std::string& irisPath)
    : npy_{npyPath, std::ios::binary}, iris_{irisPath} {
  AD_CONTRACT_CHECK(npy_.is_open(), "Could not open .npy file ", npyPath);
  AD_CONTRACT_CHECK(iris_.is_open(), "Could not open IRI list file ", irisPath);

  // Magic string + version.
  std::array<char, 8> prelude{};
  npy_.read(prelude.data(), prelude.size());
  AD_CONTRACT_CHECK(npy_.gcount() == 8 && prelude[0] == '\x93' &&
                        prelude[1] == 'N' && prelude[2] == 'U' &&
                        prelude[3] == 'M' && prelude[4] == 'P' &&
                        prelude[5] == 'Y',
                    "Not a valid .npy file: ", npyPath);
  uint8_t major = static_cast<uint8_t>(prelude[6]);

  // Header length: 2 bytes (v1) or 4 bytes (v2+), little-endian.
  uint32_t headerLen = 0;
  if (major == 1) {
    std::array<uint8_t, 2> b{};
    npy_.read(reinterpret_cast<char*>(b.data()), 2);
    headerLen = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8);
  } else {
    std::array<uint8_t, 4> b{};
    npy_.read(reinterpret_cast<char*>(b.data()), 4);
    headerLen = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
                (static_cast<uint32_t>(b[2]) << 16) |
                (static_cast<uint32_t>(b[3]) << 24);
  }
  std::string header(headerLen, '\0');
  npy_.read(header.data(), headerLen);

  // We only support little-endian float32, C-order.
  AD_CONTRACT_CHECK(header.find("'<f4'") != std::string::npos ||
                        header.find("\"<f4\"") != std::string::npos,
                    ".npy must be little-endian float32 ('<f4'). Header: ",
                    header);
  AD_CONTRACT_CHECK(header.find("'fortran_order': False") != std::string::npos,
                    ".npy must be C-order (fortran_order: False).");

  auto [n, d] = parseShape(header);
  numRows_ = n;
  dimensions_ = d;
  AD_CONTRACT_CHECK(dimensions_ > 0, ".npy has zero dimension.");
  dataOffset_ = static_cast<uint64_t>(npy_.tellg());
}

bool NpyVectorInputReader::next(std::string& iri, std::vector<float>& vector) {
  if (rowsRead_ >= numRows_) {
    return false;
  }
  // IRI line.
  if (!std::getline(iris_, iri)) {
    throw std::runtime_error(
        "IRI list has fewer lines than the .npy has rows.");
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
  AD_CONTRACT_CHECK(npy_.gcount() ==
                        static_cast<std::streamsize>(dimensions_ * sizeof(float)),
                    "Unexpected end of .npy data.");
  ++rowsRead_;
  return true;
}

}  // namespace qlever::vector
