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

#ifdef QLEVER_WITH_PARQUET

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>

namespace qlever::vector {

// All Arrow state lives here so that the header stays Arrow-free.
struct ParquetVectorInputReader::Impl {
  std::shared_ptr<arrow::RecordBatchReader> batches_;
  std::shared_ptr<arrow::RecordBatch> batch_;
  int64_t batchRow_ = 0;
  int uriCol_ = -1;
  int embeddingCol_ = -1;
  uint64_t numRows_ = 0;
  uint64_t rowsRead_ = 0;
  uint32_t dimensions_ = 0;
  // The reader must outlive the batch reader.
  std::unique_ptr<parquet::arrow::FileReader> reader_;
};

namespace {
// Unwraps an `arrow::Result`, translating errors into QLever exceptions.
template <typename T>
T expectArrow(arrow::Result<T> result, const std::string& what) {
  if (!result.ok()) {
    AD_THROW(what + ": " + result.status().ToString());
  }
  return std::move(result).ValueUnsafe();
}

int findColumn(const arrow::Schema& schema,
               const std::vector<std::string>& names,
               const std::string& description) {
  for (const auto& name : names) {
    int index = schema.GetFieldIndex(name);
    if (index >= 0) {
      return index;
    }
  }
  AD_THROW("The Parquet input has no " + description + " column (expected \"" +
           names.front() + "\").");
}
}  // namespace

// ____________________________________________________________________________
ParquetVectorInputReader::ParquetVectorInputReader(
    const std::string& parquetPath)
    : impl_{std::make_unique<Impl>()} {
  auto file = expectArrow(arrow::io::ReadableFile::Open(parquetPath),
                          "Could not open the Parquet file " + parquetPath);
  impl_->reader_ = expectArrow(
      parquet::arrow::OpenFile(std::move(file), arrow::default_memory_pool()),
      "Could not read the Parquet file " + parquetPath);
  std::shared_ptr<arrow::Schema> schema;
  auto status = impl_->reader_->GetSchema(&schema);
  if (!status.ok()) {
    AD_THROW("Could not read the schema of the Parquet file " + parquetPath +
             ": " + status.ToString());
  }
  impl_->uriCol_ = findColumn(*schema, {"uri", "iri"}, "URI");
  impl_->embeddingCol_ =
      findColumn(*schema, {"embedding", "vector"}, "embedding");
  const auto& embeddingType = *schema->field(impl_->embeddingCol_)->type();
  if (embeddingType.id() == arrow::Type::FIXED_SIZE_LIST) {
    impl_->dimensions_ = static_cast<uint32_t>(
        static_cast<const arrow::FixedSizeListType&>(embeddingType)
            .list_size());
  }
  impl_->numRows_ = static_cast<uint64_t>(
      impl_->reader_->parquet_reader()->metadata()->num_rows());
  impl_->batches_ =
      expectArrow(impl_->reader_->GetRecordBatchReader(),
                  "Could not iterate the Parquet file " + parquetPath);
}

ParquetVectorInputReader::~ParquetVectorInputReader() = default;

// ____________________________________________________________________________
uint32_t ParquetVectorInputReader::dimensions() const {
  return impl_->dimensions_;
}
uint64_t ParquetVectorInputReader::numRows() const { return impl_->numRows_; }

// ____________________________________________________________________________
bool ParquetVectorInputReader::next(std::string& iri,
                                    std::vector<float>& vector) {
  auto& impl = *impl_;
  while (!impl.batch_ || impl.batchRow_ >= impl.batch_->num_rows()) {
    impl.batch_ = expectArrow(impl.batches_->Next(),
                              "Could not read the next Parquet record batch");
    impl.batchRow_ = 0;
    if (!impl.batch_) {
      return false;
    }
  }
  const int64_t row = impl.batchRow_++;
  const uint64_t rowNumber = ++impl.rowsRead_;

  // The URI.
  const auto& uriArray = *impl.batch_->column(impl.uriCol_);
  if (uriArray.IsNull(row)) {
    AD_THROW("Row " + std::to_string(rowNumber) +
             " of the Parquet input has a null URI.");
  }
  if (uriArray.type_id() == arrow::Type::STRING) {
    iri = static_cast<const arrow::StringArray&>(uriArray).GetString(row);
  } else if (uriArray.type_id() == arrow::Type::LARGE_STRING) {
    iri = static_cast<const arrow::LargeStringArray&>(uriArray).GetString(row);
  } else {
    AD_THROW("The URI column of the Parquet input must be a string column.");
  }

  // The embedding: a (fixed-size/large) list of float32 or float64.
  const auto& embeddingArray = *impl.batch_->column(impl.embeddingCol_);
  if (embeddingArray.IsNull(row)) {
    AD_THROW("Row " + std::to_string(rowNumber) +
             " of the Parquet input has a null embedding.");
  }
  std::shared_ptr<arrow::Array> values;
  int64_t offset = 0;
  int64_t length = 0;
  switch (embeddingArray.type_id()) {
    case arrow::Type::LIST: {
      const auto& list = static_cast<const arrow::ListArray&>(embeddingArray);
      values = list.values();
      offset = list.value_offset(row);
      length = list.value_length(row);
      break;
    }
    case arrow::Type::LARGE_LIST: {
      const auto& list =
          static_cast<const arrow::LargeListArray&>(embeddingArray);
      values = list.values();
      offset = list.value_offset(row);
      length = list.value_length(row);
      break;
    }
    case arrow::Type::FIXED_SIZE_LIST: {
      const auto& list =
          static_cast<const arrow::FixedSizeListArray&>(embeddingArray);
      values = list.values();
      length = list.value_length(row);
      offset = list.value_offset(row);
      break;
    }
    default:
      AD_THROW(
          "The embedding column of the Parquet input must be a list of "
          "float32 or float64.");
  }
  if (impl.dimensions_ == 0) {
    impl.dimensions_ = static_cast<uint32_t>(length);
    if (impl.dimensions_ == 0) {
      AD_THROW("The Parquet input has zero-dimensional embeddings.");
    }
  }
  if (length != static_cast<int64_t>(impl.dimensions_)) {
    AD_THROW("Row " + std::to_string(rowNumber) +
             " of the Parquet input has an embedding of length " +
             std::to_string(length) + ", but earlier rows had " +
             std::to_string(impl.dimensions_) + ".");
  }
  vector.resize(impl.dimensions_);
  if (values->type_id() == arrow::Type::FLOAT) {
    const auto& floats = static_cast<const arrow::FloatArray&>(*values);
    for (int64_t j = 0; j < length; ++j) {
      vector[j] = floats.Value(offset + j);
    }
  } else if (values->type_id() == arrow::Type::DOUBLE) {
    const auto& doubles = static_cast<const arrow::DoubleArray&>(*values);
    for (int64_t j = 0; j < length; ++j) {
      vector[j] = static_cast<float>(doubles.Value(offset + j));
    }
  } else {
    AD_THROW(
        "The embedding column of the Parquet input must contain float32 or "
        "float64 values.");
  }
  return true;
}

}  // namespace qlever::vector

#endif  // QLEVER_WITH_PARQUET
