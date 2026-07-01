// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINPUTREADER_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINPUTREADER_H

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace qlever::vector {

// Streams `(iri, vector)` rows from a bulk input file for index building. The
// concrete format is an implementation detail; the build pass resolves each
// `iri` to a `VocabIndex` against the knowledge graph's vocabulary.
//
// `next` fills `iri` and `vector` and returns true, or returns false at EOF.
class VectorInputReader {
 public:
  virtual ~VectorInputReader() = default;
  virtual uint32_t dimensions() const = 0;
  virtual uint64_t numRows() const = 0;
  virtual bool next(std::string& iri, std::vector<float>& vector) = 0;
};

// Reads a 2-D little-endian float32 NumPy `.npy` matrix (shape `(N, D)`,
// C-order) together with a sidecar text file of `N` IRIs, one per line, where
// line `i` labels row `i` of the matrix. This is the dependency-free default
// ingest format (a Parquet reader using Apache Arrow is the planned production
// alternative; see the implementation log). `np.save("vectors.npy", arr)` plus
// writing the IRIs to a text file is all it takes to produce.
class NpyVectorInputReader : public VectorInputReader {
 public:
  NpyVectorInputReader(const std::string& npyPath, const std::string& irisPath);

  uint32_t dimensions() const override { return dimensions_; }
  uint64_t numRows() const override { return numRows_; }
  bool next(std::string& iri, std::vector<float>& vector) override;

 private:
  uint32_t dimensions_ = 0;
  uint64_t numRows_ = 0;
  uint64_t rowsRead_ = 0;
  uint64_t dataOffset_ = 0;  // byte offset of the float data in the .npy file
  std::ifstream npy_;
  std::ifstream iris_;
};

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINPUTREADER_H
