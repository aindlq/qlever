// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXEXTENSION_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXEXTENSION_H

#include <memory>
#include <string>
#include <string_view>

#include "services/vectorSearch/VectorIndex.h"
#include "util/HashMap.h"

class Index;

namespace qlever::vector {

// The name under which the loaded vector indices are stored on the `IndexImpl`
// via the generic extension mechanism (see `index/IndexExtension.h`).
inline constexpr std::string_view VECTOR_EXTENSION_NAME = "vectorSearch";

// All vector indices of a database, keyed by name. This is the object stored as
// the "vectorSearch" index extension and retrieved at query time.
class VectorIndexCollection {
 public:
  void add(const std::string& name, VectorIndex index) {
    indices_.insert_or_assign(name, std::move(index));
  }
  const VectorIndex* get(const std::string& name) const {
    auto it = indices_.find(name);
    return it == indices_.end() ? nullptr : &it->second;
  }

 private:
  ad_utility::HashMap<std::string, VectorIndex> indices_;
};

// Convenience for operations: the loaded `VectorIndex` named `name`, or an
// empty pointer. The returned `shared_ptr` aliases the whole collection, so the
// index stays valid even if the extension is ever replaced while in use.
std::shared_ptr<const VectorIndex> getVectorIndex(const Index& index,
                                                  const std::string& name);

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXEXTENSION_H
