// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_INDEX_INDEXEXTENSION_H
#define QLEVER_SRC_INDEX_INDEXEXTENSION_H

#include <functional>
#include <string>
#include <vector>

#include "util/json.h"

class Index;
class IndexImpl;

// Registry that lets a service attach an auxiliary on-disk index to QLever's
// main index, without the core index/build code knowing about the service. A
// service registers (from its own translation unit; see `src/services/`):
//   * a BUILD hook  -- run by `qlever index` AFTER the main index is built,
//   with
//     the freshly built index loaded, so URI->Id resolution is available. It
//     reads its slice of the build configuration JSON and writes its files.
//   * a LOAD hook   -- run at server/engine start, after the index is fully
//     loaded (vocabulary, permutations, and persisted updates). It memory-maps
//     its files and stores the result on the `IndexImpl` via `setExtension`,
//     to be retrieved at query time via `getExtension`. Because the index is
//     fully loaded, a load hook may also insert delta triples (e.g.
//     auto-materialized metadata triples describing its auxiliary index).
//
// The whole config JSON is passed to every build hook; each hook picks its own
// key (its service name), so a generic `--service-index <json>` CLI option
// suffices (no per-service option in the core index builder).
class IndexExtensionRegistry {
 public:
  // (loadedIndex, indexBasename, fullConfigJson) -> writes the service's files.
  using BuildHook = std::function<void(const Index&, const std::string&,
                                       const nlohmann::json&)>;
  // (indexImpl, indexBasename) -> memory-maps + stores via `setExtension`.
  using LoadHook = std::function<void(IndexImpl&, const std::string&)>;

  static IndexExtensionRegistry& get();

  void addBuildHook(BuildHook hook);
  void addLoadHook(LoadHook hook);

  const std::vector<BuildHook>& buildHooks() const { return buildHooks_; }
  const std::vector<LoadHook>& loadHooks() const { return loadHooks_; }

 private:
  std::vector<BuildHook> buildHooks_;
  std::vector<LoadHook> loadHooks_;
};

#endif  // QLEVER_SRC_INDEX_INDEXEXTENSION_H
