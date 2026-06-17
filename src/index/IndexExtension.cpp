// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#include "index/IndexExtension.h"

IndexExtensionRegistry& IndexExtensionRegistry::get() {
  static IndexExtensionRegistry instance;
  return instance;
}

void IndexExtensionRegistry::addBuildHook(BuildHook hook) {
  buildHooks_.push_back(std::move(hook));
}

void IndexExtensionRegistry::addLoadHook(LoadHook hook) {
  loadHooks_.push_back(std::move(hook));
}
