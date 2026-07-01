// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

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
