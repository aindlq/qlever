// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "engine/MagicServicePlanning.h"

MagicServicePlannerRegistry& MagicServicePlannerRegistry::get() {
  static MagicServicePlannerRegistry instance;
  return instance;
}

void MagicServicePlannerRegistry::add(std::type_index type,
                                      MagicServicePlanner planner) {
  planners_.insert_or_assign(type, std::move(planner));
}

const MagicServicePlanner* MagicServicePlannerRegistry::lookup(
    std::type_index type) const {
  auto it = planners_.find(type);
  return it == planners_.end() ? nullptr : &it->second;
}
