// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

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
