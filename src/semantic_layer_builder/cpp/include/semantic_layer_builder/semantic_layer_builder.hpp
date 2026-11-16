#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace semantic_layer_builder
{
// Semantic layer: one class id per grid cell (0 = no class), filled from labeled points
// (/points_labeled, label = class_id + 1). Says *what* a cell is; occupancy (whether it is
// there) comes from occupancy_layer_builder. Scaffolding only: the class is added in the next step.
}  // namespace semantic_layer_builder
