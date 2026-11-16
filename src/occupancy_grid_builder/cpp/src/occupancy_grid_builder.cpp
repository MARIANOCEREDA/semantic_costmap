#include <occupancy_grid_builder/occupancy_grid_builder.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace occupancy_grid_builder
{
namespace
{
float Logit(double probability)
{
  return static_cast<float>(std::log(probability / (1.0 - probability)));
}
}  // namespace

OccupancyGridBuilder::~OccupancyGridBuilder() = default;

void OccupancyGridBuilder::setLogOddsParams(const LogOddsParams& params)
{
  const bool valid = params.miss_probability > 0.0 && params.miss_probability < 0.5 &&
                     params.hit_probability > 0.5 && params.hit_probability < 1.0 &&
                     params.min_probability > 0.0 &&
                     params.min_probability < params.free_threshold &&
                     params.free_threshold < params.occupied_threshold &&
                     params.occupied_threshold < params.max_probability &&
                     params.max_probability < 1.0;
  if (!valid)
  {
    throw std::invalid_argument(
        "OccupancyGridBuilder: invalid log-odds parameters, need 0 < miss < 0.5 < hit < 1 and "
        "0 < min_probability < free_threshold < occupied_threshold < max_probability < 1");
  }
  log_hit_ = Logit(params.hit_probability);
  log_miss_ = Logit(params.miss_probability);
  log_min_ = Logit(params.min_probability);
  log_max_ = Logit(params.max_probability);
  log_occupied_ = Logit(params.occupied_threshold);
  log_free_ = Logit(params.free_threshold);
}

void OccupancyGridBuilder::markCell(int x, int y, ScanMark mark)
{
  if (!isInGrid(x, y))
  {
    return;
  }
  const std::size_t index = cellIndex(x, y);
  ScanMark& current = scan_marks_[index];
  if (current == ScanMark::kNone)
  {
    touched_.push_back(index);
  }
  current = std::max(current, mark);
}

void OccupancyGridBuilder::applyScanMarks()
{
  for (const std::size_t index : touched_)
  {
    const float update = scan_marks_[index] == ScanMark::kHit ? log_hit_ : log_miss_;
    float& log_odds = log_odds_[index];
    log_odds = std::clamp(log_odds + update, log_min_, log_max_);
    cells_[index] = toCellValue(log_odds);
    scan_marks_[index] = ScanMark::kNone;
  }
  touched_.clear();
}

std::int8_t OccupancyGridBuilder::toCellValue(float log_odds) const
{
  if (log_odds >= log_occupied_)
  {
    return kOccupied;
  }
  if (log_odds <= log_free_)
  {
    return kFree;
  }
  // Uncertain: probability in percent. Strictly between the thresholds, so never 0 or 100.
  const float probability = 1.0f / (1.0f + std::exp(-log_odds));
  return static_cast<std::int8_t>(std::lround(100.0f * probability));
}

void OccupancyGridBuilder::setScanFilterParams(const ScanFilterParams& params)
{
  if (!(params.max_range > 0.0))
  {
    throw std::invalid_argument("OccupancyGridBuilder: max_range must be > 0");
  }
  scan_filter_ = params;
}

const std::vector<std::int8_t>& OccupancyGridBuilder::integrateScan(
    const PointCloudView& cloud, const Eigen::Matrix4f& sensor_to_map_transform)
{
  // The grid is not reset: each scan adds evidence to the accumulated log-odds. Pass 1 marks
  // what this scan observed (at most one mark per cell), pass 2 applies one update per marked cell.

  // Sensor origin in the map frame (the sensor sits at 0,0,0 of its own frame)
  const Eigen::Vector4f sensor_origin_in_map_frame =
      pointToMap(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f), sensor_to_map_transform);
  const Eigen::Vector2i cell_origin = mapToGrid(sensor_origin_in_map_frame);

  const auto max_range_sq = static_cast<float>(scan_filter_.max_range * scan_filter_.max_range);
  const auto max_obstacle_z =
      sensor_origin_in_map_frame.z() + static_cast<float>(scan_filter_.max_obstacle_height);

  for (std::size_t row = 0; row < cloud.height; ++row)
  {
    const std::uint8_t* row_ptr = cloud.data + row * cloud.row_step;
    for (std::size_t col = 0; col < cloud.width; ++col)
    {
      const std::uint8_t* point_ptr = row_ptr + col * cloud.point_step;

      // memcpy: the bytes may be unaligned and are not float objects (no reinterpret_cast).
      float x = 0.0f;
      float y = 0.0f;
      float z = 0.0f;
      std::memcpy(&x, point_ptr + cloud.x_offset, sizeof(float));
      std::memcpy(&y, point_ptr + cloud.y_offset, sizeof(float));
      std::memcpy(&z, point_ptr + cloud.z_offset, sizeof(float));
      const bool is_ground = point_ptr[cloud.ground_offset] != 0;

      // NaN -> int is undefined and yields INT_MIN, making the ray billions of cells long.
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
      {
        continue;
      }

      const Eigen::Vector4f point_in_map_frame =
          pointToMap(Eigen::Vector4f(x, y, z, 1.0f), sensor_to_map_transform);

      const float dx = point_in_map_frame.x() - sensor_origin_in_map_frame.x();
      const float dy = point_in_map_frame.y() - sensor_origin_in_map_frame.y();
      if (dx * dx + dy * dy > max_range_sq)
      {
        continue;
      }
      if (!is_ground && point_in_map_frame.z() > max_obstacle_z)
      {
        continue;
      }

      // The ray saw through every cell before the endpoint
      const Eigen::Vector2i cell_end = mapToGrid(point_in_map_frame);
      traceRay(cell_origin, cell_end);

      // Endpoint: ground is free space, anything else is an obstacle
      markCell(cell_end.x(), cell_end.y(), is_ground ? ScanMark::kMiss : ScanMark::kHit);
    }
  }

  applyScanMarks();
  return cells_;
}

Eigen::Vector4f OccupancyGridBuilder::pointToMap(const Eigen::Vector4f& point_sensor_frame,
                                                 const Eigen::Matrix4f& sensor_to_map_transform)
{
  return sensor_to_map_transform * point_sensor_frame;
}

Eigen::Vector2i OccupancyGridBuilder::mapToGrid(const Eigen::Vector4f& point_in_map_frame)
{
  // floor, not a cast: a cast truncates toward zero, so -0.5 cells would land in cell 0.
  const int grid_x =
      static_cast<int>(std::floor((point_in_map_frame.x() - origin_x_) / resolution_));
  const int grid_y =
      static_cast<int>(std::floor((point_in_map_frame.y() - origin_y_) / resolution_));
  return Eigen::Vector2i(grid_x, grid_y);
}

void OccupancyGridBuilder::traceRay(Eigen::Vector2i from, const Eigen::Vector2i& to)
{
  const int dx = std::abs(to.x() - from.x());
  const int dy = std::abs(to.y() - from.y());
  const int sx = (from.x() < to.x()) ? 1 : -1;
  const int sy = (from.y() < to.y()) ? 1 : -1;
  int err = dx - dy;

  while (from != to)
  {
    markCell(from.x(), from.y(), ScanMark::kMiss);

    const int e2 = 2 * err;
    if (e2 > -dy)
    {
      err -= dy;
      from.x() += sx;
    }
    if (e2 < dx)
    {
      err += dx;
      from.y() += sy;
    }
  }
}

}  // namespace occupancy_grid_builder
