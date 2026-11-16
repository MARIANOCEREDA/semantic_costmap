#include <occupancy_grid_builder/occupancy_grid_builder.hpp>

#include <algorithm>
#include <cmath>
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

const std::vector<std::int8_t>& OccupancyGridBuilder::buildGridFromPCL(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    const Eigen::Matrix4f& pcl_origin_to_map_transform)
{
  // The grid is not reset: each scan adds evidence to the accumulated log-odds. Pass 1 marks
  // what this scan observed (at most one mark per cell), pass 2 applies one update per marked cell.

  // Compute sensor origin in map frame (sensor origin is at 0,0,0 in sensor frame)
  const Eigen::Vector4f sensor_origin_sensor_frame(0.0f, 0.0f, 0.0f, 1.0f);
  const Eigen::Vector4f sensor_origin_in_map_frame =
      pointToMap(sensor_origin_sensor_frame, pcl_origin_to_map_transform);
  const Eigen::Vector2i cell_origin = mapToGrid(sensor_origin_in_map_frame);

  for (const auto& point : cloud->points)
  {
    // NaN -> int is undefined and yields INT_MIN, making the Bresenham ray billions of cells long.
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
    {
      continue;
    }

    const Eigen::Vector4f point_sensor_frame(point.x, point.y, point.z, 1.0f);
    const Eigen::Vector4f point_in_map_frame =
        pointToMap(point_sensor_frame, pcl_origin_to_map_transform);
    const Eigen::Vector2i cell_hit = mapToGrid(point_in_map_frame);

    // Raycasting with Bresenham's line algorithm from the sensor origin to the point
    const auto ray_cells =
        bresenhamLine(cell_origin.x(), cell_origin.y(), cell_hit.x(), cell_hit.y());

    // The ray passed through every cell except the last one (the endpoint)
    for (std::size_t i = 0; i + 1 < ray_cells.size(); ++i)
    {
      markCell(ray_cells[i].x(), ray_cells[i].y(), ScanMark::kMiss);
    }

    // Endpoint: an obstacle hit if the point is colored, otherwise seen through (free)
    const bool is_obstacle = point.r > 0 || point.g > 0 || point.b > 0;
    markCell(cell_hit.x(), cell_hit.y(), is_obstacle ? ScanMark::kHit : ScanMark::kMiss);
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
  int grid_x = static_cast<int>((point_in_map_frame.x() - origin_x_) / resolution_);
  int grid_y = static_cast<int>((point_in_map_frame.y() - origin_y_) / resolution_);
  return Eigen::Vector2i(grid_x, grid_y);
}

std::vector<Eigen::Vector2i> OccupancyGridBuilder::bresenhamLine(int x0, int y0, int x1, int y1)
{
  std::vector<Eigen::Vector2i> line;
  int dx = std::abs(x1 - x0);
  int dy = std::abs(y1 - y0);
  int sx = (x0 < x1) ? 1 : -1;
  int sy = (y0 < y1) ? 1 : -1;
  int err = dx - dy;

  while (true)
  {
    line.emplace_back(x0, y0);

    if (x0 == x1 && y0 == y1)
      break;

    int e2 = 2 * err;
    if (e2 > -dy)
    {
      err -= dy;
      x0 += sx;
    }
    if (e2 < dx)
    {
      err += dx;
      y0 += sy;
    }
  }

  return line;
}

}  // namespace occupancy_grid_builder