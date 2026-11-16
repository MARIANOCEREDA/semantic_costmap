#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

namespace occupancy_grid_builder
{
// Inverse sensor model and clamping of the log-odds occupancy update, given as probabilities.
// Per scan, every cell gets at most one update: "hit" if a ray ends in it on an obstacle, else
// "miss" if a ray passes through it (a hit wins). Log-odds are clamped to
// [min_probability, max_probability] so the map keeps adapting: a cell at the upper clamp turns
// free again after a bounded number of misses.
// Must hold: 0 < miss < 0.5 < hit < 1 and
//            0 < min_probability < free_threshold < occupied_threshold < max_probability < 1.
struct LogOddsParams
{
  double hit_probability{0.7};
  double miss_probability{0.4};
  double min_probability{0.12};
  double max_probability{0.97};
  double occupied_threshold{0.65};  // published as 100 at or above this
  double free_threshold{0.2};       // published as 0 at or below this
};

class OccupancyGridBuilder
{
 public:
  // Cell values follow nav_msgs/OccupancyGrid, so the grid can be published with a plain copy.
  static constexpr std::int8_t kUnknown = -1;
  static constexpr std::int8_t kFree = 0;
  static constexpr std::int8_t kOccupied = 100;

  OccupancyGridBuilder(double resolution = 0.1,
                       int width = 1000,
                       int height = 1000,
                       double origin_x = -50.0,
                       double origin_y = -50.0)
    : resolution_(resolution),
      width_(width),
      height_(height),
      origin_x_(origin_x),
      origin_y_(origin_y),
      log_odds_(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0.0f),
      scan_marks_(log_odds_.size(), ScanMark::kNone),
      cells_(log_odds_.size(), kUnknown)
  {
    setLogOddsParams(LogOddsParams{});
  }
  ~OccupancyGridBuilder();

  // Throws std::invalid_argument if `params` violates the constraints of LogOddsParams.
  void setLogOddsParams(const LogOddsParams& params);

  // Integrates one scan into the accumulated grid and returns it: width * height cells, row-major
  // (index = y * width + x), in nav_msgs/OccupancyGrid encoding: -1 never observed, 0 free,
  // 100 occupied, 1..99 uncertain (occupancy probability in percent). The reference stays valid for the builder's lifetime; its
  // contents change on the next call.
  const std::vector<std::int8_t>& buildGridFromPCL(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                                                  const Eigen::Matrix4f& pcl_origin_to_map_transform);
  Eigen::Vector4f pointToMap(const Eigen::Vector4f& point_sensor_frame,
                             const Eigen::Matrix4f& sensor_to_map_transform);
  Eigen::Vector2i mapToGrid(const Eigen::Vector4f& point_in_map_frame);
  std::vector<Eigen::Vector2i> bresenhamLine(int x0, int y0, int x1, int y1);

 private:
  // What the current scan observed in a cell. Ordered so that a hit wins over a miss.
  enum class ScanMark : std::uint8_t
  {
    kNone = 0,
    kMiss = 1,
    kHit = 2,
  };

  // Records `mark` for cell (x, y) in the current scan; ignores cells outside the grid.
  void markCell(int x, int y, ScanMark mark);
  // Applies one log-odds update to every cell marked in this scan, then clears the marks.
  void applyScanMarks();
  std::int8_t toCellValue(float log_odds) const;

  bool isInGrid(int x, int y) const { return x >= 0 && x < width_ && y >= 0 && y < height_; }
  std::size_t cellIndex(int x, int y) const
  {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) +
           static_cast<std::size_t>(x);
  }

  double resolution_ = 0.1;  // meters/cell — 0.1m is a common default
  int width_ = 1000;         // cells → 100m
  int height_ = 1000;        // cells → 100m
  double origin_x_ = -50.0;  // meters, grid origin offset from map frame origin
  double origin_y_ = -50.0;

  // LogOddsParams converted to log-odds once (set by setLogOddsParams).
  float log_hit_ = 0.0f;
  float log_miss_ = 0.0f;
  float log_min_ = 0.0f;
  float log_max_ = 0.0f;
  float log_occupied_ = 0.0f;
  float log_free_ = 0.0f;

  // All grids: one contiguous block each, row-major, width_ * height_ cells. Declared after
  // width_/height_ (members are initialized in declaration order).
  // Accumulated log-odds of occupancy; 0 = probability 0.5.
  std::vector<float> log_odds_;
  // Per-scan marks; all kNone between scans.
  std::vector<ScanMark> scan_marks_;
  // Indices of the cells marked in this scan, so applying and clearing the marks costs
  // O(touched cells) instead of a pass over the whole grid. Capacity is kept across scans.
  std::vector<std::size_t> touched_;
  // Published values (OccupancyGrid encoding); kUnknown until a cell is first observed.
  std::vector<std::int8_t> cells_;
};
}  // namespace occupancy_grid_builder