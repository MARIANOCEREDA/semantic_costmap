#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace occupancy_layer_builder
{
// Non-owning view over a packed point buffer (e.g. the bytes of a sensor_msgs/PointCloud2).
// Points are `point_step` bytes apart, rows `row_step` bytes apart. x/y/z are native-endian float32
// and `ground` is a uint8 (1 = ground, 0 = not ground) at the given byte offsets inside each point.
// The buffer must outlive every use of the view; the view never copies or frees it.
struct PointCloudView
{
  const std::uint8_t* data{nullptr};
  std::size_t width{0};
  std::size_t height{0};
  std::size_t point_step{0};
  std::size_t row_step{0};
  std::size_t x_offset{0};
  std::size_t y_offset{0};
  std::size_t z_offset{0};
  std::size_t ground_offset{0};
};

// Which points of a scan are integrated into the grid.
struct ScanFilterParams
{
  // Points farther than this from the sensor (horizontal distance, metres) are ignored. Bounds the
  // ray length, i.e. the cost per point, and drops sparse, noisy far returns.
  double max_range{20.0};
  // Non-ground points higher than this above the sensor (map z, metres) are ignored: overhangs
  // (branches, ceilings) are not obstacles at ground level and say nothing about the floor below.
  double max_obstacle_height{2.0};
};

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

class OccupancyLayerBuilder
{
 public:
  // Cell values follow nav_msgs/OccupancyGrid, so the grid can be published with a plain copy.
  static constexpr std::int8_t kUnknown = -1;
  static constexpr std::int8_t kFree = 0;
  static constexpr std::int8_t kOccupied = 100;

  OccupancyLayerBuilder(double resolution = 0.1,
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
  ~OccupancyLayerBuilder();

  // Throws std::invalid_argument if `params` violates the constraints of LogOddsParams.
  void setLogOddsParams(const LogOddsParams& params);
  // Throws std::invalid_argument unless max_range > 0.
  void setScanFilterParams(const ScanFilterParams& params);

  // Integrates one scan (points in the sensor frame, origin at the sensor) into the accumulated
  // grid. Every ray from the sensor marks the cells it passes as free evidence; its endpoint is
  // free evidence for a ground point and occupied evidence for a non-ground point.
  // Returns the grid: width * height cells, row-major (index = y * width + x), in
  // nav_msgs/OccupancyGrid encoding: -1 never observed, 0 free, 100 occupied, 1..99 uncertain
  // (occupancy probability in percent). The reference stays valid for the builder's lifetime;
  // its contents change on the next call.
  const std::vector<std::int8_t>& integrateScan(const PointCloudView& cloud,
                                                const Eigen::Matrix4f& sensor_to_map_transform);
  Eigen::Vector4f pointToMap(const Eigen::Vector4f& point_sensor_frame,
                             const Eigen::Matrix4f& sensor_to_map_transform);
  Eigen::Vector2i mapToGrid(const Eigen::Vector4f& point_in_map_frame);

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
  // Marks every cell on the Bresenham line from `from` to `to` as a miss, excluding `to` itself.
  // Walks the line in place: no list of cells is built.
  void traceRay(Eigen::Vector2i from, const Eigen::Vector2i& to);
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

  ScanFilterParams scan_filter_;

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
}  // namespace occupancy_layer_builder