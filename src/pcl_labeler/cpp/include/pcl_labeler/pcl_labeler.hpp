#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <cstdint>
#include <pcl/filters/statistical_outlier_removal.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl_labeler
{
// Non-owning view over a packed point buffer (e.g. the bytes of a sensor_msgs/PointCloud2).
// Points are `point_step` bytes apart, rows are `row_step` bytes apart, and x/y/z are native-endian
// float32 at the given byte offsets inside each point. The buffer must outlive every use of the
// view; the view never copies or frees it.
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

  std::size_t size() const { return width * height; }
};

// Statistical outlier removal, applied separately to the points of each mask value (object).
// For every point, the mean distance to its `mean_k` nearest neighbours is computed; points whose
// mean distance exceeds mean + stddev_mul * stddev (statistics of that object) are removed.
struct OutlierRemovalParams
{
  bool enabled{true};
  int mean_k{10};
  double stddev_mul{1.0};
};

class PCLLabeler
{
 public:
  PCLLabeler();
  void set_extrinsics(const Eigen::Matrix4f& extrinsics);
  void set_intrinsics(const Eigen::Matrix3f& intrinsics);
  void set_outlier_removal(const OutlierRemovalParams& params);
  // Labels the points of `cloud` that project onto non-zero pixels of `image` and writes them to
  // `output`. The caller owns `output`; it is cleared (capacity kept) and refilled on every call.
  // `cloud` must not point into `output`'s memory.
  void fuse(const cv::Mat& image,
            const PointCloudView& cloud,
            pcl::PointCloud<pcl::PointXYZRGB>& output);

 private:
  // Erodes `mask` into the reused member buffer and returns a reference to it. The reference is
  // only valid until the next call. A pixel keeps its value only if its whole kernel
  // neighbourhood has that same value; otherwise it becomes 0. Unlike a plain erode (a min
  // filter), this never moves a pixel to another class at the border between two classes.
  const cv::Mat& erode_mask(const cv::Mat& mask);
  Eigen::Vector4f project_point_to_camera(const Eigen::Vector4f& point_lidar);
  bool is_point_in_front_of_camera(const Eigen::Vector4f& point_cam);
  bool is_point_valid(const float& x, const float& y, const float& z);
  bool is_point_in_image(const Eigen::Vector4f& point_cam, int img_width, int img_height);
  Eigen::Vector3f project_point_to_image(const Eigen::Vector4f& point_cam);
  cv::Vec3b sample_color_at_pixel(const cv::Mat& image, int u, int v);
  // Removes statistical outliers from `cloud` in place, object by object (see label_indices_).
  void remove_outliers(pcl::PointCloud<pcl::PointXYZRGB>& cloud);

  Eigen::Matrix4f extrinsics_;
  Eigen::Matrix3f intrinsics_;

  // Built once; reused by every erode_mask() call.
  cv::Mat erode_kernel_;
  // Output of erode_mask(). Reused across frames: cv::erode only reallocates it when the mask
  // size or type changes. Never hand out copies of this header (cv::Mat copies share the buffer).
  cv::Mat eroded_mask_;
  // Scratch buffers of erode_mask() (neighbourhood max, and "neighbourhood not uniform" flags).
  // Reused the same way as eroded_mask_.
  cv::Mat dilated_mask_;
  cv::Mat non_uniform_;

  // Outlier removal state. All buffers are reused across frames (capacity is kept).
  OutlierRemovalParams outlier_params_;
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor_;
  // label_indices_[m] = indices into the output cloud of the points that hit mask value m.
  std::array<pcl::Indices, 256> label_indices_;
  // One object's points, gathered so its neighbour search cannot see other objects.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud_;
  // SOR result: inlier indices into object_cloud_.
  pcl::Indices object_inliers_;
  // keep_flags_[i] != 0 if output point i survives outlier removal.
  std::vector<std::uint8_t> keep_flags_;

  constexpr static float kMinDepthThreshold = 0.0000001f;
};

}  // namespace pcl_labeler