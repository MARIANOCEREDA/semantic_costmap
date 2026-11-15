#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
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

class PCLLabeler
{
 public:
  PCLLabeler();
  void set_extrinsics(const Eigen::Matrix4f& extrinsics);
  void set_intrinsics(const Eigen::Matrix3f& intrinsics);
  // Labels the points of `cloud` that project onto non-zero pixels of `image` and writes them to
  // `output`. The caller owns `output`; it is cleared (capacity kept) and refilled on every call.
  // `cloud` must not point into `output`'s memory.
  void fuse(const cv::Mat& image,
            const PointCloudView& cloud,
            pcl::PointCloud<pcl::PointXYZRGB>& output);

 private:
  // Erodes `mask` into the reused member buffer and returns a reference to it. The reference is
  // only valid until the next call.
  const cv::Mat& erode_mask(const cv::Mat& mask);
  Eigen::Vector4f project_point_to_camera(const Eigen::Vector4f& point_lidar);
  bool is_point_in_front_of_camera(const Eigen::Vector4f& point_cam);
  bool is_point_valid(const float& x, const float& y, const float& z);
  bool is_point_in_image(const Eigen::Vector4f& point_cam, int img_width, int img_height);
  Eigen::Vector3f project_point_to_image(const Eigen::Vector4f& point_cam);
  cv::Vec3b sample_color_at_pixel(const cv::Mat& image, int u, int v);

  Eigen::Matrix4f extrinsics_;
  Eigen::Matrix3f intrinsics_;

  // Built once; reused by every erode_mask() call.
  cv::Mat erode_kernel_;
  // Output of erode_mask(). Reused across frames: cv::erode only reallocates it when the mask
  // size or type changes. Never hand out copies of this header (cv::Mat copies share the buffer).
  cv::Mat eroded_mask_;

  constexpr static float kMinDepthThreshold = 0.0000001f;
};

}  // namespace pcl_labeler