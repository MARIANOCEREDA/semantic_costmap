#include "pcl_labeler/pcl_labeler.hpp"

#include <cstring>

namespace pcl_labeler
{

PCLLabeler::PCLLabeler()
{
  extrinsics_ = Eigen::Matrix4f::Identity();
  intrinsics_ = Eigen::Matrix3f::Identity();
  erode_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
}

void PCLLabeler::set_extrinsics(const Eigen::Matrix4f& extrinsics)
{
  extrinsics_ = extrinsics;
}

void PCLLabeler::set_intrinsics(const Eigen::Matrix3f& intrinsics)
{
  intrinsics_ = intrinsics;
}

Eigen::Vector4f PCLLabeler::project_point_to_camera(const Eigen::Vector4f& point_lidar)
{
  return extrinsics_ * point_lidar;
}

bool PCLLabeler::is_point_in_front_of_camera(const Eigen::Vector4f& point_cam)
{
  return point_cam.z() > kMinDepthThreshold;
}

Eigen::Vector3f PCLLabeler::project_point_to_image(const Eigen::Vector4f& point_cam)
{
  const double inv_z = 1.0 / static_cast<double>(point_cam.z());
  const double u =
      intrinsics_(0, 0) * static_cast<double>(point_cam.x()) * inv_z + intrinsics_(0, 2);
  const double v =
      intrinsics_(1, 1) * static_cast<double>(point_cam.y()) * inv_z + intrinsics_(1, 2);
  return Eigen::Vector3f(static_cast<float>(u), static_cast<float>(v), point_cam.z());
}

bool PCLLabeler::is_point_in_image(const Eigen::Vector4f& point_cam,
                                   int img_width,
                                   int img_height)
{
  if (!is_point_in_front_of_camera(point_cam))
  {
    return false;
  }
  const Eigen::Vector3f uv = project_point_to_image(point_cam);
  return uv.x() >= 0.0f && uv.y() >= 0.0f && uv.x() < static_cast<float>(img_width) &&
         uv.y() < static_cast<float>(img_height);
}

bool PCLLabeler::is_point_valid(const float& x, const float& y, const float& z)
{
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

const cv::Mat& PCLLabeler::erode_mask(const cv::Mat& mask)
{
  cv::erode(mask, eroded_mask_, erode_kernel_, cv::Point(-1, -1), 1);
  return eroded_mask_;
}

void PCLLabeler::fuse(const cv::Mat& image,
                      const PointCloudView& cloud,
                      pcl::PointCloud<pcl::PointXYZRGB>& output)
{
  // 1. Clear the output cloud (keeps its capacity)
  output.clear();

  // 2. Erode the mask to avoid edge effects
  const cv::Mat& image_eroded = erode_mask(image);

  // 3. Loop through each point of the input buffer, reading x/y/z in place (no copy of the cloud)
  for (std::size_t row = 0; row < cloud.height; ++row)
  {
    const std::uint8_t* row_ptr = cloud.data + row * cloud.row_step;
    for (std::size_t col = 0; col < cloud.width; ++col)
    {
      const std::uint8_t* point_ptr = row_ptr + col * cloud.point_step;

      // memcpy instead of reinterpret_cast<const float*>: the bytes may be unaligned and are not
      // float objects, so a cast would be undefined behavior. The compiler emits a plain load.
      float x = 0.0f;
      float y = 0.0f;
      float z = 0.0f;
      std::memcpy(&x, point_ptr + cloud.x_offset, sizeof(float));
      std::memcpy(&y, point_ptr + cloud.y_offset, sizeof(float));
      std::memcpy(&z, point_ptr + cloud.z_offset, sizeof(float));

      if (!is_point_valid(x, y, z))
      {
        continue;
      }

      // 4. Project the point to camera coordinates
      const Eigen::Vector4f point_lidar(x, y, z, 1.0f);
      const Eigen::Vector4f point_cam = project_point_to_camera(point_lidar);

      // 5. Check if the point is in front of the camera and within the image bounds
      if (!is_point_in_image(point_cam, image_eroded.cols, image_eroded.rows))
      {
        continue;
      }

      // 6. Sample the color from the image
      const Eigen::Vector3f uv = project_point_to_image(point_cam);
      const int u = static_cast<int>(uv.x());
      const int v = static_cast<int>(uv.y());

      // Check if that pixel is part of the mask of the background
      if (image_eroded.at<uint8_t>(v, u) == 0)
      {
        continue;
      }

      const cv::Vec3b color = sample_color_at_pixel(image_eroded, u, v);

      // 7. Create a colored point and add it to the output cloud
      pcl::PointXYZRGB colored_point;
      colored_point.x = x;
      colored_point.y = y;
      colored_point.z = z;
      colored_point.b = color[0];  // OpenCV stores channels as BGR
      colored_point.g = color[1];
      colored_point.r = color[2];
      output.push_back(colored_point);
    }
  }

  // 8. Clustering and outlier removal based on statistical analysis
}

cv::Vec3b PCLLabeler::sample_color_at_pixel(const cv::Mat& image, int u, int v)
{
  const uint8_t mask_value = image.at<uint8_t>(v, u);
  return cv::Vec3b(mask_value, mask_value, mask_value);
}

}  // namespace pcl_labeler