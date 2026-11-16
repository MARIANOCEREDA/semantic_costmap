#include "pcl_labeler/pcl_labeler.hpp"

#include <cstring>
#include <stdexcept>

namespace pcl_labeler
{

PCLLabeler::PCLLabeler()
{
  extrinsics_ = Eigen::Matrix4f::Identity();
  intrinsics_ = Eigen::Matrix3f::Identity();
  erode_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  object_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  set_outlier_removal(OutlierRemovalParams{});
}

void PCLLabeler::set_outlier_removal(const OutlierRemovalParams& params)
{
  if (params.mean_k < 1 || params.stddev_mul < 0.0)
  {
    throw std::invalid_argument("PCLLabeler: outlier removal needs mean_k >= 1 and stddev_mul >= 0");
  }
  outlier_params_ = params;
  sor_.setMeanK(params.mean_k);
  sor_.setStddevMulThresh(params.stddev_mul);
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
  // min == max over the neighbourhood <=> the neighbourhood is uniform. For a 0/255 mask this is
  // identical to a plain erode.
  cv::erode(mask, eroded_mask_, erode_kernel_, cv::Point(-1, -1), 1);
  cv::dilate(mask, dilated_mask_, erode_kernel_, cv::Point(-1, -1), 1);
  cv::compare(eroded_mask_, dilated_mask_, non_uniform_, cv::CMP_NE);
  eroded_mask_.setTo(0, non_uniform_);
  return eroded_mask_;
}

void PCLLabeler::fuse(const cv::Mat& image,
                      const PointCloudView& cloud,
                      pcl::PointCloud<pcl::PointXYZRGB>& output)
{
  // 1. Clear the output cloud and the per-object index lists (both keep their capacity)
  output.clear();
  for (auto& indices : label_indices_)
  {
    indices.clear();
  }

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

      // Remember which object (mask value) this output point belongs to
      label_indices_[image_eroded.at<uint8_t>(v, u)].push_back(
          static_cast<pcl::index_t>(output.size()));

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

  // 8. Statistical outlier removal, per object
  remove_outliers(output);
}

void PCLLabeler::remove_outliers(pcl::PointCloud<pcl::PointXYZRGB>& cloud)
{
  if (!outlier_params_.enabled || cloud.empty())
  {
    return;
  }

  keep_flags_.assign(cloud.size(), 0);

  for (const pcl::Indices& object_indices : label_indices_)
  {
    if (object_indices.empty())
    {
      continue;
    }

    // Each point needs mean_k neighbours besides itself; with fewer points there are no statistics
    // to estimate, so the object is kept as is.
    if (object_indices.size() <= static_cast<std::size_t>(outlier_params_.mean_k))
    {
      for (const pcl::index_t i : object_indices)
      {
        keep_flags_[static_cast<std::size_t>(i)] = 1;
      }
      continue;
    }

    // Gather the object into its own cloud. StatisticalOutlierRemoval builds its kd-tree over the
    // whole input cloud even when setIndices() is used, so neighbours would leak across objects.
    object_cloud_->clear();
    for (const pcl::index_t i : object_indices)
    {
      object_cloud_->push_back(cloud[static_cast<std::size_t>(i)]);
    }

    sor_.setInputCloud(object_cloud_);
    sor_.filter(object_inliers_);

    // Map inliers (indices into object_cloud_) back to indices into `cloud`
    for (const pcl::index_t j : object_inliers_)
    {
      keep_flags_[static_cast<std::size_t>(object_indices[static_cast<std::size_t>(j)])] = 1;
    }
  }

  // In-place compaction: the write position never passes the read position, so no point is
  // overwritten before it is read, and no second cloud is needed.
  std::size_t write = 0;
  for (std::size_t read = 0; read < cloud.size(); ++read)
  {
    if (keep_flags_[read] != 0)
    {
      cloud.points[write++] = cloud.points[read];
    }
  }
  cloud.points.resize(write);
  cloud.width = static_cast<std::uint32_t>(write);
  cloud.height = 1;
}

cv::Vec3b PCLLabeler::sample_color_at_pixel(const cv::Mat& image, int u, int v)
{
  const uint8_t mask_value = image.at<uint8_t>(v, u);
  return cv::Vec3b(mask_value, mask_value, mask_value);
}

}  // namespace pcl_labeler