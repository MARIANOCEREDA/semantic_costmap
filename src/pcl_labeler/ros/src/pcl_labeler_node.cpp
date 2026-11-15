#include "pcl_labeler/pcl_labeler_node.hpp"

#include <chrono>
#include <optional>
#include <cv_bridge/cv_bridge.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/image_encodings.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include "pcl_labeler/pcl_labeler.hpp"

namespace pcl_labeler
{
namespace
{
// Builds a non-owning view over the message bytes. Returns nullopt (and sets `reason`) if the
// layout cannot be read safely as little-endian float32 x/y/z.
std::optional<PointCloudView> MakePointCloudView(const sensor_msgs::msg::PointCloud2& msg,
                                                 std::string& reason)
{
  std::optional<std::size_t> offsets[3];
  constexpr const char* kNames[3] = {"x", "y", "z"};
  for (const auto& field : msg.fields)
  {
    for (int i = 0; i < 3; ++i)
    {
      if (field.name == kNames[i] && field.datatype == sensor_msgs::msg::PointField::FLOAT32 &&
          field.count >= 1)
      {
        offsets[i] = field.offset;
      }
    }
  }
  if (!offsets[0] || !offsets[1] || !offsets[2])
  {
    reason = "Point cloud needs float32 x, y, z fields.";
    return std::nullopt;
  }
  if (msg.is_bigendian)
  {
    reason = "Big-endian point clouds are not supported.";
    return std::nullopt;
  }
  for (const auto& offset : offsets)
  {
    if (*offset + sizeof(float) > msg.point_step)
    {
      reason = "Field offset outside point_step.";
      return std::nullopt;
    }
  }
  const std::size_t width = msg.width;
  const std::size_t height = msg.height;
  if (static_cast<std::size_t>(msg.row_step) < width * msg.point_step ||
      msg.data.size() < static_cast<std::size_t>(msg.row_step) * height)
  {
    reason = "Point cloud data is smaller than width/height/row_step imply.";
    return std::nullopt;
  }

  PointCloudView view;
  view.data = msg.data.data();
  view.width = width;
  view.height = height;
  view.point_step = msg.point_step;
  view.row_step = msg.row_step;
  view.x_offset = *offsets[0];
  view.y_offset = *offsets[1];
  view.z_offset = *offsets[2];
  return view;
}
}  // namespace

PCLLabelerNode::PCLLabelerNode(const rclcpp::NodeOptions& options)
  : rclcpp_lifecycle::LifecycleNode("pcl_labeler_node", options)
{
  RCLCPP_INFO(get_logger(), "Camera-LiDAR Fusion Node has been created.");

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);
  labeled_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

  this->declare_parameter<std::string>("camera_frame_name", "camera_frame");
  this->declare_parameter<std::string>("lidar_frame_name", "lidar_frame");
  this->declare_parameter<std::string>("point_cloud_topic", "rslidar_points");
  this->declare_parameter<std::string>("labeled_point_cloud_topic", "points_labeled");
  this->declare_parameter<std::string>("segmentation_mask_topic", "/segmentation/mask");
  this->declare_parameter<std::string>("camera_info_topic", "camera_info");
  this->declare_parameter<int>("sync_queue_size", 10);
  this->declare_parameter<double>("sync_max_interval", 0.05);
  this->declare_parameter<bool>("display_latency", false);
}

CallbackReturn PCLLabelerNode::on_configure(const rclcpp_lifecycle::State&)
{
  RCLCPP_INFO(get_logger(), "Configuring Camera-LiDAR Fusion Node...");

  camera_frame_name_ = this->get_parameter("camera_frame_name").as_string();
  lidar_frame_name_ = this->get_parameter("lidar_frame_name").as_string();
  display_latency_ = this->get_parameter("display_latency").as_bool();

  std::string point_cloud_topic = this->get_parameter("point_cloud_topic").as_string();
  std::string segmentation_mask_topic = this->get_parameter("segmentation_mask_topic").as_string();
  std::string camera_info_topic = this->get_parameter("camera_info_topic").as_string();
  std::string labeled_point_cloud_topic =
      this->get_parameter("labeled_point_cloud_topic").as_string();

  // Create publishers
  labeled_pcl_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      labeled_point_cloud_topic, rclcpp::QoS(rclcpp::KeepLast(1)));

  rclcpp::SubscriptionOptions camera_sub_options;
  camera_sub_options.callback_group =
      this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // Create subscriptions
  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic,
      10,
      std::bind(&PCLLabelerNode::camera_info_callback, this, std::placeholders::_1),
      camera_sub_options);

  // The mask arrives after YOLO inference, so the queue must hold clouds until it does.
  const auto sync_queue_size = static_cast<size_t>(this->get_parameter("sync_queue_size").as_int());
  const double sync_max_interval = this->get_parameter("sync_max_interval").as_double();
  const rmw_qos_profile_t sync_qos =
      rclcpp::SensorDataQoS().keep_last(sync_queue_size).get_rmw_qos_profile();

  // Same callback group as camera_info so labeling is never run concurrently.
  lidar_sub_ =
      std::make_shared<CloudSubscriber>(this, point_cloud_topic, sync_qos, camera_sub_options);
  segmentation_mask_sub_ =
      std::make_shared<MaskSubscriber>(this, segmentation_mask_topic, sync_qos, camera_sub_options);

  sync_ = std::make_shared<Synchronizer>(
      SyncPolicy(sync_queue_size), *lidar_sub_, *segmentation_mask_sub_);
  sync_->getPolicy()->setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_max_interval));
  sync_->registerCallback(std::bind(
      &PCLLabelerNode::label_point_cloud, this, std::placeholders::_1, std::placeholders::_2));

  RCLCPP_INFO(get_logger(), "Camera-LiDAR Fusion Node configured.");
  return CallbackReturn::SUCCESS;
}

CallbackReturn PCLLabelerNode::on_activate(const rclcpp_lifecycle::State&)
{
  RCLCPP_INFO(get_logger(), "Activating Camera-LiDAR Fusion Node...");
  labeled_pcl_pub_->on_activate();
  RCLCPP_INFO(get_logger(), "Camera-LiDAR Fusion Node activated.");
  return CallbackReturn::SUCCESS;
}

CallbackReturn PCLLabelerNode::on_deactivate(const rclcpp_lifecycle::State&)
{
  RCLCPP_INFO(get_logger(), "Deactivating Camera-LiDAR Fusion Node...");
  labeled_pcl_pub_->on_deactivate();
  RCLCPP_INFO(get_logger(), "Camera-LiDAR Fusion Node deactivated.");
  return CallbackReturn::SUCCESS;
}

CallbackReturn PCLLabelerNode::on_cleanup(const rclcpp_lifecycle::State&)
{
  RCLCPP_INFO(get_logger(), "Cleaning up Camera-LiDAR Fusion Node...");
  labeled_pcl_pub_.reset();
  camera_info_sub_.reset();
  camera_info_received_ = false;
  sync_.reset();
  segmentation_mask_sub_.reset();
  lidar_sub_.reset();
  RCLCPP_INFO(get_logger(), "Camera-LiDAR Fusion Node cleaned up.");
  return CallbackReturn::SUCCESS;
}

CallbackReturn PCLLabelerNode::on_shutdown(const rclcpp_lifecycle::State&)
{
  RCLCPP_INFO(get_logger(), "Shutting down Camera-LiDAR Fusion Node...");
  return CallbackReturn::SUCCESS;
}

void PCLLabelerNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  camera_intrinsics_ = cv::Mat(3, 3, CV_64F);
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      camera_intrinsics_.at<double>(i, j) = msg->k[i * 3 + j];
    }
  }
  camera_info_received_ = true;
  camera_info_sub_.reset();
  RCLCPP_INFO(get_logger(), "Received camera info.");
}

void PCLLabelerNode::label_point_cloud(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& input_cloud,
    const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_mask)
{
  // Stage timing uses a monotonic clock; message age uses the ROS clock (sim time aware).
  using StageClock = std::chrono::steady_clock;
  const auto elapsed_ms = [](StageClock::time_point from, StageClock::time_point to)
  { return std::chrono::duration<double, std::milli>(to - from).count(); };
  
  const auto t_start = StageClock::now();

  cv::Mat camera_intrinsics;
  if (!camera_info_received_)
  {
    RCLCPP_INFO(get_logger(), "Camera info not received yet. Skipping labeling.");
    return;
  }
  camera_intrinsics = camera_intrinsics_.clone();

  try
  {
    // 1. Get camera-to-lidar transform
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped =
        tf_buffer_->lookupTransform(camera_frame_name_, lidar_frame_name_, tf2::TimePointZero);

    Eigen::Isometry3d transform_eigen = tf2::transformToEigen(transform_stamped.transform);
    Eigen::Matrix4f extrinsics = transform_eigen.matrix().cast<float>();

    // 2. Convert segmentation mask to OpenCV
    const size_t required_size = static_cast<size_t>(segmentation_mask->step) *
                                     (static_cast<size_t>(segmentation_mask->height) - 1) +
                                 static_cast<size_t>(segmentation_mask->width);
    const size_t actual_size = segmentation_mask->data.size();

    bool valid_img_msg = true;
    std::string reason = "";

    if (segmentation_mask->step < segmentation_mask->width)
    {
      valid_img_msg = false;
      reason = "Invalid segmentation mask step.";
    }

    if (actual_size < required_size)
    {
      valid_img_msg = false;
      reason = "Segmentation mask data size is smaller than expected.";
    }

    if (segmentation_mask->encoding != sensor_msgs::image_encodings::MONO8)
    {
      valid_img_msg = false;
      reason = "Unsupported segmentation mask encoding. Expected MONO8.";
    }

    if (!valid_img_msg)
    {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1.0, "Invalid segmentation mask message: %s", reason.c_str());
      return;
    }

    // cv::Size is (width, height); the cv::Mat(rows, cols, ...) overload would transpose the view
    // and make erode()/at<>() read past the end of the message buffer.
    cv::Mat mask_single_channel(cv::Size(static_cast<int>(segmentation_mask->width),
                                         static_cast<int>(segmentation_mask->height)),
                                CV_8UC1,
                                const_cast<uint8_t*>(segmentation_mask->data.data()),
                                static_cast<size_t>(segmentation_mask->step));

    // 3. View the PointCloud2 bytes in place. `input_cloud` (ConstSharedPtr) keeps the buffer
    //    alive for this whole callback, and fuse() is synchronous, so the view cannot dangle.
    std::string cloud_reason;
    const std::optional<PointCloudView> cloud_view = MakePointCloudView(*input_cloud, cloud_reason);
    if (!cloud_view)
    {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000, "Invalid point cloud message: %s", cloud_reason.c_str());
      return;
    }

    // 4. Update the persistent labeler's calibration and call fuse()

    // Convert camera intrinsics to Eigen
    Eigen::Matrix3f intrinsics_eigen;
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        intrinsics_eigen(i, j) = static_cast<float>(camera_intrinsics.at<double>(i, j));
      }
    }

    labeler_.set_extrinsics(extrinsics);
    labeler_.set_intrinsics(intrinsics_eigen);

    // Fuse mask with point cloud
    labeler_.fuse(mask_single_channel, *cloud_view, *labeled_cloud_);

    // 5. Convert back to ROS PointCloud2
    auto output_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(*labeled_cloud_, *output_msg);
    output_msg->header = input_cloud->header;

    // 6. Publish labeled point cloud
    labeled_pcl_pub_->publish(std::move(output_msg));
    const auto t_publish = StageClock::now();

    if (display_latency_)
    {
      RCLCPP_INFO_THROTTLE(get_logger(),
                           *get_clock(),
                           1000,
                           "Latency [ms] total=%.3f | points in=%zu out=%zu",
                           elapsed_ms(t_start, t_publish),
                           cloud_view->size(),
                           labeled_cloud_->size());
    }
  }
  catch (const tf2::TransformException& ex)
  {
    RCLCPP_WARN(get_logger(), "Could not get transform: %s", ex.what());
  }
  catch (const std::exception& ex)
  {
    RCLCPP_ERROR(get_logger(), "Error in label_point_cloud: %s", ex.what());
  }
}
}  // namespace pcl_labeler

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(pcl_labeler::PCLLabelerNode)