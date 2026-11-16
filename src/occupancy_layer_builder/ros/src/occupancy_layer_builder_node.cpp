#include <Eigen/Geometry>
#include <lifecycle_msgs/msg/state.hpp>
#include <occupancy_layer_builder/occupancy_layer_builder_node.hpp>
#include <optional>
#include <sensor_msgs/msg/point_field.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

namespace occupancy_layer_builder
{
namespace
{
const sensor_msgs::msg::PointField* FindField(const sensor_msgs::msg::PointCloud2& msg,
                                              const std::string& name)
{
  for (const auto& field : msg.fields)
  {
    if (field.name == name)
    {
      return &field;
    }
  }
  return nullptr;
}

// Builds a non-owning view over msg.data, or returns nullopt (with `reason`) if the layout is not
// what the builder can read. The view is only valid while `msg` is alive.
std::optional<PointCloudView> MakePointCloudView(const sensor_msgs::msg::PointCloud2& msg,
                                                 std::string& reason)
{
  using sensor_msgs::msg::PointField;
  const PointField* x = FindField(msg, "x");
  const PointField* y = FindField(msg, "y");
  const PointField* z = FindField(msg, "z");
  const PointField* ground = FindField(msg, "ground");
  if (x == nullptr || y == nullptr || z == nullptr || ground == nullptr)
  {
    reason = "cloud needs fields x, y, z and ground (from ground_segmentation)";
    return std::nullopt;
  }
  if (x->datatype != PointField::FLOAT32 || y->datatype != PointField::FLOAT32 ||
      z->datatype != PointField::FLOAT32 || ground->datatype != PointField::UINT8)
  {
    reason = "x/y/z must be FLOAT32 and ground UINT8";
    return std::nullopt;
  }
  if (msg.is_bigendian)
  {
    reason = "big-endian clouds are not supported";
    return std::nullopt;
  }
  if (x->offset + sizeof(float) > msg.point_step || y->offset + sizeof(float) > msg.point_step ||
      z->offset + sizeof(float) > msg.point_step || ground->offset + 1 > msg.point_step)
  {
    reason = "a field lies outside point_step";
    return std::nullopt;
  }
  if (static_cast<std::size_t>(msg.row_step) <
          static_cast<std::size_t>(msg.width) * msg.point_step ||
      msg.data.size() < static_cast<std::size_t>(msg.row_step) * msg.height)
  {
    reason = "data is smaller than row_step * height";
    return std::nullopt;
  }

  PointCloudView view;
  view.data = msg.data.data();
  view.width = msg.width;
  view.height = msg.height;
  view.point_step = msg.point_step;
  view.row_step = msg.row_step;
  view.x_offset = x->offset;
  view.y_offset = y->offset;
  view.z_offset = z->offset;
  view.ground_offset = ground->offset;
  return view;
}
}  // namespace

OccupancyLayerBuilderNode::OccupancyLayerBuilderNode(const rclcpp::NodeOptions& options)
  : rclcpp_lifecycle::LifecycleNode("occupancy_layer_builder_node", options)
{
  RCLCPP_INFO(this->get_logger(), "OccupancyLayerBuilderNode created");

  // Declare and get parameters
  // Full cloud from ground_segmentation (all points + `ground` field), not the labeled cloud:
  // occupancy comes from LiDAR geometry, labels only say what an occupied cell is.
  this->declare_parameter<std::string>("pointcloud_topic", "/segmented_points");
  this->declare_parameter<std::string>("grid_topic", "/map");
  this->declare_parameter<std::string>("map_frame", "map");
  this->declare_parameter<std::string>("sensor_frame", "lidar_frame");
  this->declare_parameter<double>("resolution", 0.1);
  this->declare_parameter<int>("width", 1000);
  this->declare_parameter<int>("height", 1000);
  this->declare_parameter<double>("origin_x", -50.0);
  this->declare_parameter<double>("origin_y", -50.0);
  this->declare_parameter<bool>("display_latency", false);
  // Log-odds accumulation (see occupancy_layer_builder::LogOddsParams)
  const LogOddsParams log_odds_defaults;
  this->declare_parameter<double>("hit_probability", log_odds_defaults.hit_probability);
  this->declare_parameter<double>("miss_probability", log_odds_defaults.miss_probability);
  this->declare_parameter<double>("min_probability", log_odds_defaults.min_probability);
  this->declare_parameter<double>("max_probability", log_odds_defaults.max_probability);
  this->declare_parameter<double>("occupied_threshold", log_odds_defaults.occupied_threshold);
  this->declare_parameter<double>("free_threshold", log_odds_defaults.free_threshold);
  // Scan filtering (see occupancy_layer_builder::ScanFilterParams)
  const ScanFilterParams scan_filter_defaults;
  this->declare_parameter<double>("max_range", scan_filter_defaults.max_range);
  this->declare_parameter<double>("max_obstacle_height", scan_filter_defaults.max_obstacle_height);
}

CallbackReturn OccupancyLayerBuilderNode::on_configure(const rclcpp_lifecycle::State&)
{
  RCLCPP_INFO(this->get_logger(), "Configuring OccupancyLayerBuilderNode");

  this->get_parameter("pointcloud_topic", pointcloud_topic_);
  this->get_parameter("grid_topic", grid_topic_);
  this->get_parameter("map_frame", map_frame_);
  this->get_parameter("sensor_frame", sensor_frame_);
  this->get_parameter("resolution", resolution_);
  this->get_parameter("width", width_);
  this->get_parameter("height", height_);
  this->get_parameter("origin_x", origin_x_);
  this->get_parameter("origin_y", origin_y_);
  this->get_parameter("display_latency", display_latency_);

  return CallbackReturn::SUCCESS;
}

CallbackReturn OccupancyLayerBuilderNode::on_activate(const rclcpp_lifecycle::State&)
{
  RCLCPP_INFO(this->get_logger(), "Activating OccupancyLayerBuilderNode");
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  layer_builder_ =
      std::make_unique<OccupancyLayerBuilder>(resolution_, width_, height_, origin_x_, origin_y_);
  try
  {
    LogOddsParams log_odds;
    log_odds.hit_probability = this->get_parameter("hit_probability").as_double();
    log_odds.miss_probability = this->get_parameter("miss_probability").as_double();
    log_odds.min_probability = this->get_parameter("min_probability").as_double();
    log_odds.max_probability = this->get_parameter("max_probability").as_double();
    log_odds.occupied_threshold = this->get_parameter("occupied_threshold").as_double();
    log_odds.free_threshold = this->get_parameter("free_threshold").as_double();
    layer_builder_->setLogOddsParams(log_odds);

    ScanFilterParams scan_filter;
    scan_filter.max_range = this->get_parameter("max_range").as_double();
    scan_filter.max_obstacle_height = this->get_parameter("max_obstacle_height").as_double();
    layer_builder_->setScanFilterParams(scan_filter);
  }
  catch (const std::invalid_argument& ex)
  {
    RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
    return CallbackReturn::FAILURE;
  }

  // Use TRANSIENT_LOCAL durability for map topic to be compatible with nav2 costmap
  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
  grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(grid_topic_, qos);

  pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      pointcloud_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&OccupancyLayerBuilderNode::pointCloudCallback, this, std::placeholders::_1));

  processing_thread_ = std::jthread(&OccupancyLayerBuilderNode::layerBuilderThread, this);
  grid_pub_->on_activate();
  RCLCPP_INFO(this->get_logger(), "Activated OccupancyLayerBuilderNode");
  return CallbackReturn::SUCCESS;
}

CallbackReturn OccupancyLayerBuilderNode::on_deactivate(const rclcpp_lifecycle::State&)
{
  RCLCPP_INFO(this->get_logger(), "Deactivating OccupancyLayerBuilderNode");
  pcl_cv_.notify_all();
  processing_thread_.request_stop();
  if (processing_thread_.joinable())
  {
    processing_thread_.join();
  }
  grid_pub_->on_deactivate();
  RCLCPP_INFO(this->get_logger(), "Deactivated OccupancyLayerBuilderNode");
  return CallbackReturn::SUCCESS;
}

CallbackReturn OccupancyLayerBuilderNode::on_cleanup(const rclcpp_lifecycle::State&)
{
  RCLCPP_INFO(this->get_logger(), "Cleaning up OccupancyLayerBuilderNode");
  pointcloud_sub_.reset();
  grid_pub_.reset();
  layer_builder_.reset();
  tf_listener_.reset();
  tf_buffer_.reset();
  pcl_cv_.notify_all();
  processing_thread_.request_stop();
  if (processing_thread_.joinable())
  {
    processing_thread_.join();
  }
  RCLCPP_INFO(this->get_logger(), "Cleaned up OccupancyLayerBuilderNode");
  return CallbackReturn::SUCCESS;
}

CallbackReturn OccupancyLayerBuilderNode::on_shutdown(const rclcpp_lifecycle::State&)
{
  RCLCPP_INFO(this->get_logger(), "Shutting down OccupancyLayerBuilderNode");
  pointcloud_sub_.reset();
  grid_pub_.reset();
  layer_builder_.reset();
  tf_listener_.reset();
  tf_buffer_.reset();
  RCLCPP_INFO(this->get_logger(), "Shut down OccupancyLayerBuilderNode");
  return CallbackReturn::SUCCESS;
}

void OccupancyLayerBuilderNode::pointCloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  {
    std::lock_guard<std::mutex> lock(pcl_mutex_);
    latest_pointcloud_msg_ = std::move(msg);
    new_pointcloud_available_ = true;
  }
  pcl_cv_.notify_one();
}

nav_msgs::msg::OccupancyGrid OccupancyLayerBuilderNode::convertToOccupancyGrid(
    const std::vector<std::int8_t>& grid, const std_msgs::msg::Header& header)
{
  nav_msgs::msg::OccupancyGrid grid_msg;
  grid_msg.header.stamp = header.stamp;
  grid_msg.header.frame_id = map_frame_;
  grid_msg.info.resolution = resolution_;
  grid_msg.info.width = width_;
  grid_msg.info.height = height_;
  grid_msg.info.origin.position.x = origin_x_;
  grid_msg.info.origin.position.y = origin_y_;
  grid_msg.info.origin.position.z = 0.0;
  grid_msg.info.origin.orientation.w = 1.0;

  // The builder already stores cells row-major in OccupancyGrid encoding
  // (-1 unknown, 0 free, 100 occupied, 1..99 uncertain), so this is a single memcpy.
  grid_msg.data.assign(grid.begin(), grid.end());

  return grid_msg;
}

void OccupancyLayerBuilderNode::layerBuilderThread(std::stop_token stop_token)
{
  while (rclcpp::ok() && !stop_token.stop_requested())
  {
    sensor_msgs::msg::PointCloud2::SharedPtr msg{nullptr};
    {
      std::unique_lock<std::mutex> lock(pcl_mutex_);
      pcl_cv_.wait(lock,
                   [this, &stop_token]
                   {
                     return new_pointcloud_available_.load() || !rclcpp::ok() ||
                            stop_token.stop_requested();
                   });

      if (stop_token.stop_requested())
      {
        break;
      }

      if (!new_pointcloud_available_ || !latest_pointcloud_msg_)
      {
        continue;
      }
      msg = std::move(latest_pointcloud_msg_);
      new_pointcloud_available_ = false;
    }

    geometry_msgs::msg::TransformStamped transform_stamped;
    try
    {
      transform_stamped = tf_buffer_->lookupTransform(
          map_frame_, msg->header.frame_id, msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
    }
    catch (const tf2::TransformException& ex)
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(),
                           *this->get_clock(),
                           1000,
                           "Could not transform %s to %s: %s",
                           msg->header.frame_id.c_str(),
                           map_frame_.c_str(),
                           ex.what());
      continue;
    }
    Eigen::Isometry3d transform_eigen = tf2::transformToEigen(transform_stamped.transform);
    transform_matrix_ = transform_eigen.matrix().cast<float>();

    // Read the points in place: `msg` (held by this scope) owns the bytes the view points into.
    std::string reason;
    const std::optional<PointCloudView> cloud_view = MakePointCloudView(*msg, reason);
    if (!cloud_view)
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(),
                           *this->get_clock(),
                           2000,
                           "Skipping point cloud on %s: %s",
                           pointcloud_topic_.c_str(),
                           reason.c_str());
      continue;
    }

    const auto& grid = layer_builder_->integrateScan(*cloud_view, transform_matrix_);

    auto grid_msg = convertToOccupancyGrid(grid, msg->header);
    auto end_time = std::chrono::high_resolution_clock::now();

    grid_pub_->publish(grid_msg);

    if (display_latency_)
    {
      const double cloud_latency =
          (this->now() - rclcpp::Time(msg->header.stamp, RCL_ROS_TIME)).seconds();
      RCLCPP_INFO_THROTTLE(
          this->get_logger(), *this->get_clock(), 1000, "Pipeline latency: %.3f s", cloud_latency);
    }
  }
}

}  // namespace occupancy_layer_builder

// Register the component with class_loader for composable nodes
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(occupancy_layer_builder::OccupancyLayerBuilderNode)
