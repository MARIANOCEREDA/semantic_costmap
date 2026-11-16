cmake_minimum_required(VERSION 3.8)
project(semantic_layer_builder)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 20)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# Find the semantic_layer_builder_cpp C++ library
if(DEFINED SEMANTIC_LAYER_BUILDER_CPP)
  # Using the library built in the cpp/ folder
  set(semantic_layer_builder_cpp_DIR "${SEMANTIC_LAYER_BUILDER_CPP}/install/lib/cmake/semantic_layer_builder_cpp")
  find_package(semantic_layer_builder_cpp REQUIRED PATHS ${semantic_layer_builder_cpp_DIR} NO_DEFAULT_PATH)
else()
  find_package(semantic_layer_builder_cpp REQUIRED)
endif()

# TODO(next step): add the node component (semantic_layer_builder_component), link it to
# semantic_layer_builder_cpp::semantic_layer_builder_cpp, register it with
# rclcpp_components_register_node, and install/export it as in occupancy_layer_builder.

ament_package()
