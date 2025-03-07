cmake_minimum_required(VERSION 3.8)
project(some_examples_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(daadbot_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rcl_interfaces REQUIRED)

add_executable(simple_publisher src/simple_publisher.cpp)
ament_target_dependencies(simple_publisher rclcpp std_msgs)



add_executable(my_moveit_interface src/my_moveit_interface.cpp)
ament_target_dependencies(my_moveit_interface rclcpp moveit_ros_planning_interface)

add_executable(lifecycle_node_ex src/lifecycle_node_ex.cpp)
ament_target_dependencies(lifecycle_node_ex rclcpp rclcpp_lifecycle std_msgs)



install(TARGETS
  simple_publisher
  my_moveit_interface
  lifecycle_node_ex

  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)

  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()