cmake_minimum_required(VERSION 3.8)
project(daadbot_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# === Required Packages ===
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SERIAL libserial)

# === Serial Hardware Interface (ESP) ===
add_library(daadbot_serial_interface SHARED 
  src/daadbot_interface.cpp
)
target_include_directories(daadbot_serial_interface PRIVATE 
  include
  "${SERIAL_INCLUDE_DIRS}"
)
target_link_libraries(daadbot_serial_interface
  "${SERIAL_LDFLAGS}"
)
ament_target_dependencies(daadbot_serial_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

# === CAN Hardware Interface (SocketCAN) ===
add_library(daadbot_can_interface SHARED 
  src/can_interface.cpp
)
target_include_directories(daadbot_can_interface PRIVATE 
  include
)
ament_target_dependencies(daadbot_can_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

# === Export both interfaces ===
pluginlib_export_plugin_description_file(hardware_interface daadbot_controller.xml)

# === Install headers, configs, plugins ===
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY include DESTINATION include)

install(TARGETS
  daadbot_serial_interface
  daadbot_can_interface
  DESTINATION lib
)

ament_export_include_directories(include)
ament_export_libraries(
  daadbot_serial_interface
  daadbot_can_interface
)
ament_export_dependencies(
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

# === Optional SocketCAN utility node ===
add_executable(motor_can_node src/motor_can_node.cpp)
ament_target_dependencies(motor_can_node rclcpp)
target_include_directories(motor_can_node PRIVATE include)
install(TARGETS motor_can_node DESTINATION lib/${PROJECT_NAME})

# === Tests ===
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()