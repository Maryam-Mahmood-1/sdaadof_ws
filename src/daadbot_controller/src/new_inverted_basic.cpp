#include "daadbot_controller/daadbot_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include "rclcpp/rclcpp.hpp"
#include <iomanip>
#include <chrono>
#include <fstream>
#include <filesystem>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace daadbot_controller
{

DaadbotInterface::DaadbotInterface()
{
}

DaadbotInterface::~DaadbotInterface()
{
}

CallbackReturn DaadbotInterface::on_init(const hardware_interface::HardwareInfo &hardware_info)
{
  CallbackReturn result = hardware_interface::SystemInterface::on_init(hardware_info);
  if (result != CallbackReturn::SUCCESS)
  {
    return result;
  }

  size_t n_joints = info_.joints.size();
  RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"),
                     "Detected Joints: " << n_joints);

  // Primary state and command storage
  position_states_.resize(n_joints, 0.0);
  velocity_states_.resize(n_joints, 0.0);
  effort_states_.resize(n_joints, 0.0);
  position_commands_.resize(n_joints, 0.0);

  // Initial states
  init_position_states_.resize(n_joints, 0.0);

  // Filtered states
  filtered_position_states_.resize(n_joints, 0.0);
  filtered_velocity_states_.resize(n_joints, 0.0);
  filtered_effort_states_.resize(n_joints, 0.0);

  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> DaadbotInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  for (size_t i = 0; i < info_.joints.size(); i++)
  {
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"),
                "Exporting state interface: %s/position", info_.joints[i].name.c_str());
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &position_states_[i]));

    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"),
                "Exporting state interface: %s/velocity", info_.joints[i].name.c_str());
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &velocity_states_[i]));

    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"),
                "Exporting state interface: %s/effort", info_.joints[i].name.c_str());
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &effort_states_[i]));
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> DaadbotInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;

  for (size_t i = 0; i < info_.joints.size(); i++)
  {
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"),
                "Exporting command interface: %s/position", info_.joints[i].name.c_str());
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &position_commands_[i]));
  }

  return command_interfaces;
}

CallbackReturn DaadbotInterface::on_activate(const rclcpp_lifecycle::State &previous_state)
{
  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Starting robot hardware ...");

  size_t n = info_.joints.size();
  position_commands_.assign(n, 0.0);
  position_states_.assign(n, 0.0);
  velocity_states_.assign(n, 0.0);
  effort_states_.assign(n, 0.0);

  init_position_states_.assign(n, 0.0);

  filtered_position_states_.assign(n, 0.0);
  filtered_velocity_states_.assign(n, 0.0);
  filtered_effort_states_.assign(n, 0.0);

  return CallbackReturn::SUCCESS;
}

CallbackReturn DaadbotInterface::on_deactivate(const rclcpp_lifecycle::State &previous_state)
{
  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Stopping robot hardware ...");
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type DaadbotInterface::read(
    const rclcpp::Time &time, const rclcpp::Duration &period)
{
  // Example: sync filtered states with raw states
  filtered_position_states_ = position_states_;
  filtered_velocity_states_ = velocity_states_;
  filtered_effort_states_ = effort_states_;
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type DaadbotInterface::write(
    const rclcpp::Time &time, const rclcpp::Duration &period)
{
  // Example: apply position commands to hardware
  return hardware_interface::return_type::OK;
}

} // namespace daadbot_controller

PLUGINLIB_EXPORT_CLASS(daadbot_controller::DaadbotInterface, hardware_interface::SystemInterface)
