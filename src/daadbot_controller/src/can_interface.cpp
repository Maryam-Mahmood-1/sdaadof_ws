#include "daadbot_controller/can_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace daadbot_controller
{

DaadbotCANInterface::DaadbotCANInterface() {}

DaadbotCANInterface::~DaadbotCANInterface() {}

CallbackReturn DaadbotCANInterface::on_init(const hardware_interface::HardwareInfo &hardware_info)
{
  CallbackReturn result = hardware_interface::SystemInterface::on_init(hardware_info);
  if (result != CallbackReturn::SUCCESS)
  {
    return result;
  }

  // Init joint data storage
  position_commands_.resize(info_.joints.size(), 0.0);
  position_states_.resize(info_.joints.size(), 0.0);

  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> DaadbotCANInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  for (size_t i = 0; i < info_.joints.size(); ++i)
  {
    state_interfaces.emplace_back(
      hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &position_states_[i]));
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> DaadbotCANInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;

  for (size_t i = 0; i < info_.joints.size(); ++i)
  {
    command_interfaces.emplace_back(
      hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &position_commands_[i]));
  }

  return command_interfaces;
}

CallbackReturn DaadbotCANInterface::on_activate(const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(rclcpp::get_logger("DaadbotCANInterface"), "Activating CAN interface (dummy mode)");
  return CallbackReturn::SUCCESS;
}

CallbackReturn DaadbotCANInterface::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(rclcpp::get_logger("DaadbotCANInterface"), "Deactivating CAN interface (dummy mode)");
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type DaadbotCANInterface::read(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // Dummy read: echo back command to state
  position_states_ = position_commands_;
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type DaadbotCANInterface::write(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // Dummy write: print out command
  for (size_t i = 0; i < position_commands_.size(); ++i)
  {
    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotCANInterface"),
                       "Command to joint " << info_.joints[i].name << ": " << position_commands_[i]);
  }

  return hardware_interface::return_type::OK;
}

} // namespace daadbot_controller

PLUGINLIB_EXPORT_CLASS(
  daadbot_controller::DaadbotCANInterface,
  hardware_interface::SystemInterface)
