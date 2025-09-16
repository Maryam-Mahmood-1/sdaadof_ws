#ifndef DAADBOT_INTERFACE_H
#define DAADBOT_INTERFACE_H

#include <rclcpp/rclcpp.hpp>
#include <hardware_interface/system_interface.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp>
#include <vector>
#include <string>

namespace daadbot_controller
{

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class DaadbotInterface : public hardware_interface::SystemInterface
{
public:
  DaadbotInterface();
  virtual ~DaadbotInterface();

  // Lifecycle interface
  CallbackReturn on_activate(const rclcpp_lifecycle::State &previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

  // SystemInterface overrides
  CallbackReturn on_init(const hardware_interface::HardwareInfo &hardware_info) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:

  int socket_fd_;

  // Raw states
  std::vector<double> position_states_;
  std::vector<double> velocity_states_;
  std::vector<double> effort_states_;

  // Initial states
  std::vector<double> init_position_states_;


  // Filtered states
  std::vector<double> filtered_position_states_;
  std::vector<double> filtered_velocity_states_;
  std::vector<double> filtered_effort_states_;

  // Commands
  std::vector<double> position_commands_;
};

}  // namespace daadbot_controller

#endif  // DAADBOT_INTERFACE_H
