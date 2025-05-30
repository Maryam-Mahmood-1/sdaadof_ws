#ifndef DAADBOT_INTERFACE_H
#define DAADBOT_INTERFACE_H

#include <rclcpp/rclcpp.hpp>
#include <hardware_interface/system_interface.hpp>
#include <libserial/SerialPort.h>
#include <rclcpp_lifecycle/state.hpp>
#include <rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp>
#include "geometry_msgs/msg/vector3.hpp"

#include <thread>        // <-- Required for std::thread
#include <atomic>        // <-- Optional: for thread-safe voltage variable
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
  virtual CallbackReturn on_activate(const rclcpp_lifecycle::State &previous_state) override;
  virtual CallbackReturn on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

  // SystemInterface overrides
  virtual CallbackReturn on_init(const hardware_interface::HardwareInfo &hardware_info) override;
  virtual std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  virtual std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  virtual hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  virtual hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  // --- Serial & ESP communication ---
  LibSerial::SerialPort esp_;
  std::string port_;

  // --- Motor state tracking ---
  bool rmp_sent_ = false;
  bool use_dummy_data_ = false;
  bool controllers_ready_ = false;
  bool initial_read_ = false;
  bool power_ = false;
  bool stp_sent_ = false;
  bool any_change = true;
  bool initial_write_ = false;

  double previous_voltage_ = -1.0;
  double exp_coeff_ = 0.081;

  std::vector<double> velocity_commands_;
  std::vector<double> prev_velocity_commands_;
  std::vector<double> prev_position_states_;
  std::vector<double> position_states_;
  std::vector<double> init_position_states_;
  std::vector<double> velocity_states_;
  std::vector<double> effort_states_;
  std::vector<double> unfil_pos_states_;
  std::vector<double> unfil_vel_states_;
  std::vector<double> unfil_effort_states_;
  


  std::thread voltage_thread_;               // <-- Thread for polling
  std::atomic<bool> run_voltage_thread_{false}; // <-- Control loop flag
  std::atomic<double> voltage_value_{0.0};   // <-- Latest voltage value (atomic for thread safety)
  std::atomic<bool> voltage_low_{false};  // true if voltage < 47
  std::mutex esp_mutex_;

  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr raw_state_pub_;
  rclcpp::Node::SharedPtr node_;



};

}  // namespace daadbot_controller

#endif  // DAADBOT_INTERFACE_H
