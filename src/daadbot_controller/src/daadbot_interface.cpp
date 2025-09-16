#include "daadbot_controller/daadbot_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include "rclcpp/rclcpp.hpp"
#include <iomanip>
#include <chrono>
#include <fstream>
#include <filesystem>

#include <cmath>
#include <cstring>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <net/if.h>
#include <map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


namespace daadbot_controller
{

// Motor ID mapping
std::map<size_t, uint16_t> joint_id_map = {
    {0, 0x000}, // joint_1 (always zero)
    {1, 0x142}, // joint_2
    {2, 0x144}, // joint_3
    {3, 0x145}, // joint_4
    {4, 0x146}, // joint_5 (read/write)
    {5, 0x147}, // joint_6 (read/write)
    {6, 0x148}, // joint_7 (read/write)
    {7, 0x000}  // gear1_joint (always zero)
};


static const std::vector<size_t> active_joints = {5, 6, 7};

bool readMultiTurnAngle(int fd, uint32_t motor_id, double &angle_deg_out)
  {
    struct can_frame cmd = {};
    cmd.can_id = motor_id;
    cmd.can_dlc = 8;
    cmd.data[0] = 0x92;

    if (write(fd, &cmd, sizeof(cmd)) < 0)
      return false;

    struct can_frame reply;
    struct timeval timeout = {0, 10000}; // 10 ms
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);

    int ret = select(fd + 1, &rfds, nullptr, nullptr, &timeout);
    if (ret > 0 && FD_ISSET(fd, &rfds))
    {
      ssize_t bytes = read(fd, &reply, sizeof(reply));
      if (bytes > 0 && reply.data[0] == 0x92 && reply.can_id == (motor_id + 0x100))
      {
        int32_t raw_angle;
        std::memcpy(&raw_angle, &reply.data[4], sizeof(raw_angle));
        angle_deg_out = raw_angle * 0.01;
        return true;
      }
    }
    return false;
  }

  void sendA4PositionCommand(int fd, uint32_t motor_id, int32_t angle_cdeg, uint16_t max_speed)
  {
    struct can_frame frame = {};
    frame.can_id = motor_id;
    frame.can_dlc = 8;
    frame.data[0] = 0xA4;
    frame.data[1] = 0x00;
    frame.data[2] = static_cast<uint8_t>(max_speed);
    frame.data[3] = static_cast<uint8_t>(max_speed >> 8);
    frame.data[4] = static_cast<uint8_t>(angle_cdeg);
    frame.data[5] = static_cast<uint8_t>(angle_cdeg >> 8);
    frame.data[6] = static_cast<uint8_t>(angle_cdeg >> 16);
    frame.data[7] = static_cast<uint8_t>(angle_cdeg >> 24);
    write(fd, &frame, sizeof(frame));
  }



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

  socket_fd_ = -1; // start invalid

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

CallbackReturn DaadbotInterface::on_activate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Starting robot hardware...");

  // open CAN socket
  socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (socket_fd_ < 0)
  {
    RCLCPP_ERROR(rclcpp::get_logger("DaadbotInterface"), "Failed to open CAN socket");
    return CallbackReturn::ERROR;
  }

  struct ifreq ifr;
  std::strncpy(ifr.ifr_name, "can0", IFNAMSIZ - 1);
  if (ioctl(socket_fd_, SIOCGIFINDEX, &ifr) < 0)
  {
    RCLCPP_ERROR(rclcpp::get_logger("DaadbotInterface"), "CAN interface can0 not found");
    return CallbackReturn::ERROR;
  }

  struct sockaddr_can addr = {};
  addr.can_family = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;
  if (bind(socket_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0)
  {
    RCLCPP_ERROR(rclcpp::get_logger("DaadbotInterface"), "Failed to bind CAN socket");
    return CallbackReturn::ERROR;
  }

  // read initial positions as offsets
  for (auto &[idx, can_id] : joint_id_map)
  {
    double init_angle = 0.0;
    if (readMultiTurnAngle(socket_fd_, can_id, init_angle))
    {
      init_position_states_[idx] = init_angle;
      RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"),
                  "Joint %zu initial angle: %.2f deg", idx, init_angle);
    }
    else
    {
      RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"),
                  "Failed to read initial angle for joint %zu", idx);
    }
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn DaadbotInterface::on_deactivate(const rclcpp_lifecycle::State &previous_state)
{
  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Stopping robot hardware ...");
  if (socket_fd_ != -1)
  {
    close(socket_fd_);
    socket_fd_ = -1;
  }
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type DaadbotInterface::read(
    const rclcpp::Time &, const rclcpp::Duration &)
{
  static bool initial_read_done = false;

  // Fixed joints (joint_1 and gear1_joint)
  position_states_[0] = 0.0;
  velocity_states_[0] = 0.0;
  effort_states_[0] = 0.0;
  position_states_.back() = 0.0;
  velocity_states_.back() = 0.0;
  effort_states_.back() = 0.0;

  // Only read from active joints (5, 6, 7)
  for (size_t idx : active_joints)
  {
    auto it = joint_id_map.find(idx);
    if (it == joint_id_map.end() || it->second == 0x000) continue;

    double angle_deg = 0.0;
    if (readMultiTurnAngle(socket_fd_, it->second, angle_deg))
    {
      if (!initial_read_done)
      {
        // Store first reading as zero/reference
        init_position_states_[idx] = angle_deg;
      }

      // Position relative to stored zero
      position_states_[idx] = (angle_deg - init_position_states_[idx]) * M_PI / 180.0;
      velocity_states_[idx] = 0.0;
      effort_states_[idx] = 0.0;
    }
  }

  if (!initial_read_done)
  {
    initial_read_done = true;
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Initial positions stored.");
  }

  // Sync filtered states
  filtered_position_states_ = position_states_;
  filtered_velocity_states_ = velocity_states_;
  filtered_effort_states_ = effort_states_;

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type DaadbotInterface::write(
    const rclcpp::Time &, const rclcpp::Duration &)
{
  const double deadband = 0.1; // degrees
  const double scale = 5.0;
  const double min_speed = 1.0;

  // Only write to active joints (5, 6, 7)
  for (size_t idx : active_joints)
  {
    auto it = joint_id_map.find(idx);
    if (it == joint_id_map.end() || it->second == 0x000) continue;

    uint32_t can_id = it->second;

    // Target in degrees (relative to initial position)
    double target_deg = (position_commands_[idx] * 180.0 / M_PI) + init_position_states_[idx];
    double current_deg = position_states_[idx] * 180.0 / M_PI + init_position_states_[idx];
    double error = target_deg - current_deg;

    uint16_t max_speed_limit = (can_id == 0x142) ? 300 : 50;

    double speed_cmd;
    if (std::abs(error) <= deadband)
      speed_cmd = min_speed;
    else
      speed_cmd = std::max(min_speed,
                           max_speed_limit * (1.0 - std::exp(-std::abs(error) / scale)));

    int32_t angle_cdeg = static_cast<int32_t>(target_deg * 100.0);
    sendA4PositionCommand(socket_fd_, can_id, angle_cdeg, static_cast<uint16_t>(speed_cmd));

    // Prevent flooding the CAN bus
    std::this_thread::sleep_for(std::chrono::microseconds(2200));
  }

  return hardware_interface::return_type::OK;
}

} // namespace daadbot_controller

PLUGINLIB_EXPORT_CLASS(daadbot_controller::DaadbotInterface, hardware_interface::SystemInterface)
