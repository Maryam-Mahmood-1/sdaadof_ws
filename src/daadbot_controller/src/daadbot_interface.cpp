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
  if (esp_.IsOpen())
  {
    try
    {
      esp_.Close();
    }
    catch (...)
    {
      RCLCPP_FATAL_STREAM(rclcpp::get_logger("DaadbotInterface"),
                          "Something went wrong while closing connection with port " << port_);
    }
  }
}

CallbackReturn DaadbotInterface::on_init(const hardware_interface::HardwareInfo &hardware_info)
{
  CallbackReturn result = hardware_interface::SystemInterface::on_init(hardware_info);
  if (result != CallbackReturn::SUCCESS)
  {
    return result;
  }

  try
  {
    port_ = info_.hardware_parameters.at("port");
  }
  catch (const std::out_of_range &e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("DaadbotInterface"), "No Serial Port provided! Aborting");
    return CallbackReturn::FAILURE;
  }

  RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"),
                     "Detected Joints: " << info_.joints.size());

  velocity_commands_.reserve(info_.joints.size());
  velocity_states_.reserve(info_.joints.size());
  effort_states_.reserve(info_.joints.size());
  position_states_.reserve(info_.joints.size());
  init_position_states_.reserve(info_.joints.size());
  prev_velocity_commands_.reserve(info_.joints.size());
  prev_position_states_.reserve(info_.joints.size());
  prev_velocity_states_.reserve(info_.joints.size());
  unfil_pos_states_.reserve(info_.joints.size());
  unfil_vel_states_.reserve(info_.joints.size());
  unfil_effort_states_.reserve(info_.joints.size());

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
                "Exporting command interface: %s/velocity", info_.joints[i].name.c_str());
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &velocity_commands_[i]));
  }

  return command_interfaces;
}



CallbackReturn DaadbotInterface::on_activate(const rclcpp_lifecycle::State &previous_state)
{
  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Starting robot hardware ...");

  velocity_commands_.assign(info_.joints.size(), 0.0);
  prev_velocity_commands_.assign(info_.joints.size(), 0.0);
  prev_position_states_.assign(info_.joints.size(), 0.0);
  prev_velocity_states_.assign(info_.joints.size(), 0.0);
  position_states_.assign(info_.joints.size(), 0.0);
  init_position_states_.assign(info_.joints.size(), 0.0);
  velocity_states_.assign(info_.joints.size(), 0.0);
  effort_states_.assign(info_.joints.size(), 0.0);
  unfil_pos_states_.assign(info_.joints.size(), 0.0);
  unfil_vel_states_.assign(info_.joints.size(), 0.0);
  unfil_effort_states_.assign(info_.joints.size(), 0.0);

  try
  {
    esp_.Open(port_);
    esp_.SetBaudRate(LibSerial::BaudRate::BAUD_921600);

    // Send "rmp\n" only once
    esp_.Write("rmp\n");
    esp_.DrainWriteBuffer();
    rmp_sent_ = true;

    // Add a delay of 500 milliseconds (adjust as needed)
    rclcpp::sleep_for(std::chrono::milliseconds(50));

    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"),
                "Sent initial rmp command, ready to take commands.");
  }
  catch (...)
  {
    RCLCPP_FATAL_STREAM(rclcpp::get_logger("DaadbotInterface"),
                        "Something went wrong while interacting with port " << port_);
    return CallbackReturn::FAILURE;
  }

  return CallbackReturn::SUCCESS;
}


CallbackReturn DaadbotInterface::on_deactivate(const rclcpp_lifecycle::State &previous_state)
{
  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Stopping robot hardware ...");

  if (esp_.IsOpen())
  {
    try
    {
      esp_.Close();
    }
    catch (...)
    {
      RCLCPP_FATAL_STREAM(rclcpp::get_logger("DaadbotInterface"),
                          "Something went wrong while closing connection with port " << port_);
    }
  }

  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Hardware stopped");
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type DaadbotInterface::read(const rclcpp::Time &time,
                                                       const rclcpp::Duration &period)
{
  auto start_total = std::chrono::steady_clock::now();
  static bool csv_initialized = false;
  static std::ofstream rtiming_csv_;
  static std::ofstream comm_log;
  static std::ofstream roundtrip_log;
  static std::chrono::steady_clock::time_point last_read_start;

  auto current_read_start = std::chrono::steady_clock::now();
  long delta_since_last_read_us = 0;

  if (last_read_start.time_since_epoch().count() != 0) {
    delta_since_last_read_us = std::chrono::duration_cast<std::chrono::microseconds>(current_read_start - last_read_start).count();
  }
  last_read_start = current_read_start;
  if (!csv_initialized) {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream filename_ss;
    filename_ss << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S");
    std::string output_dir = "/home/maryam-mahmood/udaadbot_ws/robot_logs";
    if (!std::filesystem::exists(output_dir)) {
      std::filesystem::create_directories(output_dir);
    }
    std::string readt_path = output_dir + "/read_timings_" + filename_ss.str() + ".csv";
    std::string comm_path = output_dir + "/comm_timing_" + filename_ss.str() + ".csv";
    std::string roundtripath = output_dir + "/write_to_read_timing_" + filename_ss.str() + ".csv";
    rtiming_csv_.open(readt_path);
    comm_log.open(comm_path);
    roundtrip_log.open(roundtripath);
    if (rtiming_csv_.is_open()) {
      rtiming_csv_ << "timestamp_us,serial_us,parse_us,total_us,delta_since_last_read_us\n";
    }
    csv_initialized = true;
  }
  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Position States ...");
  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "position_states_ size: %zu", position_states_.size());
  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Reading ...");
  if (!esp_.IsOpen()) {
    RCLCPP_ERROR(rclcpp::get_logger("DaadbotInterface"), "Serial port not open, cannot read.");
    return hardware_interface::return_type::ERROR;
  }
  try {
    auto serial_start = std::chrono::steady_clock::now();

    std::string response;
    esp_.ReadLine(response, '\n', 500);

    auto serial_end = std::chrono::steady_clock::now();
    auto serial_us = std::chrono::duration_cast<std::chrono::microseconds>(serial_end - serial_start).count();
    response = response.substr(response.find_first_not_of(" \t\n\r"), response.find_last_not_of(" \t\n\r") + 1);
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Trimmed response: '%s'", response.c_str());
    if (response.empty() || response.front() != '<' || response.back() != '>') {
      RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Invalid response: '%s'", response.c_str());
      return hardware_interface::return_type::ERROR;
    }
    auto read_start = std::chrono::steady_clock::now();
    double communication_time_ = std::chrono::duration<double, std::milli>(read_start - write_sent_).count();
    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"),
                       "Time from write sent to read before parsing: " << communication_time_ << " ms");

    comm_log << rclcpp::Time(read_start.time_since_epoch().count()).seconds()
             << "," << communication_time_ << std::endl;
    auto parse_start = std::chrono::steady_clock::now();
    response = response.substr(1, response.size() - 2);
    if (response.empty() || response.front() != 'R') {
      RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Unexpected response format: '%s'", response.c_str());
      return hardware_interface::return_type::ERROR;
    }
    std::istringstream ss(response.substr(1));
    std::string token;
    std::vector<size_t> joint_indices = {3, 5, 6};
    size_t token_index = 0;
    while (ss >> token && token_index < joint_indices.size()) {
      size_t joint_index = joint_indices[token_index];
      try {
        size_t sep1 = token.find('_');
        size_t sep2 = token.rfind('_');

        if (sep1 == std::string::npos || sep2 == std::string::npos || sep1 == sep2) {
          throw std::runtime_error("Invalid token: expected p_v_e format (e.g., 0_0_0)");
        }

        std::string pos_str = token.substr(0, sep1);
        std::string vel_str = token.substr(sep1 + 1, sep2 - sep1 - 1);
        std::string eff_str = token.substr(sep2 + 1);

        auto parse_or_zero = [&](const std::string& str, const std::string& label, size_t idx) -> double {
          if (str == "E") {
            RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Joint %zu: '%s' value estimated, set to 0.", idx, label.c_str());
            return 0.0;
          }
          return std::stod(str);
        };
        double pos = parse_or_zero(pos_str, "position", joint_index);
        double vel = parse_or_zero(vel_str, "velocity", joint_index);
        double eff = parse_or_zero(eff_str, "effort", joint_index);
        double current_pos_rad = (pos - init_position_states_[joint_index]) * M_PI / 180.0; 
        double jump = prev_position_states_[joint_index] - current_pos_rad;
        if (initial_read_) {
          if (abs(prev_position_states_[joint_index] - (current_pos_rad)) > 0.30) {
            RCLCPP_INFO_STREAM(
              rclcpp::get_logger("DaadbotInterface"),
              "Position jump too large, ignoring. Prev: " << prev_position_states_[joint_index]
              << ", Current: " << current_pos_rad
              << ", Jump: " << jump
            );
          } else {
            if (abs(jump) < 0.0174533) { // 1 degree in radians
              unfil_pos_states_[joint_index] = prev_position_states_[joint_index] + velocity_commands_[joint_index] * 0.0118;
              position_states_[joint_index] = unfil_pos_states_[joint_index];
            }
            else{
              unfil_pos_states_[joint_index] = current_pos_rad;
              unfil_pos_states_[joint_index] = current_pos_rad * 0.55 +
                                               (prev_position_states_[joint_index] + velocity_commands_[joint_index] * 0.0108) * (1 - 0.55);              
              position_states_[joint_index] = unfil_pos_states_[joint_index] * exp_coeff_ +
                                            prev_position_states_[joint_index] * (1 - exp_coeff_);
            }
            RCLCPP_INFO_STREAM(
              rclcpp::get_logger("DaadbotInterface"),
              "Position jump good. Prev: " << prev_position_states_[joint_index]
              << ", Current: " << unfil_pos_states_[joint_index]
              << ", Filtered: " << position_states_[joint_index]
              << ", Jump: " << jump
            );
          }
          double current_vel_rad = vel * M_PI / 180.0;
          if (std::abs(velocity_states_[joint_index] - current_vel_rad) <= 0.4) {
            unfil_vel_states_[joint_index] = current_vel_rad;
            velocity_states_[joint_index] = unfil_vel_states_[joint_index] * 0.3 +
                                            prev_velocity_commands_[joint_index] * (1 - 0.3);
            double jump = prev_velocity_states_[joint_index] - current_vel_rad;
            RCLCPP_INFO_STREAM(
              rclcpp::get_logger("DaadbotInterface"),
              "Velocity jump good. Prev: " << prev_velocity_states_[joint_index]
              << ", Current: " << unfil_vel_states_[joint_index]
              << ", Filtered: " << velocity_states_[joint_index]
              << ", Jump: " << jump
            );
          } else {
            double jump = prev_velocity_states_[joint_index] - current_vel_rad;
            RCLCPP_INFO_STREAM(
              rclcpp::get_logger("DaadbotInterface"),
              "Velocity jump too large, ignoring. Prev: " << velocity_states_[joint_index]
              << ", Current: " << current_vel_rad
              << ", Jump: " << jump
            );
          }
          if (std::abs(effort_states_[joint_index] - (eff * 0.88)) <= 2) {
            unfil_effort_states_[joint_index] = eff * 0.88;
            effort_states_[joint_index] = unfil_effort_states_[joint_index]*exp_coeff_ + effort_states_[joint_index]*(1 - exp_coeff_);           
          }
        } else {
          init_position_states_[joint_index] = pos;
        }
        prev_position_states_[joint_index] = position_states_[joint_index];
        prev_velocity_states_[joint_index] = velocity_states_[joint_index];
      } catch (const std::exception &e) {
        RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Invalid token format: '%s'", token.c_str());
        return hardware_interface::return_type::ERROR;
      }
      token_index++;
    }
    auto read_end = std::chrono::steady_clock::now();
    double write_to_read_ms = std::chrono::duration<double, std::milli>(read_end - write_sent_).count();
    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"),
                       "Time from write sent to read end: " << write_to_read_ms << " ms");
    roundtrip_log << rclcpp::Time(read_start.time_since_epoch().count()).seconds()
                  << "," << write_to_read_ms << std::endl;
    auto parse_end = std::chrono::steady_clock::now();
    auto parse_us = std::chrono::duration_cast<std::chrono::microseconds>(parse_end - parse_start).count();
    auto end_total = std::chrono::steady_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"),
                "[TIMING] Serial: %ld µs | Parse: %ld µs | Total: %ld µs | Δ Read: %ld µs",
                serial_us, parse_us, total_us, delta_since_last_read_us);
    if (rtiming_csv_.is_open()) {
      auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
      rtiming_csv_ << now_us << "," << serial_us << "," << parse_us << "," << total_us
                   << "," << delta_since_last_read_us << "\n";
    }
  } catch (const std::exception &e) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("DaadbotInterface"), "Read failed: " << e.what());
    return hardware_interface::return_type::ERROR;
  }
  if (!initial_read_) {
    initial_read_ = true;
    // auto now = std::chrono::steady_clock::now();
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Initial read completed.");
  }
  if (initial_write_) {
    any_change = false;
  }
  return hardware_interface::return_type::OK;
}




hardware_interface::return_type DaadbotInterface::write(const rclcpp::Time &time,
                                                        const rclcpp::Duration &period)
{
  using namespace std::chrono;
  auto total_start = high_resolution_clock::now();

  static bool csv_initialized = false;
  static std::ofstream write_csv_;
  static std::string folder_path = "/home/maryam-mahmood/udaadbot_ws/robot_logs";

  static high_resolution_clock::time_point last_write_start_time;
  static bool first_write_call = true;

  if (!csv_initialized) {
    std::filesystem::create_directories(folder_path);

    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream filename_ss;
    filename_ss << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S");

    std::string full_path = folder_path + "/write_timing_" + filename_ss.str() + ".csv";
    write_csv_.open(full_path);
    if (write_csv_.is_open()) {
      write_csv_ << "timestamp_us,computation_us,communication_us,misc_us,total_us,delta_write_us\n";
    }

    csv_initialized = true;
  }

  RCLCPP_ERROR(rclcpp::get_logger("DaadbotInterface"), "In write func.");
  if (!esp_.IsOpen()) {
    RCLCPP_ERROR(rclcpp::get_logger("DaadbotInterface"), "Serial port not open, cannot write.");
    return hardware_interface::return_type::ERROR;
  }

  // ---- COMPUTATION START ----
  auto computation_start = high_resolution_clock::now();

  std::ostringstream msg;
  msg << "<W";

  std::vector<size_t> joint_indices = {3, 5, 6};  // Updated indices
  bool any_change = false;

  for (size_t i : joint_indices)
  {
    float value_to_send = initial_read_ ? (velocity_commands_[i] * (180.0f / M_PI) * 0.95) : 0.0f;

    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Joint: " << i);
    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Previous Velocity command: " << prev_velocity_commands_[i]);
    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Current Velocity command: " << velocity_commands_[i]);
    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference: " << std::abs(velocity_commands_[i] - prev_velocity_commands_[i]));

    any_change = true;

    msg << std::fixed << std::setprecision(2) << " " << value_to_send;
    prev_velocity_commands_[i] = value_to_send * (M_PI / 180.0f);
  }

  if (initial_read_ && !any_change && initial_write_) {
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "No change in velocity commands, skipping write.");
    esp_.Write("rmp\n");
    return hardware_interface::return_type::OK;
  }

  msg << ">";
  std::string command = msg.str();

  auto computation_end = high_resolution_clock::now();

  // ---- COMMUNICATION START ----
  auto communication_start = high_resolution_clock::now();
  try {
    esp_.Write(command + "\n");
    esp_.DrainWriteBuffer();
    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Sent: " << command);
    write_sent_ = std::chrono::steady_clock::now();
  } catch (const std::exception &e) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("DaadbotInterface"), "Write failed: " << e.what());
    return hardware_interface::return_type::ERROR;
  }
  auto communication_end = high_resolution_clock::now();
  // ---- COMMUNICATION END ----

  if (!initial_write_) {
    initial_write_ = true;
  }

  auto total_end = high_resolution_clock::now();

  // ---- TIMING CALCULATIONS ----
  auto total_us = duration_cast<microseconds>(total_end - total_start).count();
  auto computation_us = duration_cast<microseconds>(computation_end - computation_start).count();
  auto communication_us = duration_cast<microseconds>(communication_end - communication_start).count();
  auto misc_us = total_us - computation_us - communication_us;

  // ---- WRITE DELTA TIME ----
  uint64_t delta_write_us = 0;
  if (!first_write_call) {
    delta_write_us = duration_cast<microseconds>(total_start - last_write_start_time).count();
  } else {
    first_write_call = false;
  }
  last_write_start_time = total_start;

  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"),
              "WRITE TIMING (us): computation= %ld, communication= %ld, misc= %ld, total= %ld, delta_write= %ld",
              computation_us, communication_us, misc_us, total_us, delta_write_us);

  // ---- CSV Logging ----
  try {
    if (write_csv_.is_open()) {
      uint64_t timestamp_us = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
      write_csv_ << timestamp_us << "," << computation_us << "," << communication_us << ","
                 << misc_us << "," << total_us << "," << delta_write_us << "\n";
    }
  } catch (...) {
    RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Failed to write to CSV log.");
  }

  return hardware_interface::return_type::OK;
}

}  // namespace daadbot_controller

PLUGINLIB_EXPORT_CLASS(daadbot_controller::DaadbotInterface, hardware_interface::SystemInterface)