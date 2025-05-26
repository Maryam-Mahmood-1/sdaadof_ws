#include "daadbot_controller/daadbot_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <iomanip>
#include <regex>  
#include "rclcpp/rclcpp.hpp" 
#include "geometry_msgs/msg/vector3.hpp"


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

  node_ = rclcpp::Node::make_shared("daadbot_hardware_node");

  // state_pub_ = node_->create_publisher<geometry_msgs::msg::Vector3>("filtered_joint_states", 10);
  raw_state_pub_ = node_->create_publisher<geometry_msgs::msg::Vector3>("raw_joint_states", 10);


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
  position_states_.assign(info_.joints.size(), 0.0);
  init_position_states_.assign(info_.joints.size(), 0.0);
  velocity_states_.assign(info_.joints.size(), 0.0);
  effort_states_.assign(info_.joints.size(), 0.0);

  try
  {
    esp_.Open(port_);
    esp_.SetBaudRate(LibSerial::BaudRate::BAUD_921600);

    // Send "rmp\n" only once
    esp_.Write("rmp\n");
    esp_.DrainWriteBuffer();
    rmp_sent_ = true;

    rclcpp::sleep_for(std::chrono::milliseconds(50));

    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"),
                "Sent initial rmp command, ready to take commands.");

    // ---- START VOLTAGE POLLING THREAD ----
    run_voltage_thread_ = true;
    voltage_thread_ = std::thread(&DaadbotInterface::voltagePollingLoop, this);
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

  // ---- STOP VOLTAGE POLLING THREAD ----
  run_voltage_thread_ = false;
  if (voltage_thread_.joinable())
  {
    voltage_thread_.join();
  }

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
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Position States ...");
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "position_states_ size: %zu", position_states_.size());

    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Reading ...");

    if (!esp_.IsOpen())
    {
        RCLCPP_ERROR(rclcpp::get_logger("DaadbotInterface"), "Serial port not open, cannot read.");
        return hardware_interface::return_type::ERROR;
    }

    std::string response;

    try
    {
        {
            //std::lock_guard<std::mutex> lock(esp_mutex_);
            esp_.ReadLine(response, '\n', 1500);
        }

        // Trim leading/trailing whitespace
        response = response.substr(response.find_first_not_of(" \t\n\r"), response.find_last_not_of(" \t\n\r") + 1);
        RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Trimmed response: '%s'", response.c_str());

        if (response == "All motors stopped.")
        {
            RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Received stop message: 'All motors stopped.'");
            return hardware_interface::return_type::OK;
        }
        if (response == "Error reading voltage")
        {
            RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Voltage reading error");
            return hardware_interface::return_type::OK;
        }
        if (response.rfind("Unknown command", 0) == 0)  // starts with "Unknown command"
        {
            RCLCPP_INFO(
                rclcpp::get_logger("DaadbotInterface"),
                "Wrong command sent. Response: %s",
                response.c_str()
            );
            return hardware_interface::return_type::OK;
        }
        if (response == "Error: Invalid command format")
        {
            RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Received stop message: 'All motors stopped.'");
            return hardware_interface::return_type::OK;
        }


        // Basic format check: must be wrapped in <>
        if (response.empty() || response.front() != '<' || response.back() != '>')
        {
            RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Invalid response format: '%s'", response.c_str());
            return hardware_interface::return_type::OK;
        }

        response = response.substr(1, response.size() - 2);  // Remove < >

        if (response.empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Empty response after trimming.");
            return hardware_interface::return_type::ERROR;
        }

        // Handle voltage response: <a47.6>
        if (response.front() == 'a')
        {
            try
            {
                double voltage = std::stod(response.substr(1));

                RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Parsed voltage: " << voltage);

                // Compare with previous voltage
                if (previous_voltage_ >= 0.0)  // Ensure we skip first comparison
                {
                    double diff = voltage - previous_voltage_;
                    if (std::abs(diff) < 2)
                    {
                        RCLCPP_WARN_STREAM(
                            rclcpp::get_logger("DaadbotInterface"),
                            "Voltage difference good: " << diff << "V (previous: " << previous_voltage_ << ")");
                        previous_voltage_ = voltage;  // Store current for next time
                    }
                    else
                    {
                        RCLCPP_INFO_STREAM(
                            rclcpp::get_logger("DaadbotInterface"),
                            "Voltage difference is high: " << diff << "V (previous: " << previous_voltage_ << ")");
                        previous_voltage_ = previous_voltage_;

                    }
                    RCLCPP_INFO_STREAM(
                        rclcpp::get_logger("DaadbotInterface"),
                        "Voltage difference: " << diff << "V (previous: " << previous_voltage_ << ")");
                }

                else{
                  previous_voltage_ = voltage;
                }

                stp_sent_ = false;
                voltage_low_ = (previous_voltage_ < 47.0);

                RCLCPP_INFO_STREAM(
                    rclcpp::get_logger("DaadbotInterface"),
                    "Voltage check: " << voltage << "V. voltage_low_ = " << (voltage_low_ ? "true" : "false"));
            }
            catch (const std::exception &e)
            {
                RCLCPP_WARN_STREAM(rclcpp::get_logger("DaadbotInterface"), "Failed to parse voltage: " << e.what());
                return hardware_interface::return_type::ERROR;
            }

            return hardware_interface::return_type::OK;
        }

        // Handle joint state response: <R...>
        else if (response.front() == 'R')
        {
            std::istringstream ss(response.substr(1)); // Skip the 'R'
            std::string token;
            size_t joint_index = 4;

            while (ss >> token && joint_index <= 6)
            {
                try
                {
                    size_t sep1 = token.find('_');
                    size_t sep2 = token.rfind('_');

                    if (sep1 == std::string::npos || sep2 == std::string::npos || sep1 == sep2)
                    {
                        throw std::runtime_error("Invalid token: expected p_v_e format (e.g., 0_0_0)");
                    }

                    std::string pos_str = token.substr(0, sep1);
                    std::string vel_str = token.substr(sep1 + 1, sep2 - sep1 - 1);
                    std::string eff_str = token.substr(sep2 + 1);

                    auto parse_or_zero = [&](const std::string& str, const std::string& label, size_t idx) -> double {
                        if (str == "E")
                        {
                            RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Joint %zu: '%s' value estimated, set to 0.", idx, label.c_str());
                            return 0.0;
                        }
                        return std::stod(str);
                    };

                    double pos = parse_or_zero(pos_str, "position", joint_index);
                    double vel = parse_or_zero(vel_str, "velocity", joint_index);
                    double eff = parse_or_zero(eff_str, "effort", joint_index);

                    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Position: '" << pos << "'");
                    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Velocity: '" << vel << "'");
                    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Effort: '" << eff << "'");

                    geometry_msgs::msg::Vector3 raw_msg;
                    raw_msg.x = pos* M_PI / 180.0;
                    raw_msg.y = vel* M_PI / 180.0;
                    raw_msg.z = eff * 0.88;
                    raw_state_pub_->publish(raw_msg);

                    if (initial_read_)
                    {
                        double current_pos = (pos - init_position_states_[joint_index]) * M_PI / 180.0;
                        double pos_diff = std::abs(prev_position_states_[joint_index] - current_pos);

                        if (pos_diff > 0.75)
                        {
                            RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference too high, ignoring update");
                            RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Position diff: " << pos_diff);
                            RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Current position: " << prev_position_states_[joint_index]);
                            RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "New Position: " << current_pos);
                        }
                        else
                        {
                            position_states_[joint_index] = 0.9 * current_pos + 0.1 * prev_position_states_[joint_index];

                        }

                        if (std::abs(velocity_states_[joint_index] - (vel * M_PI / 180.0)) <= 0.4)
                        {
                            velocity_states_[joint_index] = vel * M_PI / 180.0;
                            RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Velocity diff: " << abs(velocity_states_[joint_index] - vel));
                        }
                        else {
                          RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference too high for velocity, ignoring update");
                          RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Current velocity: " << velocity_states_[joint_index]);
                          RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "New velocity: " << vel * M_PI / 180.0);
                        }

                        if (std::abs(effort_states_[joint_index] - (eff*0.88)) <= 2)
                        {
                            effort_states_[joint_index] = eff * 0.88;
                            RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Effort diff: " << abs(effort_states_[joint_index] - eff));
                        }
                        else
                        {
                          RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference too high for effort, ignoring update");
                          RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Current effort: " << effort_states_[joint_index]);
                          RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "New effort: " << eff * 0.88);
                        }
                    }
                    else
                    {
                        init_position_states_[joint_index] = pos;
                    }

                    // velocity_states_[joint_index] = vel * M_PI / 180.0;
                    // effort_states_[joint_index] = eff * 0.88;
                    prev_position_states_[joint_index] = position_states_[joint_index];
                }
                catch (const std::exception &e)
                {
                    RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Invalid token format: '%s'", token.c_str());
                    return hardware_interface::return_type::OK;
                }

                joint_index++;
            }
        }
        // Handle "All motors stopped." case
        else if (response == "All motors stopped.")
        {
            RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Received stop message: 'All motors stopped.'");
            // return hardware_interface::return_type::OK;
        }
        
        else
        {
            RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Unexpected response prefix or content: '%s'", response.c_str());
            return hardware_interface::return_type::OK;
        }
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("DaadbotInterface"), "Read failed: " << e.what());
        return hardware_interface::return_type::OK;
    }

    if (!initial_read_)
    {
        initial_read_ = true;
        RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Initial read completed.");
    }

    return hardware_interface::return_type::OK;
}



hardware_interface::return_type DaadbotInterface::write(const rclcpp::Time &time,
                                                         const rclcpp::Duration &period)
{
  static rclcpp::Time last_voltage_check_time = time;

  if (!esp_.IsOpen())
  {
    RCLCPP_ERROR(rclcpp::get_logger("DaadbotInterface"), "Serial port not open, cannot write.");
    return hardware_interface::return_type::ERROR;
  }

  try
  {
    // Check and update voltage every 100ms
    if ((time - last_voltage_check_time).nanoseconds() > 100000000) // 100ms
    {
      last_voltage_check_time = time;

      esp_.Write("rmv\n");
      esp_.DrainWriteBuffer();
      RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Sent: rmv");

    }

    if (!voltage_low_)
    {
      std::ostringstream msg;
      msg << "<W";

      for (size_t i = 4; i <= 6; i++)
      {
        float value_to_send = initial_read_ ? velocity_commands_[i] * (180.0f / M_PI) : 0.0f;
        msg << std::fixed << std::setprecision(2) << " " << value_to_send;
        prev_velocity_commands_[i] = value_to_send;
      }

      msg << ">";
      std::string command = msg.str();

      esp_.Write(command + "\n");
      esp_.DrainWriteBuffer();
      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Sent: " << command);
    }
    else
    {
      if (!stp_sent_){
        esp_.Write("stp\n");
        stp_sent_ = true;
        esp_.DrainWriteBuffer();
        RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Sent: stp");
      }
    }
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("DaadbotInterface"), "Write failed: " << e.what());
    return hardware_interface::return_type::ERROR;
  }

  return hardware_interface::return_type::OK;
}




void DaadbotInterface::voltagePollingLoop()
{
    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "In Voltage Pooling Loop:");
    const std::regex voltage_pattern(R"(<a(\d+(\.\d+)?)>)");  // Matches <a47.3> or <a47>

    while (false)
    {
        if (initial_read_)
        {
            try
            {
                std::string voltage_response;

                {
                    // 🔒 Protect access to esp_ with a mutex
                    std::lock_guard<std::mutex> lock(esp_mutex_);
                    esp_.Write("rmv\n");
                    esp_.ReadLine(voltage_response, '\n', 500);
                }

                if (!voltage_response.empty())
                {
                    std::smatch match;
                    if (std::regex_search(voltage_response, match, voltage_pattern))
                    {
                        double voltage = std::stod(match[1]);  // Parse as float
                        voltage_value_ = voltage;

                        voltage_low_ = (voltage < 47.0);  // Compare as float
                    }
                    else
                    {
                        RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"),
                                    "Malformed voltage response: '%s'", voltage_response.c_str());
                    }
                }
                else
                {
                    RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"),
                                "Timeout while waiting for voltage response");
                }
            }
            catch (const std::exception &e)
            {
                RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"),
                            "Voltage read error: %s", e.what());
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}



}  // namespace daadbot_controller

PLUGINLIB_EXPORT_CLASS(daadbot_controller::DaadbotInterface, hardware_interface::SystemInterface)