#include "daadbot_controller/daadbot_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>

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
  prev_velocity_commands_.reserve(info_.joints.size());

  return CallbackReturn::SUCCESS;
}

// std::vector<hardware_interface::StateInterface> DaadbotInterface::export_state_interfaces()
// {
//   std::vector<hardware_interface::StateInterface> state_interfaces;

//   // Provide both position and velocity state interfaces
//   for (size_t i = 0; i < info_.joints.size(); i++)
//   {
//     state_interfaces.emplace_back(hardware_interface::StateInterface(
//         info_.joints[i].name, hardware_interface::HW_IF_POSITION, &position_states_[i]));

//     state_interfaces.emplace_back(hardware_interface::StateInterface(
//         info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &velocity_states_[i]));
//   }

//   return state_interfaces;
// }

// std::vector<hardware_interface::CommandInterface> DaadbotInterface::export_command_interfaces()
// {
//   std::vector<hardware_interface::CommandInterface> command_interfaces;

//   // Provide velocity command interface
//   for (size_t i = 0; i < info_.joints.size(); i++)
//   {
//     command_interfaces.emplace_back(hardware_interface::CommandInterface(
//         info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &velocity_commands_[i]));
//   }

//   return command_interfaces;
// }

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
  position_states_.assign(info_.joints.size(), 0.0);
  velocity_states_.assign(info_.joints.size(), 0.0);
  effort_states_.assign(info_.joints.size(), 0.0);

  try
  {
    esp_.Open(port_);
    esp_.SetBaudRate(LibSerial::BaudRate::BAUD_921600);

    Send "rmp\n" only once
    esp_.Write("rmp\n");
    esp_.DrainWriteBuffer();
    rmp_sent_ = true;

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
  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Reading ...");

  if (use_dummy_data_)
  {
    static double dummy_angle = 0.0;
    dummy_angle += 0.01;

    for (size_t i = 0; i < position_states_.size(); ++i)
    {
      position_states_[i] = std::sin(dummy_angle + i);
      velocity_states_[i] = std::cos(dummy_angle + i);
      effort_states_[i] = 0.1 * std::sin(dummy_angle + i);
    }

    return hardware_interface::return_type::OK;
  }

  if (!esp_.IsOpen())
  {
    RCLCPP_ERROR(rclcpp::get_logger("DaadbotInterface"), "Serial port not open, cannot read.");
    return hardware_interface::return_type::ERROR;
  }

  try
  {
    std::string response;
    esp_.ReadLine(response, '\n', 500); // timeout = 500 ms

    if (response.empty() || response.front() != '<' || response.back() != '>')
    {
      RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Invalid response: %s", response.c_str());
      return hardware_interface::return_type::ERROR;
    }

    response = response.substr(1, response.size() - 2); // Remove < and >
    std::istringstream ss(response);
    std::string token;
    ss >> token;

    if (token != "R")
    {
      RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Unexpected response format: %s", response.c_str());
      return hardware_interface::return_type::ERROR;
    }

    // Initialize all values to zero
    std::fill(position_states_.begin(), position_states_.end(), 0.0);
    std::fill(velocity_states_.begin(), velocity_states_.end(), 0.0);
    std::fill(effort_states_.begin(), effort_states_.end(), 0.0);

    size_t joint_index = 4;
    while (ss >> token && joint_index <= 6)
    {
      try
      {
        size_t dash1 = token.find('-');
        size_t dash2 = token.find('-', dash1 + 1);

        if (dash1 == std::string::npos || dash2 == std::string::npos)
          throw std::runtime_error("Invalid token: expected p-v-e format");

        std::string pos_str = token.substr(0, dash1);
        std::string vel_str = token.substr(dash1 + 1, dash2 - dash1 - 1);
        std::string eff_str = token.substr(dash2 + 1);

        position_states_[joint_index] = std::stod(pos_str);
        velocity_states_[joint_index] = std::stod(vel_str);
        effort_states_[joint_index] = std::stod(eff_str);
      }
      catch (...)
      {
        RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Invalid token format: %s", token.c_str());
        return hardware_interface::return_type::ERROR;
      }

      joint_index++;
    }
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("DaadbotInterface"), "Read failed: " << e.what());
    return hardware_interface::return_type::ERROR;
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type DaadbotInterface::write(const rclcpp::Time &time,
                                                        const rclcpp::Duration &period)
{
  if (!esp_.IsOpen())
  {
    RCLCPP_ERROR(rclcpp::get_logger("DaadbotInterface"), "Serial port not open, cannot write.");
    return hardware_interface::return_type::ERROR;
  }

  // Only compare indices 4, 5, 6 for change detection
  if (velocity_commands_[4] == prev_velocity_commands_[4] &&
      velocity_commands_[5] == prev_velocity_commands_[5] &&
      velocity_commands_[6] == prev_velocity_commands_[6])
  {
    return hardware_interface::return_type::OK;
  }

  try
  {
    std::ostringstream msg;
    msg << "<W";

    for (size_t i = 4; i <= 6; i++)
    {
      msg << " " << static_cast<int>(velocity_commands_[i]);
    }

    msg << ">";
    std::string command = msg.str();

    esp_.Write(command + "\n");
    esp_.DrainWriteBuffer();

    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Sent: " << command);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("DaadbotInterface"), "Write failed: " << e.what());
    return hardware_interface::return_type::ERROR;
  }

  // Update only relevant previous values
  for (size_t i = 4; i <= 6; i++)
  {
    prev_velocity_commands_[i] = velocity_commands_[i];
  }

  return hardware_interface::return_type::OK;
}


}  // namespace daadbot_controller

PLUGINLIB_EXPORT_CLASS(daadbot_controller::DaadbotInterface, hardware_interface::SystemInterface)