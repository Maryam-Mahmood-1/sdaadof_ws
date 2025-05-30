#include "daadbot_controller/daadbot_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <iomanip>

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


#include "rclcpp/rclcpp.hpp"  // Include this header for rclcpp::sleep_for

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
  if (true){
    // any_change = false;
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Position States ...");
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "position_states_ size: %zu", position_states_.size());
    // for (size_t i = 0; i < position_states_.size(); ++i)
    // {
    //     RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "position_states_[%zu]: %f", i, position_states_[i]);
    // }
    // for (const auto &pos : position_states_) {
    //   std::cout << pos << " ";
    // }
    // std::cout << std::endl;
  
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

        // Trim leading and trailing whitespace
        response = response.substr(response.find_first_not_of(" \t\n\r"), response.find_last_not_of(" \t\n\r") + 1);
        RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Trimmed response: '%s'", response.c_str());

        // Check < > and 'R' prefix
        if (response.empty() || response.front() != '<' || response.back() != '>')
        {
            RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Invalid response: '%s'", response.c_str());
            return hardware_interface::return_type::ERROR;
        }

        response = response.substr(1, response.size() - 2); // remove < >
        if (response.empty() || response.front() != 'R')
        {
            RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Unexpected response format: '%s'", response.c_str());
            return hardware_interface::return_type::ERROR;
        }
        // RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Here after checking < > and 'R' prefix");
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
                // RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Position String: '%s'", pos_str.c_str());
                std::string vel_str = token.substr(sep1 + 1, sep2 - sep1 - 1);
                // RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Velocity String: '%s'", vel_str.c_str());
                std::string eff_str = token.substr(sep2 + 1);
                // RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Effort String: '%s'", eff_str.c_str());

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

                RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "prev position: '" << prev_position_states_[joint_index] + init_position_states_[joint_index]<< "'");
                RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "init position: '" << init_position_states_[joint_index] << "'");
                RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "prev position rel: '" << prev_position_states_[joint_index] << "'");


                if (initial_read_)
                {
                  RCLCPP_INFO_STREAM(
                    rclcpp::get_logger("DaadbotInterface"),
                    "1 time step diff: '" << abs(prev_position_states_[joint_index] - ((pos - init_position_states_[joint_index]) * M_PI / 180.0)) << "'"
                );
                                    // Converting degrees to radians before visualizing in Rviz (joint_states)
                    if (abs(prev_position_states_[joint_index] - ((pos - init_position_states_[joint_index]) * M_PI / 180.0)) > 0.30){
                      position_states_[joint_index] = position_states_[joint_index];
                      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Prev Position: '" << prev_position_states_[joint_index] << "'");
                      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Currrent Position: '" << (pos - init_position_states_[joint_index]) * M_PI / 180.0 << "'");
                      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Current Position being written: '" << position_states_[joint_index] << "'");
                      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference: '" << abs(prev_position_states_[joint_index] - ((pos - init_position_states_[joint_index]) * M_PI / 180.0)) << "'");
                      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Position diff filtered: " << abs(prev_position_states_[joint_index] - ((pos - init_position_states_[joint_index]) * M_PI / 180.0 )));
                      
                      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference is too high");
                    }
                    else{
                        unfil_pos_states_[joint_index] = (pos - init_position_states_[joint_index]) * M_PI / 180.0;
                        position_states_[joint_index] = unfil_pos_states_[joint_index]*exp_coeff_ + prev_position_states_[joint_index]*(1-exp_coeff_);
                        RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference: '" << abs(prev_position_states_[joint_index] - ((pos - init_position_states_[joint_index]) * M_PI / 180.0)) << "'");
                        RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference is good");
                        // RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Position diff filtered: " << abs(prev_position_states_[joint_index] - position_states_[joint_index]));
                        RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "New Position filtered updated: " << position_states_[joint_index]);
                        // RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Here after initial read and now setting position state '" << position_states_[joint_index] << "'");
                    }

                    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Velocity diff: " << abs(velocity_states_[joint_index] - (vel * M_PI / 180.0)));
                    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Current velocity: " << velocity_states_[joint_index]);
                    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "New velocity: " << vel * M_PI / 180.0);
                    
                    if (std::abs(velocity_states_[joint_index] - (vel * M_PI / 180.0)) <= 0.4)
                    {
                        unfil_vel_states_[joint_index] = vel * M_PI / 180.0;
                        velocity_states_[joint_index] = unfil_vel_states_[joint_index]*(exp_coeff_) + prev_velocity_commands_[joint_index]*(1-exp_coeff_);
                        RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Velocity diff filtered: " << abs(velocity_states_[joint_index] - (vel * M_PI / 180.0)));
                        RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "New Velocity filtered updated: " << velocity_states_[joint_index]);
                        
                    }
                    else {
                      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference too high for velocity, ignoring update");
                      
                    }

                    if (std::abs(effort_states_[joint_index] - (eff*0.88)) <= 2)
                    {
                        unfil_effort_states_[joint_index] = eff * 0.88;
                        effort_states_[joint_index] = unfil_effort_states_[joint_index]*exp_coeff_ + effort_states_[joint_index]*(1-exp_coeff_);
                        RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Effort diff filtered: " << abs(effort_states_[joint_index] - eff));
                        RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "New Effort filtered updated: " << effort_states_[joint_index]);
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
                    // RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Here before initial read and now setting initial position state '" << init_position_states_[joint_index] << "'");
                    init_position_states_[joint_index] = pos;
                    // RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Here before initial read and after setting initial position state '" << init_position_states_[joint_index] << "'");

                }
                // RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Here after initial read and now setting position state '" << position_states_[joint_index] << "'");

                // velocity_states_[joint_index] = vel * M_PI / 180.0; // Converting dps to rad/s
                // effort_states_[joint_index] = eff;
                prev_position_states_[joint_index] = position_states_[joint_index];
            }
            catch (const std::exception &e)
            {
                RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Invalid token format: '%s'", token.c_str());
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

    if (!initial_read_)
    {
        initial_read_ = true;
        RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Initial read completed.");
    }
    if (initial_write_)
    {
      any_change = false;
    }
    
    return hardware_interface::return_type::OK;
  }
  else{
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "No change in velocity states, skipping read.");
    return hardware_interface::return_type::OK;
  }
}


hardware_interface::return_type DaadbotInterface::write(const rclcpp::Time &time,
                                                        const rclcpp::Duration &period)
{
  RCLCPP_ERROR(rclcpp::get_logger("DaadbotInterface"), "In write func.");
  if (!esp_.IsOpen())
  {
    RCLCPP_ERROR(rclcpp::get_logger("DaadbotInterface"), "Serial port not open, cannot write.");
    return hardware_interface::return_type::ERROR;
  }

  std::ostringstream msg;
  msg << "<W";


  for (size_t i = 4; i <= 6; i++)
  {
    // Converting radians to degrees before sending to hardware (speeds in 0.01 dsp)
    float value_to_send = initial_read_ ? velocity_commands_[i] * (180.0f / M_PI) : 0.0f;

    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Joint: " << i);
    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Previous Velocity command: " << prev_velocity_commands_[i]);
    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Current Velocity command: " << velocity_commands_[i]);
    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference: " << std::abs(velocity_commands_[i] - prev_velocity_commands_[i]));

    // Change detection (only matters if initial_read_ is true)
    if (true)
    {
      any_change = true;
      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Joint: " << i);
      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Previous Velocity command: " << prev_velocity_commands_[i]);
      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Current Velocity command: " << velocity_commands_[i]);
      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference: " << std::abs(velocity_commands_[i] - prev_velocity_commands_[i]));
    }
    

      //Skip serial write if nothing changed (only relevant during initial_read_)
    
    
    
    msg << std::fixed << std::setprecision(2) << " " << value_to_send;
    prev_velocity_commands_[i] = value_to_send * (M_PI / 180.0f);
  }

  if (initial_read_ && !any_change && initial_write_)
  {
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "No change in velocity commands, skipping write.");
    esp_.Write("rmp\n");
    return hardware_interface::return_type::OK;
  }
  

  msg << ">";
  std::string command = msg.str();

  

  try
  {
    esp_.Write(command + "\n");
    esp_.DrainWriteBuffer();
    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Sent: " << command);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("DaadbotInterface"), "Write failed: " << e.what());
    return hardware_interface::return_type::ERROR;
  }
  if (!initial_write_)
  {
    initial_write_ = true;
  }
  return hardware_interface::return_type::OK;
}

}  // namespace daadbot_controller

PLUGINLIB_EXPORT_CLASS(daadbot_controller::DaadbotInterface, hardware_interface::SystemInterface)