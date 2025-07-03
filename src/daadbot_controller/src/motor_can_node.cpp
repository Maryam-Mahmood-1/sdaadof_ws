#include <chrono>
#include <cstring>
#include <iostream>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <rclcpp/rclcpp.hpp>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

class RMDStatusListener : public rclcpp::Node {
public:
  RMDStatusListener()
      : Node("rmd_status_listener"), broadcast_id_(0x280),
        motor_ids_{0x241, 0x242, 0x243, 0x244, 0x245, 0x246, 0x247, 0x248},
        num_motors_(motor_ids_.size()) {

    socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (socket_fd_ < 0) {
      RCLCPP_FATAL(this->get_logger(), "Failed to create CAN socket");
      rclcpp::shutdown();
      return;
    }

    const char *can_interface = "can0";
    struct ifreq ifr {};
    std::strcpy(ifr.ifr_name, can_interface);
    ioctl(socket_fd_, SIOCGIFINDEX, &ifr);

    struct sockaddr_can addr {};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(socket_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
      RCLCPP_FATAL(this->get_logger(), "Failed to bind CAN socket");
      rclcpp::shutdown();
      return;
    }

    timer_ = this->create_wall_timer(1s, std::bind(&RMDStatusListener::pollMotors, this));
    RCLCPP_INFO(this->get_logger(), "RMD Status Listener Initialized");
  }

  ~RMDStatusListener() override {
    close(socket_fd_);
  }

private:
  int socket_fd_;
  rclcpp::TimerBase::SharedPtr timer_;
  uint32_t broadcast_id_;
  const std::vector<uint16_t> motor_ids_;
  const size_t num_motors_;

  struct MotorStatus {
    bool received = false;
    float iq = 0;
    int16_t speed = 0;
    int16_t angle = 0;
  };
  std::map<uint16_t, MotorStatus> status_map_;

  void pollMotors() {
    status_map_.clear();
    for (auto id : motor_ids_) {
      status_map_[id] = {};
    }

    sendBroadcastCommand();

    auto listen_start = Clock::now();
    listenForReplies(100);  // 100 ms
    auto listen_end = Clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(listen_end - listen_start).count();

    printMotorStatus();
    RCLCPP_INFO(this->get_logger(), "⏱ Listening took: %ld us", duration_us);
  }

  void sendBroadcastCommand() {
    struct can_frame frame {};
    frame.can_id = broadcast_id_;
    frame.can_dlc = 8;
    frame.data[0] = 0x9C;
    std::memset(&frame.data[1], 0x00, 7);

    auto t_start = Clock::now();
    if (write(socket_fd_, &frame, sizeof(frame)) != sizeof(frame)) {
      RCLCPP_WARN(this->get_logger(), "Failed to send broadcast 0x9C");
    } else {
      auto t_end = Clock::now();
      auto send_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
      RCLCPP_INFO(this->get_logger(), "✅ Sent 0x9C | ⏱ Send took: %ld us", send_us);
    }
  }

  void listenForReplies(uint16_t timeout_ms) {
    struct timeval timeout = {timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    fd_set read_fds;
    struct can_frame frame;

    size_t replies_received = 0;
    size_t total_expected = status_map_.size();

    while (replies_received < total_expected) {
      FD_ZERO(&read_fds);
      FD_SET(socket_fd_, &read_fds);

      int ret = select(socket_fd_ + 1, &read_fds, nullptr, nullptr, &timeout);
      if (ret <= 0) break;  // Timeout or error

      if (FD_ISSET(socket_fd_, &read_fds)) {
        ssize_t len = read(socket_fd_, &frame, sizeof(frame));
        if (len > 0 && frame.data[0] == 0x9C) {
          auto it = status_map_.find(frame.can_id);
          if (it != status_map_.end() && !it->second.received) {
            it->second.iq = ((int16_t)(frame.data[3] << 8 | frame.data[2])) * 0.01f;
            it->second.speed = (int16_t)(frame.data[5] << 8 | frame.data[4]);
            it->second.angle = (int16_t)(frame.data[7] << 8 | frame.data[6]);
            it->second.received = true;
            replies_received++;
          }
        }
      }
    }
  }


  void printMotorStatus() {
    for (const auto &[id, status] : status_map_) {
      if (status.received) {
        RCLCPP_INFO(this->get_logger(), "Motor 0x%X | Iq: %.2f A | Speed: %d dps | Angle: %d",
                    id, status.iq, status.speed, status.angle);
      } else {
        RCLCPP_WARN(this->get_logger(), "⚠️ No reply from motor ID: 0x%X", id);
      }
    }
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RMDStatusListener>());
  rclcpp::shutdown();
  return 0;
}



/******************************************************************************************************************************************************
*******************************************************************************************************************************************************
*******************************************************************************************************************************************************/

/*(ONGOING) A2 (consecutive send - consecutive receive) and 92 (broadcast) based */

// #include <chrono>
// #include <cstring>
// #include <iostream>
// #include <linux/can.h>
// #include <linux/can/raw.h>
// #include <net/if.h>
// #include <rclcpp/rclcpp.hpp>
// #include <sys/ioctl.h>
// #include <sys/socket.h>
// #include <unistd.h>

// using namespace std::chrono_literals;
// using Clock = std::chrono::steady_clock;

// struct MotorState {
//   float initial_angle;
//   float last_error;
//   float target_angle;
//   bool initial_read;
// };

// class PositionVelocityControlNode : public rclcpp::Node {
// public:
//   PositionVelocityControlNode()
//       : Node("multi_motor_control_node"), Kp_(1.0), Kd_(0.0), waiting_for_input_(true) {

//     motor_ids_ = {0x146, 0x147, 0x148};

//     for (uint32_t id : motor_ids_) {
//       MotorState s;
//       s.initial_angle = 0.0f;
//       s.last_error = 0.0f;
//       s.target_angle = 0.0f;
//       s.initial_read = false;
//       state_[id] = s;
//     }

//     const char *can_interface = "can0";
//     struct ifreq ifr {};
//     struct sockaddr_can addr {};

//     socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
//     if (socket_fd_ < 0) {
//       RCLCPP_FATAL(this->get_logger(), "Failed to create CAN socket");
//       rclcpp::shutdown();
//       return;
//     }

//     std::strcpy(ifr.ifr_name, can_interface);
//     ioctl(socket_fd_, SIOCGIFINDEX, &ifr);
//     addr.can_family = AF_CAN;
//     addr.can_ifindex = ifr.ifr_ifindex;

//     if (bind(socket_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
//       RCLCPP_FATAL(this->get_logger(), "Failed to bind CAN socket");
//       rclcpp::shutdown();
//       return;
//     }

//     timer_ = this->create_wall_timer(1ms, std::bind(&PositionVelocityControlNode::control_loop, this));
//   }

//   ~PositionVelocityControlNode() override {
//     close(socket_fd_);
//   }

// private:
//   int socket_fd_;
//   rclcpp::TimerBase::SharedPtr timer_;
//   std::vector<uint32_t> motor_ids_;
//   std::map<uint32_t, MotorState> state_;
//   float Kp_, Kd_;
//   bool waiting_for_input_;

//   void control_loop() {
//     if (waiting_for_input_) {
//       for (uint32_t id : motor_ids_) {
//         std::cout << "Enter target angle for motor " << std::hex << id << " (deg): ";
//         std::cin >> state_[id].target_angle;
//         state_[id].initial_read = false;
//       }
//       waiting_for_input_ = false;
//     }

//     send92Broadcast();

//     for (uint32_t id : motor_ids_) {
//       float current_angle;
//       if (!state_[id].initial_read) {
//         if (readReply92(id, current_angle)) {
//           state_[id].initial_angle = current_angle;
//           state_[id].initial_read = true;
//           RCLCPP_INFO(this->get_logger(), "Motor 0x%X initial angle: %.2f", id, current_angle);
//         } else {
//           RCLCPP_WARN(this->get_logger(), "Motor 0x%X failed to read initial angle", id);
//           return;
//         }
//       }

//       if (!readReply92(id, current_angle)) {
//         RCLCPP_WARN(this->get_logger(), "Motor 0x%X failed to read angle", id);
//         return;
//       }

//       float relative_angle = current_angle - state_[id].initial_angle;
//       float error = state_[id].target_angle - relative_angle;

//       if (std::abs(error) < 0.1f) {
//         sendA2Speed(id, 0);
//         continue;
//       }

//       float dt = 0.001f;
//       float dError = (error - state_[id].last_error) / dt;
//       state_[id].last_error = error;

//       float speed_dps = Kp_ * error + Kd_ * dError;
//       speed_dps = std::clamp(speed_dps, -100.0f, 100.0f);
//       if (std::abs(speed_dps) < 2.0f && std::abs(error) > 0.1f)
//         speed_dps = (speed_dps > 0) ? 2.0f : -2.0f;

//       int32_t speed_val = static_cast<int32_t>(speed_dps * 100.0f);
//       sendA2Speed(id, speed_val);

//       RCLCPP_INFO(this->get_logger(),
//         "0x%X | Init: %.2f | Curr: %.2f | Rel: %.2f | Err: %.2f | Spd: %.2f dps",
//         id, state_[id].initial_angle, current_angle, relative_angle, error, speed_dps);
//     }
//   }

//   void sendA2Speed(uint32_t id, int32_t speed) {
//     struct can_frame frame {};
//     frame.can_id = id;
//     frame.can_dlc = 8;
//     frame.data[0] = 0xA2;
//     frame.data[1] = frame.data[2] = frame.data[3] = 0x00;
//     frame.data[4] = speed & 0xFF;
//     frame.data[5] = (speed >> 8) & 0xFF;
//     frame.data[6] = (speed >> 16) & 0xFF;
//     frame.data[7] = (speed >> 24) & 0xFF;

//     write(socket_fd_, &frame, sizeof(frame));

//     // wait for reply
//     struct can_frame reply;
//     struct timeval timeout = {0, 10000};  // 10ms
//     fd_set read_set;
//     FD_ZERO(&read_set);
//     FD_SET(socket_fd_, &read_set);

//     int ret = select(socket_fd_ + 1, &read_set, nullptr, nullptr, &timeout);
//     if (ret > 0 && FD_ISSET(socket_fd_, &read_set)) {
//       read(socket_fd_, &reply, sizeof(reply));
//       if (reply.can_id == (id + 0x100) && reply.data[0] == 0xA2) {
//         RCLCPP_DEBUG(this->get_logger(), "Motor 0x%X A2 acknowledged", id);
//       }
//     }
//   }

//   void send92Broadcast() {
//     struct can_frame frame {};
//     frame.can_id = 0x280;
//     frame.can_dlc = 8;
//     frame.data[0] = 0x92;
//     write(socket_fd_, &frame, sizeof(frame));
//   }

//   bool readReply92(uint32_t expected_id, float &angle_out) {
//     struct can_frame frame;
//     struct timeval timeout = {0, 10000};  // 10ms
//     fd_set read_set;
//     FD_ZERO(&read_set);
//     FD_SET(socket_fd_, &read_set);

//     int ret = select(socket_fd_ + 1, &read_set, nullptr, nullptr, &timeout);
//     if (ret > 0 && FD_ISSET(socket_fd_, &read_set)) {
//       ssize_t nbytes = read(socket_fd_, &frame, sizeof(frame));
//       if (nbytes > 0 && frame.data[0] == 0x92 && frame.can_id == (expected_id + 0x100)) {
//         int32_t raw_angle;
//         std::memcpy(&raw_angle, &frame.data[4], 4);
//         angle_out = raw_angle * 0.01f;
//         return true;
//       }
//     }
//     return false;
//   }
// };

// int main(int argc, char *argv[]) {
//   rclcpp::init(argc, argv);
//   rclcpp::spin(std::make_shared<PositionVelocityControlNode>());
//   rclcpp::shutdown();
//   return 0;
// }




/******************************************************************************************************************************************************
*******************************************************************************************************************************************************
*******************************************************************************************************************************************************/

/*(ONGOING) Checking the right amounts of delay - all reads + all writes for 9C. (4 MOTORS)*/

// #include <chrono>
// #include <cstring>
// #include <iostream>
// #include <linux/can.h>
// #include <linux/can/raw.h>
// #include <map>
// #include <net/if.h>
// #include <rclcpp/rclcpp.hpp>
// #include <sys/ioctl.h>
// #include <sys/socket.h>
// #include <thread>
// #include <unistd.h>

// using namespace std::chrono_literals;
// using Clock = std::chrono::steady_clock;

// class RMDStatusQueryNode : public rclcpp::Node {
// public:
//   RMDStatusQueryNode()
//       : Node("rmd_status_query_node"),
//         command_ids_{0x145, 0x146, 0x147, 0x148},
//         num_motors_(command_ids_.size()) {

//     socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
//     if (socket_fd_ < 0) {
//       RCLCPP_FATAL(this->get_logger(), "❌ Failed to create CAN socket");
//       rclcpp::shutdown();
//       return;
//     }

//     const char *can_interface = "can0";
//     struct ifreq ifr {};
//     std::strcpy(ifr.ifr_name, can_interface);
//     ioctl(socket_fd_, SIOCGIFINDEX, &ifr);

//     struct sockaddr_can addr {};
//     addr.can_family = AF_CAN;
//     addr.can_ifindex = ifr.ifr_ifindex;
//     if (bind(socket_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
//       RCLCPP_FATAL(this->get_logger(), "❌ Failed to bind CAN socket");
//       rclcpp::shutdown();
//       return;
//     }

//     timer_ = this->create_wall_timer(1s, std::bind(&RMDStatusQueryNode::pollMotors, this));
//     RCLCPP_INFO(this->get_logger(), "✅ RMD Status Query Node Initialized");
//   }

//   ~RMDStatusQueryNode() override {
//     close(socket_fd_);
//   }

// private:
//   int socket_fd_;
//   rclcpp::TimerBase::SharedPtr timer_;
//   const std::vector<uint16_t> command_ids_;
//   const size_t num_motors_;

//   struct MotorStatus {
//     bool received = false;
//     float iq = 0.0;
//     int16_t speed = 0;
//     int16_t angle = 0;
//   };
//   std::map<uint16_t, MotorStatus> status_map_;  // keyed by reply ID (0x2XX)

//   void pollMotors() {
//     status_map_.clear();
//     for (auto cmd_id : command_ids_) {
//       status_map_[cmd_id + 0x100] = {};
//     }

//     auto t_send_start = Clock::now();
//     for (const auto &cmd_id : command_ids_) {
//       send9CCommand(cmd_id);
//       // std::this_thread::sleep_for(std::chrono::microseconds(100));
//     }
//     auto t_send_end = Clock::now();

//     auto t_recv_start = Clock::now();
//     listenForReplies(100);
//     auto t_recv_end = Clock::now();

//     auto t_print_start = Clock::now();
//     printMotorStatus();
//     auto t_print_end = Clock::now();

//     RCLCPP_INFO(this->get_logger(),
//       "⏱ Send: %ld us | Receive: %ld us | Print: %ld us | Total: %ld us",
//       std::chrono::duration_cast<std::chrono::microseconds>(t_send_end - t_send_start).count(),
//       std::chrono::duration_cast<std::chrono::microseconds>(t_recv_end - t_recv_start).count(),
//       std::chrono::duration_cast<std::chrono::microseconds>(t_print_end - t_print_start).count(),
//       std::chrono::duration_cast<std::chrono::microseconds>(t_print_end - t_send_start).count());
//   }

//   void send9CCommand(uint16_t id) {
//     struct can_frame frame {};
//     frame.can_id = id;
//     frame.can_dlc = 8;
//     frame.data[0] = 0x9C;
//     std::memset(&frame.data[1], 0x00, 7);

//     auto t_start = Clock::now();
//     ssize_t sent = write(socket_fd_, &frame, sizeof(frame));
//     auto t_end = Clock::now();

//     auto send_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
//     auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(t_start.time_since_epoch()).count();

//     if (sent != sizeof(frame)) {
//       RCLCPP_WARN(this->get_logger(), "❌ Failed to send 0x9C to 0x%X", id);
//     } else {
//       RCLCPP_INFO(this->get_logger(), "[%ld us] ✅ Sent 0x9C to 0x%X | Took: %ld us", timestamp, id, send_us);
//     }
//   }


//   void listenForReplies(uint16_t timeout_ms) {
//     struct can_frame frame;
//     fd_set read_fds;
//     size_t replies = 0;
//     size_t expected = status_map_.size();

//     struct timeval timeout = {timeout_ms / 1000, (timeout_ms % 1000) * 1000};
//     FD_ZERO(&read_fds);
//     FD_SET(socket_fd_, &read_fds);
//     int ret = select(socket_fd_ + 1, &read_fds, nullptr, nullptr, &timeout);
//     if (ret <= 0) return;

//     while (replies < expected) {
//       FD_ZERO(&read_fds);
//       FD_SET(socket_fd_, &read_fds);
//       struct timeval short_timeout = {0, 10000}; // 10ms
//       int ret = select(socket_fd_ + 1, &read_fds, nullptr, nullptr, &short_timeout);
//       if (ret <= 0) break;

//       if (FD_ISSET(socket_fd_, &read_fds)) {
//         ssize_t len = read(socket_fd_, &frame, sizeof(frame));
//         if (len > 0 && frame.data[0] == 0x9C) {
//           auto it = status_map_.find(frame.can_id);
//           if (it != status_map_.end() && !it->second.received) {
//             it->second.iq = ((int16_t)(frame.data[3] << 8 | frame.data[2])) * 0.01f;
//             it->second.speed = (int16_t)(frame.data[5] << 8 | frame.data[4]);
//             it->second.angle = (int16_t)(frame.data[7] << 8 | frame.data[6]);
//             it->second.received = true;
//             replies++;
//           }
//         }
//       }
//     }
//   }

//   void printMotorStatus() {
//     std::vector<std::string> ok_ids, missing_ids;

//     for (const auto &[reply_id, status] : status_map_) {
//       if (status.received) {
//         RCLCPP_INFO(this->get_logger(), "Motor 0x%X | Iq: %.2f A | Speed: %d dps | Angle: %d",
//                     reply_id, status.iq, status.speed, status.angle);
//         ok_ids.push_back("0x" + intToHex(reply_id));
//       } else {
//         RCLCPP_WARN(this->get_logger(), "⚠️ No reply from motor ID: 0x%X", reply_id);
//         missing_ids.push_back("0x" + intToHex(reply_id));
//       }
//     }

//     if (!ok_ids.empty()) {
//       RCLCPP_INFO(this->get_logger(), "✅ Replied this cycle: [%s]", join(ok_ids).c_str());
//     }
//     if (!missing_ids.empty()) {
//       RCLCPP_WARN(this->get_logger(), "❌ Missing this cycle: [%s]", join(missing_ids).c_str());
//     }
//   }

//   std::string intToHex(uint16_t val) {
//     std::stringstream ss;
//     ss << std::hex << std::uppercase << val;
//     return ss.str();
//   }

//   std::string join(const std::vector<std::string>& vec) {
//     std::ostringstream oss;
//     for (size_t i = 0; i < vec.size(); ++i) {
//       oss << vec[i];
//       if (i != vec.size() - 1) oss << ", ";
//     }
//     return oss.str();
//   }
// };

// int main(int argc, char *argv[]) {
//   rclcpp::init(argc, argv);
//   rclcpp::spin(std::make_shared<RMDStatusQueryNode>());
//   rclcpp::shutdown();
//   return 0;
// }


/******************************************************************************************************************************************************
*******************************************************************************************************************************************************
*******************************************************************************************************************************************************/

/*Checking the multicast command - all reads + all writes for 9C. (ALL MOTORS)*/

// #include <chrono>
// #include <cstring>
// #include <iostream>
// #include <linux/can.h>
// #include <linux/can/raw.h>
// #include <net/if.h>
// #include <rclcpp/rclcpp.hpp>
// #include <sys/ioctl.h>
// #include <sys/socket.h>
// #include <unistd.h>

// using namespace std::chrono_literals;
// using Clock = std::chrono::steady_clock;

// class RMDStatusListener : public rclcpp::Node {
// public:
//   RMDStatusListener()
//       : Node("rmd_status_listener"), broadcast_id_(0x280),
//         motor_ids_{0x241, 0x242, 0x243, 0x244, 0x245, 0x246, 0x247, 0x248},
//         num_motors_(motor_ids_.size()) {

//     socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
//     if (socket_fd_ < 0) {
//       RCLCPP_FATAL(this->get_logger(), "Failed to create CAN socket");
//       rclcpp::shutdown();
//       return;
//     }

//     const char *can_interface = "can0";
//     struct ifreq ifr {};
//     std::strcpy(ifr.ifr_name, can_interface);
//     ioctl(socket_fd_, SIOCGIFINDEX, &ifr);

//     struct sockaddr_can addr {};
//     addr.can_family = AF_CAN;
//     addr.can_ifindex = ifr.ifr_ifindex;
//     if (bind(socket_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
//       RCLCPP_FATAL(this->get_logger(), "Failed to bind CAN socket");
//       rclcpp::shutdown();
//       return;
//     }

//     timer_ = this->create_wall_timer(1s, std::bind(&RMDStatusListener::pollMotors, this));
//     RCLCPP_INFO(this->get_logger(), "RMD Status Listener Initialized");
//   }

//   ~RMDStatusListener() override {
//     close(socket_fd_);
//   }

// private:
//   int socket_fd_;
//   rclcpp::TimerBase::SharedPtr timer_;
//   uint32_t broadcast_id_;
//   const std::vector<uint16_t> motor_ids_;
//   const size_t num_motors_;

//   struct MotorStatus {
//     bool received = false;
//     float iq = 0;
//     int16_t speed = 0;
//     int16_t angle = 0;
//   };
//   std::map<uint16_t, MotorStatus> status_map_;

//   void pollMotors() {
//     status_map_.clear();
//     for (auto id : motor_ids_) {
//       status_map_[id] = {};
//     }

//     sendBroadcastCommand();

//     auto listen_start = Clock::now();
//     listenForReplies(100);  // 100 ms
//     auto listen_end = Clock::now();
//     auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(listen_end - listen_start).count();

//     printMotorStatus();
//     RCLCPP_INFO(this->get_logger(), "⏱ Listening took: %ld us", duration_us);
//   }

//   void sendBroadcastCommand() {
//     struct can_frame frame {};
//     frame.can_id = broadcast_id_;
//     frame.can_dlc = 8;
//     frame.data[0] = 0x9C;
//     std::memset(&frame.data[1], 0x00, 7);

//     auto t_start = Clock::now();
//     if (write(socket_fd_, &frame, sizeof(frame)) != sizeof(frame)) {
//       RCLCPP_WARN(this->get_logger(), "Failed to send broadcast 0x9C");
//     } else {
//       auto t_end = Clock::now();
//       auto send_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
//       RCLCPP_INFO(this->get_logger(), "✅ Sent 0x9C | ⏱ Send took: %ld us", send_us);
//     }
//   }

//   void listenForReplies(uint16_t timeout_ms) {
//     struct timeval timeout = {timeout_ms / 1000, (timeout_ms % 1000) * 1000};
//     fd_set read_fds;
//     struct can_frame frame;

//     size_t replies_received = 0;
//     size_t total_expected = status_map_.size();

//     while (replies_received < total_expected) {
//       FD_ZERO(&read_fds);
//       FD_SET(socket_fd_, &read_fds);

//       int ret = select(socket_fd_ + 1, &read_fds, nullptr, nullptr, &timeout);
//       if (ret <= 0) break;  // Timeout or error

//       if (FD_ISSET(socket_fd_, &read_fds)) {
//         ssize_t len = read(socket_fd_, &frame, sizeof(frame));
//         if (len > 0 && frame.data[0] == 0x9C) {
//           auto it = status_map_.find(frame.can_id);
//           if (it != status_map_.end() && !it->second.received) {
//             it->second.iq = ((int16_t)(frame.data[3] << 8 | frame.data[2])) * 0.01f;
//             it->second.speed = (int16_t)(frame.data[5] << 8 | frame.data[4]);
//             it->second.angle = (int16_t)(frame.data[7] << 8 | frame.data[6]);
//             it->second.received = true;
//             replies_received++;
//           }
//         }
//       }
//     }
//   }


//   void printMotorStatus() {
//     for (const auto &[id, status] : status_map_) {
//       if (status.received) {
//         RCLCPP_INFO(this->get_logger(), "Motor 0x%X | Iq: %.2f A | Speed: %d dps | Angle: %d",
//                     id, status.iq, status.speed, status.angle);
//       } else {
//         RCLCPP_WARN(this->get_logger(), "⚠️ No reply from motor ID: 0x%X", id);
//       }
//     }
//   }
// };

// int main(int argc, char *argv[]) {
//   rclcpp::init(argc, argv);
//   rclcpp::spin(std::make_shared<RMDStatusListener>());
//   rclcpp::shutdown();
//   return 0;
// }



/******************************************************************************************************************************************************
*******************************************************************************************************************************************************
*******************************************************************************************************************************************************/

/*A2 and 92 based velocity + position control.*/

// #include <chrono>
// #include <cstring>
// #include <iostream>
// #include <linux/can.h>
// #include <linux/can/raw.h>
// #include <net/if.h>
// #include <rclcpp/rclcpp.hpp>
// #include <sys/ioctl.h>
// #include <sys/socket.h>
// #include <unistd.h>
// #include <termios.h>

// using namespace std::chrono_literals;
// using Clock = std::chrono::steady_clock;

// class PositionVelocityControlNode : public rclcpp::Node {
// public:
//   PositionVelocityControlNode()
//       : Node("position_velocity_control_node"),
//         motor_id_(0x148), Kp_(1.0), Kd_(0.0), last_error_(0.0),
//         initial_angle_read_(false), waiting_for_input_(true),
//         target_angle_deg_(0.0) {

//     const char *can_interface = "can0";
//     struct ifreq ifr {};
//     struct sockaddr_can addr {};

//     socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
//     if (socket_fd_ < 0) {
//       RCLCPP_FATAL(this->get_logger(), "Failed to create CAN socket");
//       rclcpp::shutdown();
//       return;
//     }

//     std::strcpy(ifr.ifr_name, can_interface);
//     ioctl(socket_fd_, SIOCGIFINDEX, &ifr);
//     addr.can_family = AF_CAN;
//     addr.can_ifindex = ifr.ifr_ifindex;

//     if (bind(socket_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
//       RCLCPP_FATAL(this->get_logger(), "Failed to bind CAN socket");
//       rclcpp::shutdown();
//       return;
//     }

//     timer_ = this->create_wall_timer(1ms, std::bind(&PositionVelocityControlNode::control_loop, this));
//     last_loop_time_ = Clock::now();
//   }

//   ~PositionVelocityControlNode() override {
//     close(socket_fd_);
//   }

// private:
//   int socket_fd_;
//   rclcpp::TimerBase::SharedPtr timer_;
//   uint32_t motor_id_;
//   float Kp_, Kd_;
//   float last_error_;
//   float initial_angle_deg_;
//   bool initial_angle_read_;
//   bool waiting_for_input_;
//   float target_angle_deg_;
//   Clock::time_point last_loop_time_;

//   void control_loop() {
//     auto loop_start = Clock::now();

//     if (waiting_for_input_) {
//       std::cout << "Enter target angle in degrees: ";
//       std::cin >> target_angle_deg_;
//       initial_angle_read_ = false;
//       waiting_for_input_ = false;
//     }

//     float current_angle_deg;
//     if (!initial_angle_read_) {
//       if (readMultiTurnAngle(current_angle_deg)) {
//         initial_angle_deg_ = current_angle_deg;
//         initial_angle_read_ = true;
//         RCLCPP_INFO(this->get_logger(), "Initial Angle: %.2f deg", initial_angle_deg_);
//       } else {
//         RCLCPP_WARN(this->get_logger(), "Failed to read initial angle. Retrying...");
//         return;
//       }
//     }

//     if (!readMultiTurnAngle(current_angle_deg)) {
//       RCLCPP_WARN(this->get_logger(), "Failed to read current angle");
//       return;
//     }

//     float relative_angle_deg = current_angle_deg - initial_angle_deg_;
//     float error = target_angle_deg_ - relative_angle_deg;

//     if (std::abs(error) < 0.1f) {
//       sendA2SpeedCommand(0);
//       waiting_for_input_ = true;
//       return;
//     }

//     float dt = 0.001f;  // 1 ms loop
//     float dError = (error - last_error_) / dt;
//     last_error_ = error;

//     float control_speed_dps = Kp_ * error + Kd_ * dError;
//     control_speed_dps = std::clamp(control_speed_dps, -100.0f, 100.0f);

//     if (std::abs(control_speed_dps) < 2.0f && std::abs(error) > 0.1f) {
//       control_speed_dps = (control_speed_dps > 0) ? 2.0f : -2.0f;
//     }

//     int32_t speed_val = static_cast<int32_t>(control_speed_dps * 100.0f);
//     sendA2SpeedCommand(speed_val);

//     auto loop_end = Clock::now();
//     auto loop_us = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start).count();

//     RCLCPP_INFO(this->get_logger(),
//       "Init: %.2f | Curr: %.2f | Rel: %.2f | Err: %.2f | Spd: %.2f dps | Loop: %ld us",
//       initial_angle_deg_, current_angle_deg, relative_angle_deg, error, control_speed_dps, loop_us);
//   }

//   void sendA2SpeedCommand(int32_t speedControl) {
//     auto t_start = Clock::now();

//     struct can_frame frame {};
//     frame.can_id = motor_id_;
//     frame.can_dlc = 8;
//     frame.data[0] = 0xA2;
//     frame.data[1] = frame.data[2] = frame.data[3] = 0x00;
//     frame.data[4] = static_cast<uint8_t>(speedControl);
//     frame.data[5] = static_cast<uint8_t>(speedControl >> 8);
//     frame.data[6] = static_cast<uint8_t>(speedControl >> 16);
//     frame.data[7] = static_cast<uint8_t>(speedControl >> 24);

//     auto write_start = Clock::now();
//     ssize_t sent = write(socket_fd_, &frame, sizeof(frame));
//     auto write_end = Clock::now();

//     if (sent < 0) {
//       RCLCPP_WARN(this->get_logger(), "Failed to send A2 command");
//       return;
//     }

//     struct timeval timeout = {0, 10000};  // 10ms
//     fd_set read_set;
//     FD_ZERO(&read_set);
//     FD_SET(socket_fd_, &read_set);

//     struct can_frame reply;
//     auto read_wait_start = Clock::now();
//     int ret = select(socket_fd_ + 1, &read_set, nullptr, nullptr, &timeout);
//     auto read_wait_end = Clock::now();

//     if (ret > 0 && FD_ISSET(socket_fd_, &read_set)) {
//       ssize_t nbytes = read(socket_fd_, &reply, sizeof(reply));
//       if (nbytes > 0 && reply.can_id == (motor_id_ + 0x100) && reply.data[0] == 0xA2) {
//         auto t_end = Clock::now();
//         auto write_us = std::chrono::duration_cast<std::chrono::microseconds>(write_end - write_start).count();
//         auto wait_us = std::chrono::duration_cast<std::chrono::microseconds>(read_wait_end - read_wait_start).count();
//         auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();

//         RCLCPP_DEBUG(this->get_logger(), "A2 Write: %ld us | Wait: %ld us | Total: %ld us", write_us, wait_us, total_us);
//         return;
//       } else {
//         RCLCPP_WARN(this->get_logger(), "Invalid A2 reply");
//       }
//     } else {
//       RCLCPP_WARN(this->get_logger(), "No A2 reply within timeout");
//     }
//   }

//   bool readMultiTurnAngle(float &angle_deg_out) {
//     struct can_frame cmd = {};
//     cmd.can_id = motor_id_;
//     cmd.can_dlc = 8;
//     cmd.data[0] = 0x92;

//     auto write_start = Clock::now();
//     write(socket_fd_, &cmd, sizeof(cmd));
//     auto write_end = Clock::now();

//     struct can_frame reply;
//     struct timeval timeout = {0, 10000};  // 10ms
//     fd_set read_set;
//     FD_ZERO(&read_set);
//     FD_SET(socket_fd_, &read_set);

//     auto read_start = Clock::now();
//     int ret = select(socket_fd_ + 1, &read_set, nullptr, nullptr, &timeout);
//     auto read_end = Clock::now();

//     if (ret > 0 && FD_ISSET(socket_fd_, &read_set)) {
//       ssize_t bytes = read(socket_fd_, &reply, sizeof(reply));
//       if (bytes > 0 && reply.data[0] == 0x92 && reply.can_id == (motor_id_ + 0x100)) {
//         int32_t angle_raw;
//         std::memcpy(&angle_raw, &reply.data[4], sizeof(angle_raw));
//         angle_deg_out = angle_raw * 0.01f;

//         auto write_us = std::chrono::duration_cast<std::chrono::microseconds>(write_end - write_start).count();
//         auto read_us = std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start).count();
//         RCLCPP_DEBUG(this->get_logger(), "Read 92 Write: %ld us | Wait: %ld us", write_us, read_us);
//         return true;
//       }
//     }
//     return false;
//   }
// };

// int main(int argc, char *argv[]) {
//   rclcpp::init(argc, argv);
//   rclcpp::spin(std::make_shared<PositionVelocityControlNode>());
//   rclcpp::shutdown();
//   return 0;
// }




/******************************************************************************************************************************************************
*******************************************************************************************************************************************************
*******************************************************************************************************************************************************/

/*A6 command angle read resolution - 1 degrees verified.*/

// #include <chrono>
// #include <cstring>
// #include <iostream>
// #include <linux/can.h>
// #include <linux/can/raw.h>
// #include <net/if.h>
// #include <rclcpp/rclcpp.hpp>
// #include <sys/ioctl.h>
// #include <sys/socket.h>
// #include <unistd.h>

// using namespace std::chrono_literals;

// class MotorCANNode : public rclcpp::Node {
// public:
//   MotorCANNode()
//       : Node("motor_can_node"), motor_id_(0x146), max_speed_dps_(0), target_angle_(0), spin_direction_(0x00) {
//     // Setup CAN socket
//     const char *can_interface = "can0";
//     struct ifreq ifr {};
//     struct sockaddr_can addr {};

//     socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
//     if (socket_fd_ < 0) {
//       RCLCPP_FATAL(this->get_logger(), "Failed to create CAN socket");
//       rclcpp::shutdown();
//       return;
//     }

//     std::strcpy(ifr.ifr_name, can_interface);
//     ioctl(socket_fd_, SIOCGIFINDEX, &ifr);

//     addr.can_family = AF_CAN;
//     addr.can_ifindex = ifr.ifr_ifindex;

//     if (bind(socket_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
//       RCLCPP_FATAL(this->get_logger(), "Failed to bind CAN socket");
//       rclcpp::shutdown();
//       return;
//     }

//     // Periodic command
//     timer_ = this->create_wall_timer(500ms, std::bind(&MotorCANNode::send_position_command, this));
//   }

//   ~MotorCANNode() override { close(socket_fd_); }

// private:
//   int socket_fd_;
//   rclcpp::TimerBase::SharedPtr timer_;
//   uint32_t motor_id_;
//   uint16_t max_speed_dps_;
//   uint16_t target_angle_;     // 0~35999 (0.01°/LSB)
//   uint8_t spin_direction_;    // 0x00 = CW, 0x01 = CCW

//   void send_position_command() {
//     struct can_frame frame {};
//     frame.can_id = motor_id_;
//     frame.can_dlc = 8;
//     frame.data[0] = 0xA6;  // Single-turn position command
//     frame.data[1] = spin_direction_;
//     frame.data[2] = static_cast<uint8_t>(max_speed_dps_ & 0xFF);
//     frame.data[3] = static_cast<uint8_t>(max_speed_dps_ >> 8);
//     frame.data[4] = static_cast<uint8_t>(target_angle_ & 0xFF);
//     frame.data[5] = static_cast<uint8_t>(target_angle_ >> 8);
//     frame.data[6] = 0x00;
//     frame.data[7] = 0x00;

//     auto send_time = std::chrono::steady_clock::now();

//     if (write(socket_fd_, &frame, sizeof(frame)) != sizeof(frame)) {
//       RCLCPP_ERROR(this->get_logger(), "Failed to send CAN frame");
//       return;
//     } else {
//       RCLCPP_INFO(this->get_logger(), "Sent 0xA6 Single-Turn Command to 0° at max speed %d dps", max_speed_dps_);
//     }

//     // Await reply
//     struct can_frame rx_frame;
//     struct timeval timeout = {0, 100000}; // 100ms timeout
//     fd_set read_set;
//     FD_ZERO(&read_set);
//     FD_SET(socket_fd_, &read_set);

//     int ret = select(socket_fd_ + 1, &read_set, nullptr, nullptr, &timeout);
//     if (ret > 0 && FD_ISSET(socket_fd_, &read_set)) {
//       ssize_t nbytes = read(socket_fd_, &rx_frame, sizeof(rx_frame));
//       if (nbytes > 0 && rx_frame.can_id == (motor_id_ + 0x100)) {
//         auto receive_time = std::chrono::steady_clock::now();
//         auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(receive_time - send_time).count();

//         int8_t temperature = static_cast<int8_t>(rx_frame.data[1]);
//         int16_t current = static_cast<int16_t>((rx_frame.data[3] << 8) | rx_frame.data[2]);
//         int16_t speed = static_cast<int16_t>((rx_frame.data[5] << 8) | rx_frame.data[4]);
//         uint16_t encoder = static_cast<uint16_t>((rx_frame.data[7] << 8) | rx_frame.data[6]);

//         RCLCPP_INFO(this->get_logger(),
//                     "Temp: %d°C | Current: %.2f A | Speed: %d dps | Encoder: %u | Reply time: %ld µs",
//                     temperature, current * 0.01f, speed, encoder, elapsed_us);
//       }
//     } else {
//       RCLCPP_WARN(this->get_logger(), "No CAN reply received");
//     }
//   }
// };

// int main(int argc, char *argv[]) {
//   rclcpp::init(argc, argv);
//   rclcpp::spin(std::make_shared<MotorCANNode>());
//   rclcpp::shutdown();
//   return 0;
// }

