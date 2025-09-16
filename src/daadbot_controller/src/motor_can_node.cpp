#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <rclcpp/rclcpp.hpp>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include <map>
#include <stdexcept>

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

class A4MultiMotorControlNode : public rclcpp::Node {
public:
  A4MultiMotorControlNode()
  : Node("a4_multi_motor_control_node"),
    // Order here matches declaration order below to silence -Wreorder
    motor_ids_({0x142, 0x144, 0x145, 0x146, 0x147, 0x148}),
    current_motor_idx_(0),
    cycle_idx_(0),
    csv_file_("a4_multimotor_log.csv")
  {
    RCLCPP_INFO(this->get_logger(), "=== A4MultiMotorControlNode starting ===");

    // ---- Open CAN socket on can0 ----
    socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (socket_fd_ < 0) {
      throw std::runtime_error("Failed to create CAN socket");
    }

    struct ifreq ifr;
    std::memset(&ifr, 0, sizeof(ifr));
    std::strncpy(ifr.ifr_name, "can0", IFNAMSIZ - 1);

    if (ioctl(socket_fd_, SIOCGIFINDEX, &ifr) < 0) {
      close(socket_fd_);
      throw std::runtime_error("Failed to get index for interface 'can0' (is it up?)");
    }

    struct sockaddr_can addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if (bind(socket_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
      close(socket_fd_);
      throw std::runtime_error("Failed to bind CAN socket to 'can0'");
    }

    RCLCPP_INFO(this->get_logger(), "Bound CAN socket to 'can0' (ifindex=%d)", ifr.ifr_ifindex);

    // ---- Flush any stale frames (non-blocking via select) ----
    flushCANBuffer();
    RCLCPP_INFO(this->get_logger(), "Flushed CAN RX buffer");

    // ---- Read initial angles w/ retries ----
    for (auto id : motor_ids_) {
      float init_angle = 0.0f;
      if (!readMultiTurnAngleWithRetry(id, init_angle, 5)) {
        close(socket_fd_);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "Failed to read angle for motor ID: 0x%X", id);
        throw std::runtime_error(buf);
      }
      initial_angles_[id] = init_angle;
      last_angle_deg_[id] = init_angle;
      RCLCPP_INFO(this->get_logger(), "Motor 0x%X initial angle: %.2f deg", id, init_angle);
    }

    // ---- Fixed relative offsets (deg) per motor ----
    fixed_offsets_ = {
      {0x142,  0.0},
      {0x144,  0.0},
      {0x145, 0.0},
      {0x146,  0.0},
      {0x147,  20.0},
      {0x148, -60.0}
    };

    // ---- Compute absolute target angles (centi-deg) ----
    for (auto id : motor_ids_) {
      target_angles_[id] = static_cast<int32_t>((initial_angles_[id] + fixed_offsets_[id]) * 100.0f);
      executed_[id] = false;
      last_cmd_time_[id] = Clock::now() - 1s; // allow immediate send
    }

    // ---- CSV header ----
    csv_file_ << "motor_id,timestamp_us,write_us,read_us,reply_latency_us,full_cycle_us,"
                 "temperature_C,iq_A,speed_dps,angle_deg\n";

    // ---- Control loop ----
    timer_ = this->create_wall_timer(10ms, std::bind(&A4MultiMotorControlNode::control_loop, this));
    RCLCPP_INFO(this->get_logger(), "Control loop started at 100 Hz");

    RCLCPP_INFO(this->get_logger(), "=== Node initialized OK ===");
  }

  ~A4MultiMotorControlNode() override {
    if (socket_fd_ >= 0) close(socket_fd_);
    if (csv_file_.is_open()) csv_file_.close();
  }

private:
  // Declaration order intentionally matches constructor init list to avoid -Wreorder warnings
  int socket_fd_{-1};
  std::vector<uint32_t> motor_ids_;
  size_t current_motor_idx_;
  size_t cycle_idx_;
  std::ofstream csv_file_;

  std::map<uint32_t, float> initial_angles_;
  std::map<uint32_t, float> last_angle_deg_;
  std::map<uint32_t, int32_t> target_angles_;
  std::map<uint32_t, double> fixed_offsets_;
  std::map<uint32_t, bool> executed_;
  std::map<uint32_t, Clock::time_point> last_cmd_time_;

  rclcpp::TimerBase::SharedPtr timer_;

  void flushCANBuffer() {
    struct can_frame frame;
    while (true) {
      fd_set read_set;
      FD_ZERO(&read_set);
      FD_SET(socket_fd_, &read_set);
      struct timeval tv {0, 0}; // zero-timeout: poll
      int ret = select(socket_fd_ + 1, &read_set, nullptr, nullptr, &tv);
      if (ret > 0 && FD_ISSET(socket_fd_, &read_set)) {
        ssize_t n = read(socket_fd_, &frame, sizeof(frame));
        if (n <= 0) break;
        // discarded
      } else {
        break;
      }
    }
  }

  void control_loop() {
    static uint64_t tick_counter = 0;
    tick_counter++;
    if (tick_counter % 50 == 0) {
      RCLCPP_INFO(this->get_logger(), "[tick %lu] Loop alive. Current motor index: %zu",
                  static_cast<unsigned long>(tick_counter), current_motor_idx_);
    }

    if (motor_ids_.empty()) return;

    uint32_t id = motor_ids_[current_motor_idx_];

    const uint16_t max_speed_limit =
        (id == 0x142) ? 300 : (id == 0x148 ? 75 : 50);

    const double target_deg = target_angles_[id] / 100.0;
    const double current_angle_deg = last_angle_deg_[id];
    const double error = target_deg - current_angle_deg;
    const double threshold = 1.0; // deg

    // (Re)send command at most every 100 ms until reached
    const auto now = Clock::now();
    if (!executed_[id] && (now - last_cmd_time_[id] >= 100ms)) {
      sendA4PositionCommand(id, target_angles_[id], max_speed_limit);
      last_cmd_time_[id] = now;
      RCLCPP_INFO(this->get_logger(),
        "CMD -> Motor 0x%X | target: %.2f deg | current: %.2f deg | err: %.2f deg | vmax: %u",
        id, target_deg, current_angle_deg, error, max_speed_limit);
    }

    // Wait up to 20 ms for any reply frame
    struct can_frame reply;
    struct timeval timeout = {0, 20000};
    fd_set read_set;
    FD_ZERO(&read_set);
    FD_SET(socket_fd_, &read_set);

    const auto t_before = Clock::now();
    int sel = select(socket_fd_ + 1, &read_set, nullptr, nullptr, &timeout);
    const auto t_after = Clock::now();

    if (sel < 0) {
      RCLCPP_WARN(this->get_logger(), "select() error; will continue next tick");
      return;
    }
    if (sel == 0) {
      // timeout
      RCLCPP_INFO(this->get_logger(),
        "WAIT -> No CAN reply for motor 0x%X in %ld us (target %.2f | curr %.2f | err %.2f)",
        id,
        static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(t_after - t_before).count()),
        target_deg, current_angle_deg, error);
      return;
    }

    // We have at least one frame
    ssize_t nbytes = read(socket_fd_, &reply, sizeof(reply));
    if (nbytes <= 0) {
      RCLCPP_WARN(this->get_logger(), "read() returned %zd; ignoring", nbytes);
      return;
    }

    // Print every received frame so you see bus activity
    RCLCPP_INFO(this->get_logger(), "RX <- can_id=0x%X dlc=%d data[0]=0x%02X",
                reply.can_id, reply.can_dlc, reply.data[0]);

    // We expect feedback from <id + 0x100> with data[0] == 0xA4
    const uint32_t expected_id = id + 0x100;
    if (reply.can_id != expected_id || reply.data[0] != 0xA4) {
      // Not the frame we are looking for; keep it visible for debugging
      RCLCPP_INFO(this->get_logger(),
                  "IGN -> Frame not for current motor: expect can_id=0x%X 0xA4, got 0x%X 0x%02X",
                  expected_id, reply.can_id, reply.data[0]);
      return;
    }

    // Parse feedback
    const int8_t temp = static_cast<int8_t>(reply.data[1]);
    const int16_t iq = static_cast<int16_t>((reply.data[3] << 8) | reply.data[2]);
    const int16_t speed = static_cast<int16_t>((reply.data[5] << 8) | reply.data[4]);
    const int16_t angle_deg_int = static_cast<int16_t>((reply.data[7] << 8) | reply.data[6]);
    last_angle_deg_[id] = static_cast<float>(angle_deg_int);

    const double new_error = target_deg - last_angle_deg_[id];

    RCLCPP_INFO(this->get_logger(),
      "FBK -> Motor 0x%X | Temp:%d C | Iq:%.2f A | ActSpeed:%d dps | Angle:%d deg | err:%.2f",
      id, temp, iq * 0.01f, speed, angle_deg_int, new_error);

    // Log to CSV
    const auto t_now = Clock::now().time_since_epoch();
    const auto timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(t_now).count();
    csv_file_ << std::hex << id << std::dec << "," << timestamp_us << ","
              << "0,0,0,0," // write/read timings omitted here; keep structure
              << static_cast<int>(temp) << "," << iq * 0.01f << ","
              << speed << "," << angle_deg_int << "\n";

    // Decide completion
    if (std::abs(new_error) < threshold) {
      executed_[id] = true;
      RCLCPP_INFO(this->get_logger(), "OK  -> Motor 0x%X reached target (|err| < %.2f)", id, threshold);

      // Advance to next motor
      current_motor_idx_ = (current_motor_idx_ + 1) % motor_ids_.size();

      // If wrapped, start a new cycle (reset executed)
      if (current_motor_idx_ == 0) {
        for (auto &kv : executed_) kv.second = false;
        cycle_idx_++;
        RCLCPP_INFO(this->get_logger(), "=== Cycle %zu completed; restarting sequence ===", cycle_idx_);
      }
    }
  }

  bool readMultiTurnAngleWithRetry(uint32_t motor_id, float &angle_deg_out, int retries = 3) {
    for (int i = 0; i < retries; i++) {
      if (readMultiTurnAngle(motor_id, angle_deg_out)) return true;
      usleep(20000);
    }
    return false;
  }

  bool readMultiTurnAngle(uint32_t motor_id, float &angle_deg_out) {
    struct can_frame cmd = {};
    cmd.can_id = motor_id;
    cmd.can_dlc = 8;
    cmd.data[0] = 0x92;
    if (write(socket_fd_, &cmd, sizeof(cmd)) < 0) {
      RCLCPP_WARN(this->get_logger(), "Write failed for angle request 0x92 (motor 0x%X)", motor_id);
      return false;
    }

    struct can_frame reply;
    struct timeval timeout = {0, 10000}; // 10ms
    fd_set read_set;
    FD_ZERO(&read_set);
    FD_SET(socket_fd_, &read_set);

    int ret = select(socket_fd_ + 1, &read_set, nullptr, nullptr, &timeout);
    if (ret > 0 && FD_ISSET(socket_fd_, &read_set)) {
      ssize_t bytes = read(socket_fd_, &reply, sizeof(reply));
      if (bytes > 0 && reply.data[0] == 0x92 && reply.can_id == (motor_id + 0x100)) {
        int32_t angle_raw;
        std::memcpy(&angle_raw, &reply.data[4], sizeof(angle_raw));
        angle_deg_out = angle_raw * 0.01f;
        return true;
      }
    }
    return false;
  }

  void sendA4PositionCommand(uint32_t motor_id, int32_t angleControl, uint16_t maxSpeed) {
    struct can_frame frame {};
    frame.can_id = motor_id;
    frame.can_dlc = 8;

    frame.data[0] = 0xA4;
    frame.data[1] = 0x00;
    frame.data[2] = static_cast<uint8_t>(maxSpeed);
    frame.data[3] = static_cast<uint8_t>(maxSpeed >> 8);
    frame.data[4] = static_cast<uint8_t>(angleControl);
    frame.data[5] = static_cast<uint8_t>(angleControl >> 8);
    frame.data[6] = static_cast<uint8_t>(angleControl >> 16);
    frame.data[7] = static_cast<uint8_t>(angleControl >> 24);

    if (write(socket_fd_, &frame, sizeof(frame)) < 0) {
      RCLCPP_WARN(this->get_logger(), "Write failed for A4 command (motor 0x%X)", motor_id);
    }
  }
};

int main(int argc, char *argv[]) {
  try {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<A4MultiMotorControlNode>());
    rclcpp::shutdown();
  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}