#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <geometry_msgs/msg/pose.hpp>

class HandPose : public rclcpp::Node
{
public:
  HandPose() : Node("hand_pose_node")
  {
    RCLCPP_INFO(get_logger(), "Initializing MoveIt for direct execution...");

    // Schedule initialization after node setup
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&HandPose::initializeMoveIt, this));
  }

private:
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_move_group_, gripper_move_group_;
  rclcpp::TimerBase::SharedPtr timer_;  // Timer to delay initialization

  void initializeMoveIt()
  {
    // Stop the timer to prevent repeated execution
    timer_->cancel();

    // Now it's safe to use shared_from_this()
    arm_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "arm");
    gripper_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "gripper");

    moveArm();
  }

  void moveArm()
  {
    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = 0.35;
    target_pose.position.y = 0.0;
    target_pose.position.z = 0.6;
    target_pose.orientation.w = 0.015;
    target_pose.orientation.x = 0.1;
    target_pose.orientation.y = 0.01;
    target_pose.orientation.z = 0.0;

    arm_move_group_->setPoseTarget(target_pose);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (arm_move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_INFO(get_logger(), "Plan successful! Executing...");
      arm_move_group_->move();
    }
    else
    {
      RCLCPP_ERROR(get_logger(), "Planning failed.");
    }
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HandPose>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
