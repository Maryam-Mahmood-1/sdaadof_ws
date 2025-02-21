#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include "daadbot_msgs/action/daadbot_task_server.hpp"
#include <moveit/move_group_interface/move_group_interface.hpp>

#include <memory>


using namespace std::placeholders;

namespace daadbot_posegen
{
class TaskServer : public rclcpp::Node
{
public:
  explicit TaskServer(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
    : Node("multi_task_server", options)
  {
    RCLCPP_INFO(get_logger(), "Starting the Server");
    action_server_ = rclcpp_action::create_server<daadbot_msgs::action::DaadbotTaskServer>(
        this, "multi_task_server", std::bind(&TaskServer::goalCallback, this, _1, _2),
        std::bind(&TaskServer::cancelCallback, this, _1),
        std::bind(&TaskServer::acceptedCallback, this, _1));
  }

private:
  rclcpp_action::Server<daadbot_msgs::action::DaadbotTaskServer>::SharedPtr action_server_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_move_group_, gripper_move_group_;
  std::vector<double> arm_joint_goal_, gripper_joint_goal_;

  rclcpp_action::GoalResponse goalCallback(
      const rclcpp_action::GoalUUID& uuid,
      std::shared_ptr<const daadbot_msgs::action::DaadbotTaskServer::Goal> goal)
  {
    RCLCPP_INFO(get_logger(), "Received goal request with id %d", goal->task_id);
    (void)uuid;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse cancelCallback(
      const std::shared_ptr<rclcpp_action::ServerGoalHandle<daadbot_msgs::action::DaadbotTaskServer>> goal_handle)
  {
    (void)goal_handle;
    RCLCPP_INFO(get_logger(), "Received request to cancel goal");
    if(arm_move_group_){
      arm_move_group_->stop();
    }
    if(gripper_move_group_){
      gripper_move_group_->stop();
    }
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void acceptedCallback(
      const std::shared_ptr<rclcpp_action::ServerGoalHandle<daadbot_msgs::action::DaadbotTaskServer>> goal_handle)
  {
    // this needs to return quickly to avoid blocking the executor, so spin up a new thread
    std::thread{ std::bind(&TaskServer::execute, this, _1), goal_handle }.detach();
  }

  void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<daadbot_msgs::action::DaadbotTaskServer>> goal_handle)
  {
    RCLCPP_INFO(get_logger(), "Executing goal");
    auto result = std::make_shared<daadbot_msgs::action::DaadbotTaskServer::Result>();

    // MoveIt 2 Interface
    if(!arm_move_group_){
      arm_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "arm");
    }
    if(!gripper_move_group_){
      gripper_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "gripper");
    }

    if (goal_handle->get_goal()->task_id == 0)
    {
      arm_joint_goal_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      gripper_joint_goal_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    else if (goal_handle->get_goal()->task_id == 1)
    {
      arm_joint_goal_ = {0.0, 0.543, 0.0, -0.209, 0.0, -1.239, 0.0};
      gripper_joint_goal_ = {0.5, 0.5, -0.5, 0.5, 0.5, -0.5};
    }
    else if (goal_handle->get_goal()->task_id == 2)
    {
      arm_joint_goal_ = {0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0};
      gripper_joint_goal_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    else if (goal_handle->get_goal()->task_id == 3)
    {
      arm_joint_goal_ = {0.0, -0.4363, 0.0, 3.333, 0.0, -2.3 , 0.0};
      gripper_joint_goal_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    else
    {
      RCLCPP_ERROR(get_logger(), "Invalid Task Number");
      return;
    }

    arm_move_group_->setStartState(*arm_move_group_->getCurrentState());
    gripper_move_group_->setStartState(*gripper_move_group_->getCurrentState());

    bool arm_within_bounds = arm_move_group_->setJointValueTarget(arm_joint_goal_);
    bool gripper_within_bounds = gripper_move_group_->setJointValueTarget(gripper_joint_goal_);
    if (!arm_within_bounds | !gripper_within_bounds)
    {
      RCLCPP_WARN(get_logger(),
                  "Target joint position(s) were outside of limits, but we will plan and clamp to the limits ");
      return;
    }

    moveit::planning_interface::MoveGroupInterface::Plan arm_plan;
    moveit::planning_interface::MoveGroupInterface::Plan gripper_plan;
    bool arm_plan_success = (arm_move_group_->plan(arm_plan) == moveit::core::MoveItErrorCode::SUCCESS);
    bool gripper_plan_success = (gripper_move_group_->plan(gripper_plan) == moveit::core::MoveItErrorCode::SUCCESS);
    
    if(arm_plan_success && gripper_plan_success)
    {
      RCLCPP_INFO(get_logger(), "Planner SUCCEED, moving the arme and the gripper");
      arm_move_group_->move();
      gripper_move_group_->move();
    }
    else
    {
      RCLCPP_ERROR(get_logger(), "One or more planners failed!");
      return;
    }
  
    result->success = true;
    goal_handle->succeed(result);
    RCLCPP_INFO(get_logger(), "Goal succeeded");
  }
};
}  // namespace daadbot_posegen

RCLCPP_COMPONENTS_REGISTER_NODE(daadbot_posegen::TaskServer)