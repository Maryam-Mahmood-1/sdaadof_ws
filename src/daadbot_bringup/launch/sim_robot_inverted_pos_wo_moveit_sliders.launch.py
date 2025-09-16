import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    gazebo = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("daadbot_desc"),
            "launch",
            "gazebo_inverted_pos.launch.py"
        )
    )

    controller = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("daadbot_controller"),
            "launch",
            "pos_controller_table.launch.py"
        ),
        launch_arguments={"is_sim": "True"}.items()
    )

    # Joint State Publisher GUI
    sliders = ExecuteProcess(
        cmd=["ros2", "run", "joint_state_publisher_gui", "joint_state_publisher_gui"],
        output="screen"
    )

    # Target setter node
    target_node = Node(
        package="some_examples_cpp",
        executable="target_angle_gui.py",
        output="screen"
    )

    return LaunchDescription([
        gazebo,
        controller,
        sliders,
        target_node
    ])
