from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch.substitutions import Command, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory



def generate_launch_description():

    # -----------------------------
    # Paths
    # -----------------------------
    urdf_path = "/home/maryam-mahmood/test_bot.urdf"
    world_path = "/home/maryam-mahmood/udaadbot_ws/src/daadbot_desc/worlds/empty.sdf"

    # -----------------------------
    # Gazebo resource path
    # -----------------------------
    gazebo_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value="/home/maryam-mahmood/udaadbot_ws/src/daadbot_desc/worlds"
    )

    # -----------------------------
    # robot_description parameter
    # -----------------------------
    robot_description = Command(['xacro ', urdf_path])  # in case xacro, otherwise just read URDF

    # -----------------------------
    # Robot State Publisher
    # -----------------------------
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}]
    )

    # -----------------------------
    # Launch Gazebo
    # -----------------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            )
        ),
        launch_arguments=[('gz_args', f'-r -v 4 {world_path}')],
    )

    # -----------------------------
    # Spawn robot
    # -----------------------------
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-file', urdf_path,  # absolute path
            '-name', 'test_bot',
            '-x', '0', '-y', '0', '-z', '0',
            '-R', '0', '-P', '0', '-Y', '0'
        ],
        output='screen'
    )

    # -----------------------------
    # Bridge / Clock
    # -----------------------------
    gz_ros2_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
    )

    return LaunchDescription([
        gazebo_resource_path,
        robot_state_publisher_node,
        gazebo,
        gz_spawn_entity,
        gz_ros2_bridge
    ])
