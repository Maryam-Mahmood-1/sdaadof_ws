import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # 1. Path Setup
    pkg_desc = get_package_share_directory("daadbot_desc")
    pkg_controller = get_package_share_directory("daadbot_controller")
    
    # Path to your URDF (Required for Rviz to see the robot model)
    xacro_file = os.path.join(pkg_desc, 'urdf/2_link_urdf/2link_robot.urdf.xacro')


    # 2. Gazebo Simulation
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_desc, "launch", "gazebo_2link_effort.launch.py")
        )
    )
    
    # 3. Controller Manager
    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_controller, "launch", "effort_forw_controller_2link.launch.py")
        ),
        launch_arguments={"is_sim": "True"}.items()
    )

    # 4. Robot State Publisher 
    robot_description = ParameterValue(
        Command([
            'xacro ', xacro_file
        ]), 
        value_type=str
    )
    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    # # 5. Rviz2
    # rviz_node = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     name='rviz2',
    #     output='screen',
    #     arguments=['-d', os.path.join(pkg_desc, 'rviz/traj_safety2.rviz')]
    # )

   

    return LaunchDescription([
        gazebo,
        controller,
        robot_state_publisher_node,
        # rviz_node,
    ])