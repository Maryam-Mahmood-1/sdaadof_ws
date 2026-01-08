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
    xacro_file = os.path.join(pkg_desc, 'urdf/urdf_inverted_torque/daadbot.urdf')

    # 2. Gazebo Simulation
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_desc, "launch", "gazebo_inverted_effort.launch.py")
        )
    )
    
    # 3. Controller Manager
    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_controller, "launch", "effort_forw_controller_inverted.launch.py")
        ),
        launch_arguments={"is_sim": "True"}.items()
    )

    # 4. Robot State Publisher 
    # (Crucial: Reads URDF and publishes the TF tree so Rviz can display the robot)
    robot_description = ParameterValue(Command(['cat ', xacro_file]), value_type=str)
    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    # 5. Rviz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        # If you have a saved config, uncomment the next line:
        arguments=['-d', os.path.join(pkg_desc, 'rviz/traj_ellipse.rviz')]
    )
    # 6. Trajectory Visualizer Node (The Red Curve)
    # REPLACE 'some_examples_py' with the actual package name where you saved the script from Step 1
    # If the script is just a loose python file, you might need to run it separately or 
    # register it in your setup.py entry_points.
    
    # Assuming you added 'trajectory_visualizer = some_examples_py.trajectory_visualizer:main' 
    # to your console_scripts in setup.py:
    traj_viz_node = Node(
        package='some_examples_py', 
        executable='trajectory_visualizer',
        name='trajectory_visualizer'
    )

    return LaunchDescription([
        gazebo,
        controller,
        robot_state_publisher_node,
        rviz_node,
        traj_viz_node
    ])