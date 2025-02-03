from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch.substitutions import Command, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    daadbot_desc_dir = get_package_share_directory('daadbot_desc')

    model_arg = DeclareLaunchArgument(
        name = "model", 
        default_value= os.path.join(daadbot_desc_dir, 'urdf/daadbot.urdf.xacro'),
        description = 'Absolute path to robot urdf file'
        )
    
    robot_description = ParameterValue(Command(["xacro ", LaunchConfiguration("model")]))
    gazebo_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH", 
        value = [str(Path(daadbot_desc_dir).parent.resolve())]
    )

    ros_distro = os.environ.get('ROS_DISTRO')
    physics_engine = "" if ros_distro == 'humble' else "--physics-engine gz-physics-bullet-featherstone-plugin"


    robot_state_publisher_node = Node(
        package = 'robot_state_publisher', 
        executable = 'robot_state_publisher',
        parameters = [{"robot_description": robot_description,
                       "use_sim_time": True}]
        )
    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 
            'launch'
            ), '/gz_sim.launch.py']
        ),
        launch_arguments = [
            ("gz_args", [" -v 4 -r empty.sdf ", physics_engine])
        ]
    )

    gz_spawn_entity = Node(
        package = 'ros_gz_sim',
        executable = 'create',
        arguments = ['-topic', 'robot_description', 'name', 'daadbot'],
        output = 'screen'
    )

    gz_ros2_bridge = Node(
        package = 'ros_gz_bridge',
        executable = 'parameter_bridge',
        arguments= ['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
                ]
    )

    return LaunchDescription([
        model_arg,  
        robot_state_publisher_node,
        gazebo_resource_path,
        gazebo,
        gz_spawn_entity,
        gz_ros2_bridge
    ])