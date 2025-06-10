import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("daadbot_desc"),
                "launch",
                "gazebo_table_vel.launch.py"
            )
        )
    )

    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("daadbot_controller"),
                "launch",
                "vel_controller_table.launch.py"
            )
        ),
        launch_arguments={"is_sim": "True"}.items()
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("daadbot_moveit"),
                "launch",
                "moveit_table.launch.py"
            )
        ),
        launch_arguments={"is_sim": "True"}.items()
    )

    z_torque_aggregator = Node(
        package="some_examples_py",
        executable="z_torque_aggregator",
        name="z_torque_aggregator",
        output="screen"
    )

    return LaunchDescription([
        gazebo,
        controller,
        moveit,
        z_torque_aggregator,  # ✅ New node added
    ])
