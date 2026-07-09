from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("rl_enhanced_gz_scene")
    world_path = PathJoinSubstitution(
        [pkg_share, "worlds", "rescue_ruins_world.sdf"]
    )

    pkg_resource_path = PathJoinSubstitution([pkg_share])

    gz_server = ExecuteProcess(
        cmd=["gz", "sim", "-s", "-r", world_path],
        output="screen",
    )

    gz_gui = TimerAction(
        period=1.0,
        actions=[
            ExecuteProcess(
                cmd=["gz", "sim", "-g"],
                output="screen",
                additional_env={
                    "__NV_PRIME_RENDER_OFFLOAD": "1",
                    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
                    "__VK_LAYER_NV_optimus": "NVIDIA_only",
                },
            )
        ],
        condition=IfCondition(LaunchConfiguration("gui")),
    )

    clock_bridge = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "ros_gz_bridge",
            "parameter_bridge",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("bridge_clock")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "bridge_clock",
                default_value="true",
                description="Bridge Gazebo /clock into ROS 2.",
            ),
            DeclareLaunchArgument(
                "gui",
                default_value="false",
                description="Start Gazebo GUI in a separate process.",
            ),
            SetEnvironmentVariable(
                name="GZ_SIM_RESOURCE_PATH",
                value=pkg_resource_path,
            ),
            gz_server,
            gz_gui,
            clock_bridge,
        ]
    )
