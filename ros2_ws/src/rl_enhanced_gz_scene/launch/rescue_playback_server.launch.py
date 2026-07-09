from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("rl_enhanced_gz_scene")
    world_path = PathJoinSubstitution([pkg_share, "worlds", "rescue_ruins_world.sdf"])
    rviz_config = PathJoinSubstitution([pkg_share, "rviz", "payload_playback.rviz"])
    playback_script = PathJoinSubstitution([pkg_share, "scripts", "rviz_payload_playback.py"])

    xyz_default = "/home/eugene/RL_enhanced/qp3d_sample_xyz.npy"
    t_default = "/home/eugene/RL_enhanced/qp3d_sample_t.npy"
    export_default = "/home/eugene/RL_enhanced/corridor_export.mat"

    gz_server = ExecuteProcess(
        cmd=["gz", "sim", "-s", "-r", world_path],
        output="screen",
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

    world_tf = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "tf2_ros",
            "static_transform_publisher",
            "--frame-id",
            "world",
            "--child-frame-id",
            "scene_origin",
        ],
        output="screen",
    )

    rviz = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=["rviz2", "-d", rviz_config, "-f", "world"],
                output="screen",
            )
        ],
    )

    playback = TimerAction(
        period=2.5,
        actions=[
            ExecuteProcess(
                cmd=[
                    "python3",
                    playback_script,
                    "--xyz",
                    LaunchConfiguration("xyz_path"),
                    "--time",
                    LaunchConfiguration("t_path"),
                    "--speed",
                    LaunchConfiguration("speed"),
                    "--export",
                    LaunchConfiguration("export_path"),
                ],
                output="screen",
            )
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("bridge_clock", default_value="true"),
            DeclareLaunchArgument("xyz_path", default_value=xyz_default),
            DeclareLaunchArgument("t_path", default_value=t_default),
            DeclareLaunchArgument("export_path", default_value=export_default),
            DeclareLaunchArgument("speed", default_value="1.0"),
            SetEnvironmentVariable(name="GZ_SIM_RESOURCE_PATH", value=pkg_share),
            gz_server,
            clock_bridge,
            world_tf,
            rviz,
            playback,
        ]
    )
