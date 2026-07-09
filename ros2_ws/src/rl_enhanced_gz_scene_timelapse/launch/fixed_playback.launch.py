from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("rl_enhanced_gz_scene")
    world_name = "rl_enhanced_fixed_scene"
    world_path = PathJoinSubstitution([pkg_share, "worlds", "fixed_scene_world.sdf"])
    model_path = PathJoinSubstitution([pkg_share, "models", "payload_box", "model.sdf"])
    playback_script = PathJoinSubstitution([pkg_share, "scripts", "playback_payload_trajectory.py"])

    xyz_default = "/home/eugene/RL_enhanced/qp3d_sample_xyz.npy"
    t_default = "/home/eugene/RL_enhanced/qp3d_sample_t.npy"
    export_default = "/home/eugene/RL_enhanced/corridor_export.mat"

    gz_server = ExecuteProcess(
        cmd=["gz", "sim", "-s", "-r", world_path],
        output="screen",
    )

    gz_gui = TimerAction(
        period=4.0,
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

    spawn_payload = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "run",
                    "ros_gz_sim",
                    "create",
                    "-world",
                    world_name,
                    "-file",
                    model_path,
                    "-name",
                    "trajectory_payload",
                    "-x",
                    "0.0",
                    "-y",
                    "-2.0",
                    "-z",
                    "0.5",
                ],
                output="screen",
            )
        ],
    )

    playback = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "python3",
                    "-u",
                    playback_script,
                    "--world",
                    world_name,
                    "--entity",
                    "trajectory_payload",
                    "--xyz",
                    LaunchConfiguration("xyz_path"),
                    "--time",
                    LaunchConfiguration("t_path"),
                    "--export",
                    LaunchConfiguration("export_path"),
                    "--speed",
                    LaunchConfiguration("speed"),
                    "--z-offset",
                    LaunchConfiguration("z_offset"),
                    "--theta-sign",
                    LaunchConfiguration("theta_sign"),
                ],
                output="screen",
            )
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("bridge_clock", default_value="true"),
            DeclareLaunchArgument("gui", default_value="false"),
            DeclareLaunchArgument("xyz_path", default_value=xyz_default),
            DeclareLaunchArgument("t_path", default_value=t_default),
            DeclareLaunchArgument("export_path", default_value=export_default),
            DeclareLaunchArgument("speed", default_value="1.0"),
            DeclareLaunchArgument("z_offset", default_value="0.0"),
            DeclareLaunchArgument("theta_sign", default_value="1.0"),
            SetEnvironmentVariable(name="GZ_SIM_RESOURCE_PATH", value=pkg_share),
            gz_server,
            gz_gui,
            clock_bridge,
            spawn_payload,
            playback,
        ]
    )
