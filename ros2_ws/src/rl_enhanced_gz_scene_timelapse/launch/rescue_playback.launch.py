from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("rl_enhanced_gz_scene")
    world_name = "rescue_ruins_world"
    world_path = PathJoinSubstitution([pkg_share, "worlds", "rescue_ruins_world.sdf"])
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
                    "1.0",
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
                    "--flip-theta-ymin",
                    LaunchConfiguration("flip_theta_ymin"),
                    "--flip-theta-ymax",
                    LaunchConfiguration("flip_theta_ymax"),
                    "--flip-theta-sign",
                    LaunchConfiguration("flip_theta_sign"),
                    "--goal-marker-entity",
                    LaunchConfiguration("goal_marker_entity"),
                    "--goal-marker-x",
                    LaunchConfiguration("goal_marker_x"),
                    "--goal-marker-y",
                    LaunchConfiguration("goal_marker_y"),
                    "--goal-marker-z",
                    LaunchConfiguration("goal_marker_z"),
                    "--goal-marker-yaw-rate",
                    LaunchConfiguration("goal_marker_yaw_rate"),
                    "--dt",
                    LaunchConfiguration("playback_dt"),
                    "--rope-node-count",
                    LaunchConfiguration("rope_node_count"),
                    "--rope-enable",
                    LaunchConfiguration("rope_enable"),
                    "--rope-update-every",
                    LaunchConfiguration("rope_update_every"),
                    "--marker-update-every",
                    LaunchConfiguration("marker_update_every"),
                    "--trail-enable",
                    LaunchConfiguration("trail_enable"),
                    "--service-timeout-ms",
                    LaunchConfiguration("service_timeout_ms"),
                ],
                output="screen",
            )
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("bridge_clock", default_value="true"),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("xyz_path", default_value=xyz_default),
            DeclareLaunchArgument("t_path", default_value=t_default),
            DeclareLaunchArgument("export_path", default_value=export_default),
            DeclareLaunchArgument("speed", default_value="1.0"),
            DeclareLaunchArgument("z_offset", default_value="1.10"),
            DeclareLaunchArgument("theta_sign", default_value="1.0"),
            DeclareLaunchArgument("flip_theta_ymin", default_value="nan"),
            DeclareLaunchArgument("flip_theta_ymax", default_value="nan"),
            DeclareLaunchArgument("flip_theta_sign", default_value="-1.0"),
            DeclareLaunchArgument("goal_marker_entity", default_value="goal_cross_marker"),
            DeclareLaunchArgument("goal_marker_x", default_value="0.0"),
            DeclareLaunchArgument("goal_marker_y", default_value="19.0"),
            DeclareLaunchArgument("goal_marker_z", default_value="2.05"),
            DeclareLaunchArgument("goal_marker_yaw_rate", default_value="0.30"),
            DeclareLaunchArgument("playback_dt", default_value="0.05"),
            DeclareLaunchArgument("rope_node_count", default_value="5"),
            DeclareLaunchArgument("rope_enable", default_value="false"),
            DeclareLaunchArgument("rope_update_every", default_value="3"),
            DeclareLaunchArgument("marker_update_every", default_value="4"),
            DeclareLaunchArgument("trail_enable", default_value="false"),
            DeclareLaunchArgument("service_timeout_ms", default_value="120"),
            SetEnvironmentVariable(name="GZ_SIM_RESOURCE_PATH", value=pkg_share),
            gz_server,
            gz_gui,
            clock_bridge,
            spawn_payload,
            playback,
        ]
    )
