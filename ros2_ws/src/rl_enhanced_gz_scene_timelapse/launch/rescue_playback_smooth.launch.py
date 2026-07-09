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
    prep_script = PathJoinSubstitution([pkg_share, "scripts", "prepare_playback_csv.py"])

    xyz_default = "/home/eugene/RL_enhanced/qp3d_sample_xyz.npy"
    t_default = "/home/eugene/RL_enhanced/qp3d_sample_t.npy"
    export_default = "/home/eugene/RL_enhanced/corridor_export.mat"
    csv_default = "/tmp/rl_enhanced_playback_samples.csv"

    prepare_csv = ExecuteProcess(
        cmd=[
            "python3",
            "-u",
            prep_script,
            "--xyz",
            LaunchConfiguration("xyz_path"),
            "--time",
            LaunchConfiguration("t_path"),
            "--export",
            LaunchConfiguration("export_path"),
            "--out",
            LaunchConfiguration("csv_path"),
            "--theta-sign",
            LaunchConfiguration("theta_sign"),
            "--flip-theta-ymin",
            LaunchConfiguration("flip_theta_ymin"),
            "--flip-theta-ymax",
            LaunchConfiguration("flip_theta_ymax"),
            "--flip-theta-sign",
            LaunchConfiguration("flip_theta_sign"),
        ],
        output="screen",
    )

    gz_server = TimerAction(
        period=1.2,
        actions=[ExecuteProcess(cmd=["gz", "sim", "-s", world_path], output="screen")],
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

    return LaunchDescription(
        [
            DeclareLaunchArgument("bridge_clock", default_value="true"),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("xyz_path", default_value=xyz_default),
            DeclareLaunchArgument("t_path", default_value=t_default),
            DeclareLaunchArgument("export_path", default_value=export_default),
            DeclareLaunchArgument("csv_path", default_value=csv_default),
            DeclareLaunchArgument("speed", default_value="1.0"),
            DeclareLaunchArgument("z_offset", default_value="1.10"),
            DeclareLaunchArgument("theta_sign", default_value="-1.0"),
            DeclareLaunchArgument("flip_theta_ymin", default_value="nan"),
            DeclareLaunchArgument("flip_theta_ymax", default_value="nan"),
            DeclareLaunchArgument("flip_theta_sign", default_value="-1.0"),
            SetEnvironmentVariable(name="GZ_SIM_RESOURCE_PATH", value=pkg_share),
            SetEnvironmentVariable(
                name="GZ_SIM_SYSTEM_PLUGIN_PATH",
                value="/home/eugene/ros2_ws/install/rl_enhanced_gz_scene/lib:/opt/ros/jazzy/opt/gz_sim_vendor/lib",
            ),
            SetEnvironmentVariable(name="RL_ENHANCED_SMOOTH_ENABLE", value="1"),
            SetEnvironmentVariable(name="RL_ENHANCED_PLAYBACK_CSV", value=LaunchConfiguration("csv_path")),
            SetEnvironmentVariable(name="RL_ENHANCED_SMOOTH_SPEED", value=LaunchConfiguration("speed")),
            SetEnvironmentVariable(name="RL_ENHANCED_SMOOTH_Z_OFFSET", value=LaunchConfiguration("z_offset")),
            prepare_csv,
            gz_server,
            gz_gui,
            clock_bridge,
            spawn_payload,
        ]
    )
