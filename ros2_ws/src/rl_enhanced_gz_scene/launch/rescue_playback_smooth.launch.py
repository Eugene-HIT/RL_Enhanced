import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackagePrefix, FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("rl_enhanced_gz_scene")
    pkg_prefix = FindPackagePrefix("rl_enhanced_gz_scene")
    default_partition = f"rl_enhanced_payload_demo_{os.getpid()}"
    plugin_lib_path = PathJoinSubstitution([pkg_prefix, "lib"])
    model_path = PathJoinSubstitution([pkg_share, "models", "payload_box", "model.sdf"])
    gui_config_path = PathJoinSubstitution([pkg_share, "gui", "payload_demo_gui.config"])
    prep_script = PathJoinSubstitution([pkg_share, "scripts", "prepare_playback_csv.py"])
    prep_inference_script = PathJoinSubstitution([pkg_share, "scripts", "prepare_inference_playback_csv.py"])
    default_world_path = PathJoinSubstitution([pkg_share, "worlds", "payload_visual_demo_world.sdf"])

    xyz_default = "/home/eugene/RL_enhanced/qp3d_sample_xyz.npy"
    t_default = "/home/eugene/RL_enhanced/qp3d_sample_t.npy"
    export_default = "/home/eugene/RL_enhanced/corridor_export.mat"
    inference_default = "/home/eugene/Payload_Model/RL_Enhanced/inverse_transport_development/results/planner_export_inference/planner_inference_series.npz"
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
        condition=UnlessCondition(LaunchConfiguration("use_inference_npz")),
    )

    prepare_inference_csv = ExecuteProcess(
        cmd=[
            "python3",
            "-u",
            prep_inference_script,
            "--npz",
            LaunchConfiguration("inference_npz_path"),
            "--out",
            LaunchConfiguration("csv_path"),
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_inference_npz")),
    )

    gz_gui = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "/usr/bin/env",
                    "-u",
                    "GIO_MODULE_DIR",
                    "-u",
                    "GSETTINGS_SCHEMA_DIR",
                    "-u",
                    "GTK_EXE_PREFIX",
                    "-u",
                    "GTK_IM_MODULE_FILE",
                    "-u",
                    "GTK_PATH",
                    "-u",
                    "LOCPATH",
                    "-u",
                    "XDG_DATA_HOME",
                    "-u",
                    "XDG_DATA_DIRS",
                    "gz",
                    "sim",
                    "-g",
                    "--gui-config",
                    gui_config_path,
                ],
                output="screen",
            )
        ],
        condition=IfCondition(LaunchConfiguration("gui")),
    )

    gz_server_paused = TimerAction(
        period=1.2,
        actions=[
            ExecuteProcess(
                cmd=["gz", "sim", "-s", LaunchConfiguration("world_path")],
                output="screen",
            )
        ],
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("auto_run"), "' != 'true'"])),
    )

    gz_server_running = TimerAction(
        period=1.2,
        actions=[
            ExecuteProcess(
                cmd=["gz", "sim", "-s", "-r", LaunchConfiguration("world_path")],
                output="screen",
            )
        ],
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("auto_run"), "' == 'true'"])),
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
                    LaunchConfiguration("world_name"),
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
        condition=IfCondition(LaunchConfiguration("spawn_payload_entity")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("bridge_clock", default_value="true"),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("auto_run", default_value="false"),
            DeclareLaunchArgument("gz_partition", default_value=default_partition),
            DeclareLaunchArgument("world_name", default_value="payload_demo_world"),
            DeclareLaunchArgument("world_path", default_value=default_world_path),
            DeclareLaunchArgument("spawn_payload_entity", default_value="false"),
            DeclareLaunchArgument("xyz_path", default_value=xyz_default),
            DeclareLaunchArgument("t_path", default_value=t_default),
            DeclareLaunchArgument("export_path", default_value=export_default),
            DeclareLaunchArgument("inference_npz_path", default_value=inference_default),
            DeclareLaunchArgument("use_inference_npz", default_value="true"),
            DeclareLaunchArgument("csv_path", default_value=csv_default),
            DeclareLaunchArgument("speed", default_value="1.0"),
            DeclareLaunchArgument("z_offset", default_value="1.10"),
            DeclareLaunchArgument("theta_sign", default_value="-1.0"),
            DeclareLaunchArgument("flip_theta_ymin", default_value="nan"),
            DeclareLaunchArgument("flip_theta_ymax", default_value="nan"),
            DeclareLaunchArgument("flip_theta_sign", default_value="-1.0"),
            SetEnvironmentVariable(name="GZ_SIM_RESOURCE_PATH", value=pkg_share),
            SetEnvironmentVariable(name="GZ_PARTITION", value=LaunchConfiguration("gz_partition")),
            SetEnvironmentVariable(
                name="GZ_SIM_SYSTEM_PLUGIN_PATH",
                value=[plugin_lib_path, ":/opt/ros/jazzy/opt/gz_sim_vendor/lib"],
            ),
            SetEnvironmentVariable(name="RL_ENHANCED_SMOOTH_ENABLE", value="1"),
            SetEnvironmentVariable(name="RL_ENHANCED_PLAYBACK_CSV", value=LaunchConfiguration("csv_path")),
            SetEnvironmentVariable(name="RL_ENHANCED_SMOOTH_SPEED", value=LaunchConfiguration("speed")),
            SetEnvironmentVariable(name="RL_ENHANCED_SMOOTH_Z_OFFSET", value=LaunchConfiguration("z_offset")),
            prepare_inference_csv,
            prepare_csv,
            gz_server_paused,
            gz_server_running,
            gz_gui,
            clock_bridge,
            spawn_payload,
        ]
    )
