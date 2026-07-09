from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackagePrefix, FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("rl_enhanced_gz_scene_timelapse")
    pkg_prefix = FindPackagePrefix("rl_enhanced_gz_scene_timelapse")
    plugin_lib_path = PathJoinSubstitution([pkg_prefix, "lib"])
    world_name = "rescue_ruins_world"
    world_path = PathJoinSubstitution([pkg_share, "worlds", "rescue_ruins_world.sdf"])
    model_path = PathJoinSubstitution([pkg_share, "models", "payload_box", "model.sdf"])
    prep_script = PathJoinSubstitution([pkg_share, "scripts", "prepare_playback_csv.py"])
    prep_inference_script = PathJoinSubstitution([pkg_share, "scripts", "prepare_inference_playback_csv.py"])
    timelapse_script = PathJoinSubstitution([pkg_share, "scripts", "spawn_timelapse_snapshots.py"])

    xyz_default = "/home/eugene/RL_enhanced/qp3d_sample_xyz.npy"
    t_default = "/home/eugene/RL_enhanced/qp3d_sample_t.npy"
    export_default = "/home/eugene/RL_enhanced/corridor_export.mat"
    inference_default = "/home/eugene/Payload_Model/RL_Enhanced/inverse_transport_development/results/planner_export_inference/planner_inference_series.npz"
    csv_default = "/tmp/rl_enhanced_playback_samples.csv"
    overlay_default = "/tmp/rl_enhanced_timelapse_overlay.sdf"

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

    prepare_timelapse_overlay = ExecuteProcess(
        cmd=[
            "python3",
            "-u",
            timelapse_script,
            "--csv",
            LaunchConfiguration("csv_path"),
            "--out",
            LaunchConfiguration("timelapse_overlay_path"),
            "--spacing",
            LaunchConfiguration("timelapse_spacing"),
            "--z-offset",
            LaunchConfiguration("z_offset"),
            "--max-samples",
            LaunchConfiguration("timelapse_max_samples"),
            "--doorway-y",
            LaunchConfiguration("timelapse_doorway_y"),
            "--doorway-lead",
            LaunchConfiguration("timelapse_doorway_lead"),
        ],
        output="screen",
    )

    prepare_timelapse_overlay_after_csv = RegisterEventHandler(
        OnProcessExit(
            target_action=prepare_csv,
            on_exit=[prepare_timelapse_overlay],
        ),
        condition=IfCondition(LaunchConfiguration("timelapse_enable")),
    )

    prepare_timelapse_overlay_after_inference_csv = RegisterEventHandler(
        OnProcessExit(
            target_action=prepare_inference_csv,
            on_exit=[prepare_timelapse_overlay],
        ),
        condition=IfCondition(LaunchConfiguration("timelapse_enable")),
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

    spawn_timelapse_overlay = TimerAction(
        period=3.8,
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
                    LaunchConfiguration("timelapse_overlay_path"),
                    "-name",
                    "trajectory_timelapse_overlay",
                ],
                output="screen",
            )
        ],
        condition=IfCondition(LaunchConfiguration("timelapse_enable")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("bridge_clock", default_value="true"),
            DeclareLaunchArgument("gui", default_value="true"),
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
            DeclareLaunchArgument("timelapse_enable", default_value="true"),
            DeclareLaunchArgument("timelapse_spacing", default_value="0.50"),
            DeclareLaunchArgument("timelapse_max_samples", default_value="90"),
            DeclareLaunchArgument("timelapse_doorway_y", default_value="1.0,6.0,11.0,16.0"),
            DeclareLaunchArgument("timelapse_doorway_lead", default_value="0.35"),
            DeclareLaunchArgument("timelapse_overlay_path", default_value=overlay_default),
            SetEnvironmentVariable(name="GZ_SIM_RESOURCE_PATH", value=pkg_share),
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
            prepare_timelapse_overlay_after_inference_csv,
            prepare_timelapse_overlay_after_csv,
            gz_server,
            gz_gui,
            clock_bridge,
            spawn_payload,
            spawn_timelapse_overlay,
        ]
    )
