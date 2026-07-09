from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io


DOOR_RENDER_THICKNESS_M = 0.20
DOOR_FRAME_MARGIN_X_M = 0.40
DOOR_FRAME_MARGIN_Z_M = 0.35


def _unwrap_mat_sequence(value):
	if value is None:
		return []
	if isinstance(value, np.ndarray):
		if value.shape == ():
			return [value.item()]
		return list(np.asarray(value, dtype=object).reshape(-1))
	return [value]


def _as_polygon_array(poly_xz) -> np.ndarray | None:
	poly_xz = np.asarray(poly_xz, dtype=float)
	if poly_xz.ndim != 2 or poly_xz.shape[1] != 2 or poly_xz.shape[0] < 3:
		return None
	return poly_xz


def _box_model_xml(name: str, pose: tuple[float, float, float, float, float, float], size: tuple[float, float, float], color: str, transparency: float, static: bool = True) -> str:
	pose_str = " ".join(f"{value:.6f}" for value in pose)
	size_str = " ".join(f"{value:.6f}" for value in size)
	static_str = "true" if static else "false"
	return f"""
    <model name=\"{name}\">
      <static>{static_str}</static>
      <pose>{pose_str}</pose>
      <link name=\"link\">
        <gravity>false</gravity>
        <inertial>
          <mass>0.10</mass>
          <inertia>
            <ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz>
            <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
          </inertia>
        </inertial>
        <collision name=\"collision\">
          <geometry><box><size>{size_str}</size></box></geometry>
        </collision>
        <visual name=\"visual\">
          <geometry><box><size>{size_str}</size></box></geometry>
          <material>
            <ambient>{color}</ambient>
            <diffuse>{color}</diffuse>
            <specular>0.08 0.08 0.08 1</specular>
            <emissive>0.02 0.02 0.02 1</emissive>
          </material>
          <transparency>{transparency:.6f}</transparency>
        </visual>
      </link>
    </model>
"""


def _drone_model_xml(name: str, pose: tuple[float, float, float, float, float, float], body_color: str) -> str:
	pose_str = " ".join(f"{value:.6f}" for value in pose)
	return f"""
    <model name=\"{name}\">
      <static>false</static>
      <pose>{pose_str}</pose>
      <link name=\"link\">
        <gravity>false</gravity>
        <inertial>
          <mass>0.15</mass>
          <inertia>
            <ixx>0.0005</ixx><iyy>0.0005</iyy><izz>0.0008</izz>
            <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
          </inertia>
        </inertial>
        <visual name=\"core\">
          <geometry><sphere><radius>0.070000</radius></sphere></geometry>
          <material><ambient>{body_color}</ambient><diffuse>{body_color}</diffuse></material>
        </visual>
        <visual name=\"arm_x\">
          <geometry><box><size>0.280000 0.030000 0.020000</size></box></geometry>
          <material><ambient>0.92 0.92 0.92 1</ambient><diffuse>0.92 0.92 0.92 1</diffuse></material>
        </visual>
        <visual name=\"arm_y\">
          <geometry><box><size>0.030000 0.280000 0.020000</size></box></geometry>
          <material><ambient>0.92 0.92 0.92 1</ambient><diffuse>0.92 0.92 0.92 1</diffuse></material>
        </visual>
      </link>
    </model>
"""


def _goal_marker_xml(goal_xyz: np.ndarray) -> str:
	return f"""
    <model name=\"goal_cross_marker\">
      <static>false</static>
      <pose>{goal_xyz[0]:.6f} {goal_xyz[1]:.6f} {goal_xyz[2]:.6f} 0 0 0</pose>
      <link name=\"link\">
        <gravity>false</gravity>
        <inertial>
          <mass>0.010</mass>
          <inertia>
            <ixx>0.0001</ixx><iyy>0.0001</iyy><izz>0.0001</izz>
            <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
          </inertia>
        </inertial>
        <visual name=\"ring\">
          <pose>0 0 -0.050000 0 0 0</pose>
          <geometry><cylinder><radius>0.220000</radius><length>0.060000</length></cylinder></geometry>
          <material>
            <ambient>0.12 0.78 0.30 1</ambient>
            <diffuse>0.18 0.92 0.36 1</diffuse>
            <emissive>0.10 0.45 0.15 1</emissive>
          </material>
          <transparency>0.15</transparency>
        </visual>
      </link>
    </model>
"""


def _door_frame_segments(poly_xz: np.ndarray, y_min: float, y_max: float) -> list[tuple[tuple[float, float, float, float, float, float], tuple[float, float, float]]]:
	min_x = float(np.min(poly_xz[:, 0]))
	max_x = float(np.max(poly_xz[:, 0]))
	min_z = float(np.min(poly_xz[:, 1]))
	max_z = float(np.max(poly_xz[:, 1]))
	outer_min_x = min_x - DOOR_FRAME_MARGIN_X_M
	outer_max_x = max_x + DOOR_FRAME_MARGIN_X_M
	outer_min_z = min_z - DOOR_FRAME_MARGIN_Z_M
	outer_max_z = max_z + DOOR_FRAME_MARGIN_Z_M
	y_center = 0.5 * (float(y_min) + float(y_max))
	segments: list[tuple[tuple[float, float, float, float, float, float], tuple[float, float, float]]] = []

	def add_box(x0: float, x1: float, z0: float, z1: float) -> None:
		size_x = x1 - x0
		size_z = z1 - z0
		if size_x <= 1e-6 or size_z <= 1e-6:
			return
		segments.append(
			(
				(0.5 * (x0 + x1), y_center, 0.5 * (z0 + z1), 0.0, 0.0, 0.0),
				(size_x, DOOR_RENDER_THICKNESS_M, size_z),
			)
		)

	add_box(outer_min_x, min_x, outer_min_z, outer_max_z)
	add_box(max_x, outer_max_x, outer_min_z, outer_max_z)
	add_box(min_x, max_x, outer_min_z, min_z)
	add_box(min_x, max_x, max_z, outer_max_z)
	return segments


def build_world_sdf(mat_path: Path) -> str:
	mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
	doors_raw = _unwrap_mat_sequence(mat.get("doors"))
	forbidden_raw = _unwrap_mat_sequence(mat.get("forbidden"))
	keyframes = mat.get("keyframes")
	goal_xyz = np.array([0.0, 9.0, 1.0], dtype=float)
	if keyframes is not None and hasattr(keyframes, "P_wp"):
		points = np.asarray(getattr(keyframes, "P_wp"), dtype=float)
		if points.ndim == 2 and points.shape[1] == 3 and points.shape[0] > 0:
			goal_xyz = points[-1].astype(float)

	models: list[str] = []
	models.append(_goal_marker_xml(goal_xyz))
	models.append(
		_box_model_xml(
			name="safe_start_ring",
			pose=(0.0, -1.4, 0.92, 1.570796, 0.0, 0.0),
			size=(0.84, 0.05, 0.84),
			color="0.15 0.85 0.35 1",
			transparency=0.35,
		)
	)
	models.append(
		_box_model_xml(
			name="trajectory_payload",
			pose=(0.0, -1.4, 0.92, 0.0, 0.0, 0.0),
			size=(0.68, 0.32, 0.22),
			color="0.08 0.42 0.98 1",
			transparency=0.0,
			static=False,
		)
	)
	models.append(_drone_model_xml("world_drone_front", (0.20, -1.40, 1.47, 0.0, 0.0, 0.0), "0.95 0.95 0.95 1"))
	models.append(_drone_model_xml("world_drone_rear_left", (-0.22, -1.23, 1.47, 0.0, 0.0, 0.60), "0.95 0.78 0.22 1"))
	models.append(_drone_model_xml("world_drone_rear_right", (-0.22, -1.57, 1.47, 0.0, 0.0, -0.60), "0.18 0.85 0.85 1"))

	for door_index, door in enumerate(doors_raw):
		poly_xz = _as_polygon_array(getattr(door, "poly_xz", None))
		if poly_xz is None:
			continue
		y_min = float(getattr(door, "y_min", 0.0))
		y_max = float(getattr(door, "y_max", 0.0))
		for segment_index, (pose, size) in enumerate(_door_frame_segments(poly_xz, y_min, y_max)):
			models.append(
				_box_model_xml(
					name=f"door_frame_{door_index}_{segment_index}",
					pose=pose,
					size=size,
					color="0.62 0.80 0.60 1",
					transparency=0.12,
				)
			)

	for forbidden_index, forbidden in enumerate(forbidden_raw):
		poly_xz = _as_polygon_array(getattr(forbidden, "poly_xz", None))
		if poly_xz is None:
			continue
		min_x = float(np.min(poly_xz[:, 0]))
		max_x = float(np.max(poly_xz[:, 0]))
		min_z = float(np.min(poly_xz[:, 1]))
		max_z = float(np.max(poly_xz[:, 1]))
		y_min = float(getattr(forbidden, "y_min", 0.0))
		y_max = float(getattr(forbidden, "y_max", 0.0))
		models.append(
			_box_model_xml(
				name=f"forbidden_{forbidden_index}",
				pose=(0.5 * (min_x + max_x), 0.5 * (y_min + y_max), 0.5 * (min_z + max_z), 0.0, 0.0, 0.0),
				size=(max_x - min_x, y_max - y_min, max_z - min_z),
				color="0.82 0.24 0.24 1",
				transparency=0.10,
			)
		)

	models_text = "\n".join(models)
	return f"""<?xml version=\"1.0\" ?>
<sdf version=\"1.10\">
  <world name=\"simple_passage_world\">
    <gravity>0 0 -9.8</gravity>
    <physics name=\"1ms\" type=\"ignored\">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename=\"gz-sim-physics-system\" name=\"gz::sim::systems::Physics\"/>
    <plugin filename=\"gz-sim-user-commands-system\" name=\"gz::sim::systems::UserCommands\"/>
    <plugin filename=\"gz-sim-scene-broadcaster-system\" name=\"gz::sim::systems::SceneBroadcaster\"/>
    <plugin filename=\"gz-sim-sensors-system\" name=\"gz::sim::systems::Sensors\"/>
    <plugin filename=\"libsmooth_playback_system.so\" name=\"rl_enhanced_gz_scene::SmoothPlaybackSystem\">
      <enabled>false</enabled>
      <csv_path>/tmp/rl_enhanced_playback_samples.csv</csv_path>
      <payload_name>trajectory_payload</payload_name>
      <drone_front_name>world_drone_front</drone_front_name>
      <drone_rear_left_name>world_drone_rear_left</drone_rear_left_name>
      <drone_rear_right_name>world_drone_rear_right</drone_rear_right_name>
      <goal_marker_name>goal_cross_marker</goal_marker_name>
      <goal_marker_x>{goal_xyz[0]:.6f}</goal_marker_x>
      <goal_marker_y>{goal_xyz[1]:.6f}</goal_marker_y>
      <goal_marker_z>{goal_xyz[2]:.6f}</goal_marker_z>
      <goal_marker_yaw_rate>0.30</goal_marker_yaw_rate>
      <speed>1.0</speed>
      <z_offset>0.0</z_offset>
      <drone_z_rel>0.40</drone_z_rel>
      <drone_front_x>0.20</drone_front_x>
      <drone_front_y>0.00</drone_front_y>
      <drone_rear_left_x>-0.17</drone_rear_left_x>
      <drone_rear_left_y>0.13</drone_rear_left_y>
      <drone_rear_right_x>-0.17</drone_rear_right_x>
      <drone_rear_right_y>-0.13</drone_rear_right_y>
    </plugin>

    <scene>
      <ambient>0.20 0.22 0.24 1</ambient>
      <background>0.10 0.11 0.13 1</background>
    </scene>

    <light name=\"sun\" type=\"directional\">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 25 0 0 0</pose>
      <diffuse>0.95 0.95 0.95 1</diffuse>
      <specular>0.25 0.25 0.25 1</specular>
      <direction>-0.20 0.10 -0.97</direction>
    </light>

    <model name=\"ground_plane\">
      <static>true</static>
      <link name=\"ground_link\">
        <collision name=\"collision\">
          <geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
        </collision>
        <visual name=\"visual\">
          <geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
          <material>
            <ambient>0.22 0.23 0.24 1</ambient>
            <diffuse>0.22 0.23 0.24 1</diffuse>
            <specular>0.03 0.03 0.03 1</specular>
          </material>
        </visual>
      </link>
    </model>

{models_text}
  </world>
</sdf>
"""


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate a Gazebo world for the simplified passage scene")
	parser.add_argument(
		"--mat",
		type=Path,
		default=Path("inverse_transport_development/results/simple_passage_scene/simple_passage_scene.mat"),
		help="Path to the simplified corridor_export-compatible MAT file",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("inverse_transport_development/results/simple_passage_scene/simple_passage_gazebo_world.sdf"),
		help="Output Gazebo world SDF path",
	)
	args = parser.parse_args()
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(build_world_sdf(args.mat), encoding="ascii")
	print(f"[generate_simple_passage_gazebo_world] saved {args.output}")


if __name__ == "__main__":
	main()