# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-06-10
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-06-10
# 目的 / Purpose:
#   导出碰撞优先 UAV 位置反推阶段的基础接口骨架。
#   Export the base interfaces for collision-first UAV pose inference.
# 主要输入 / Main Inputs:
#   安全走廊、参考 UAV 轨迹、载荷 wrench 与挂点几何。
#   Safe corridors, reference UAV trajectories, payload wrench, and attachment geometry.
# 主要输出 / Main Outputs:
#   走廊数据结构、张力可行性检查、局部 QP 接口、顺序规划入口。
#   Corridor data structures, tension feasibility checks, local QP interface,
#   and the sequential planning entry point.
# ==============================================

from .safe_corridor import (
	CorridorGenerationConfig,
	HalfspaceCorridor3D,
	InflatedObstaclePrismXZ,
	InflatedObstacleSphere,
	SceneObstacle,
	build_axis_aligned_corridors_from_reference,
	evaluate_obstacle_collisions,
	make_box_halfspaces,
	sample_segment_points,
	signed_distance_to_obstacle,
	evaluate_cable_obstacle_collisions,
)
from .tension_feasibility import (
	TensionFeasibilityInput,
	TensionFeasibilityResult,
	compute_cable_directions_from_uav_positions,
	solve_tension_feasibility,
)
from .uav_corridor_qp import (
	LocalQPProblem,
	LocalQPResult,
	UAVKinematicLimits,
	evaluate_discrete_kinematics,
	solve_local_corridor_qp,
)
from .sequential_planner import (
	SequentialPlanningConfig,
	SequentialPlanningResult,
	plan_collision_aware_uav_positions,
)

__all__ = [
	"InflatedObstacleSphere",
	"InflatedObstaclePrismXZ",
	"SceneObstacle",
	"HalfspaceCorridor3D",
	"CorridorGenerationConfig",
	"make_box_halfspaces",
	"build_axis_aligned_corridors_from_reference",
	"signed_distance_to_obstacle",
	"evaluate_obstacle_collisions",
	"sample_segment_points",
	"evaluate_cable_obstacle_collisions",
	"TensionFeasibilityInput",
	"TensionFeasibilityResult",
	"compute_cable_directions_from_uav_positions",
	"solve_tension_feasibility",
	"UAVKinematicLimits",
	"LocalQPProblem",
	"LocalQPResult",
	"evaluate_discrete_kinematics",
	"solve_local_corridor_qp",
	"SequentialPlanningConfig",
	"SequentialPlanningResult",
	"plan_collision_aware_uav_positions",
]