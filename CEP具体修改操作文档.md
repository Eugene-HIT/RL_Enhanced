# CEP具体修改操作文档

## 1. 文档定位

本文档是在 [CEP修改报告意见.md](/home/eugene/Payload_Model/RL_Enhanced/CEP修改报告意见.md) 基础上进一步收紧得到的执行版改稿方案，目标不是再讨论“该不该改”，而是明确：

1. 稿件层面具体改哪些章节。
2. 实验层面优先补哪些结果。
3. 代码和脚本层面优先改哪些入口。
4. 每项修改的交付物、依赖项和验收标准是什么。

当前工作区中未找到 `Paper.tex` 原稿文件，因此本操作文档无法逐段锚定到主稿行号，只能基于现有修改意见、当前实验链路和仓库中已存在脚本来落地。等主稿重新放回工作区后，可再把本文档映射到具体章节位置。

## 2. 转投CEP的核心目标

本次改稿的目标不是把现有论文“小修一下再投”，而是完成一次明确的定位转移：

1. 从“RL 求 narrow-opening passing configuration 的方法论文”，转成“面向狭窄开口吊载运输任务的工程化分层规划方案”。
2. 从“展示方法能工作”，转成“证明方案在工程语境下可信、鲁棒、可集成、具备准在线能力”。
3. 从“以 RL 模块为主叙事”，转成“以 payload-level planning pipeline 为主叙事，RL 只是局部通过构型求解器之一”。

如果只改标题和摘要、不补对比基线和鲁棒性实验，这次转投大概率仍然不够稳。真正决定 CEP 适配性的，不是写法本身，而是是否补出工程证据。

## 3. 当前基础与可直接复用资产

### 3.1 论文叙事可直接保留的骨架

以下内容原则上不需要推翻：

1. “局部 opening passing configuration + 全局 3D trajectory generation”的两阶段结构。
2. 载荷层而不是单机层的任务抽象。
3. opening 截面中的 position-orientation 建模。
4. 下游 A*、safety corridor、minimum-snap 的轨迹组织方式。

### 3.2 当前仓库里可直接复用的实验入口

下列入口已经具备继续扩展为 CEP 实验素材的基础：

1. [inverse_transport_development/experiments/run_planner_export_inference.py](/home/eugene/Payload_Model/RL_Enhanced/inverse_transport_development/experiments/run_planner_export_inference.py)
用途：主实验导出入口，已经支持 `baseline-only`、固定场景障碍、passage hints、多轮 refinement、collision/cable 统计。

2. [inverse_transport_development/experiments/plot_fixed_scene_uav_paths.py](/home/eugene/Payload_Model/RL_Enhanced/inverse_transport_development/experiments/plot_fixed_scene_uav_paths.py)
用途：固定场景 2D/3D 可视化，已经支持 UAV 轨迹、payload 轨迹、door/forbidden、cable collision 统计。

3. [inverse_transport_development/experiments/generate_simple_passage_scene.py](/home/eugene/Payload_Model/RL_Enhanced/inverse_transport_development/experiments/generate_simple_passage_scene.py)
用途：构造可控的 simplified benchmark scene，可继续扩展成 baseline、扰动和串行任务实验的统一输入源。

4. [inverse_transport_development/experiments/generate_simple_passage_gazebo_world.py](/home/eugene/Payload_Model/RL_Enhanced/inverse_transport_development/experiments/generate_simple_passage_gazebo_world.py)
用途：把 simplified scene 输出为 Gazebo world，适合做补充展示图和执行层回放。

5. [ros2_ws/src/rl_enhanced_gz_scene/scripts/prepare_inference_playback_csv.py](/home/eugene/Payload_Model/RL_Enhanced/ros2_ws/src/rl_enhanced_gz_scene/scripts/prepare_inference_playback_csv.py)
用途：将 `planner_inference_series.npz` 转换为 Gazebo smooth playback 使用的 14 列 CSV。

6. [ros2_ws/src/rl_enhanced_gz_scene/launch/rescue_playback_smooth.launch.py](/home/eugene/Payload_Model/RL_Enhanced/ros2_ws/src/rl_enhanced_gz_scene/launch/rescue_playback_smooth.launch.py)
用途：Gazebo 主回放链路，已支持直接读取 inference npz。

### 3.3 当前已存在但尚未充分利用的论文证据点

根据当前代码与记录，下面这些点已经具备写进 CEP 版论文的基础：

1. 主实验链已能输出 `planner_inference_series.npz`，包含 payload 与三架 UAV 的时序轨迹。
2. 已存在最小范数基线导出模式，即 `--baseline-only`，可直接作为一个 baseline 起点。
3. 已存在 obstacle collision 与 cable collision 统计链，适合作为 CEP 关注的几何安全指标。
4. 已存在 Gazebo world 回放链，可作为“planner-to-execution interface”展示素材。

## 4. 稿件层面的具体修改任务

### 4.1 标题

目标：把标题主语从 RL 方法切换为工程任务与系统方案。

建议动作：

1. 准备 3 个 CEP 风格候选标题。
2. 标题中避免把 reinforcement learning 放在主干位置。
3. 关键词优先保留：narrow opening、payload transportation、hierarchical planning、trajectory generation。

建议候选方向：

1. Hierarchical Passing-Configuration Planning and Safe Trajectory Generation for Payload Transportation Through Narrow Openings
2. Payload-Level Traversal Planning for Multi-UAV Suspended Transportation in Constrained Openings
3. A Hierarchical Planning Pipeline for Narrow-Opening Payload Transportation with Fast Passing-Configuration Selection

交付物：

1. 标题候选 3 版。
2. 对应的一句话定位说明，供和导师快速决策。

验收标准：

1. 标题读起来像工程系统论文，而不是纯 RL 方法论文。

### 4.2 摘要

目标：摘要必须同时回答“是什么任务”“为什么工程上重要”“方案是什么”“与什么相比更可信”“结果说明了什么”。

建议重写结构：

1. 第一句：限定任务背景，如 rescue、inspection、confined transport 中的 narrow-opening bottleneck。
2. 第二句：指出现有 tracking/control 不能直接解决 through-opening configuration selection。
3. 第三句：说明本文提出 payload-level hierarchical planning pipeline。
4. 第四句：说明 passing configuration selection 只是 pipeline 的第一阶段，后接 A*、corridor、minimum-snap。
5. 第五句：用 1 句话概括基线比较结果。
6. 第六句：用 1 句话概括鲁棒性或非理想扰动结果。
7. 最后一句：给出实时性与工程适用性的结论。

必须新增的信息：

1. 至少一个基线对比结论。
2. 至少一个扰动鲁棒性结论。
3. 总规划时间或模块耗时，而不是只报 RL inference time。

交付物：

1. 200 到 250 词 CEP 风格摘要新稿。

验收标准：

1. 摘要不再把创新点局限在 RL，而是强调系统链条与工程可信性。

### 4.3 引言

目标：压缩纯方法背景，强化工程问题与系统必要性。

建议操作：

1. 把引言前两段重写为任务驱动，而不是 RL 驱动。
2. 明确 narrow-opening traversal 是吊载运输中的局部 bottleneck，而不是附带细节。
3. 把相关工作按“payload transport / bottleneck traversal / planning-controller interface”三线重组，而不是按 RL/non-RL 分类堆砌。
4. 引出本文的核心中间变量：passing configuration。
5. 在贡献段中，把“泛化推理快”降为第三层贡献，把“系统集成与工程验证”提到前两条。

建议贡献重写为三条：

1. 提出面向狭窄开口吊载运输的 payload-level hierarchical planning pipeline。
2. 提出适用于 irregular polygonal openings 的 passing configuration selection 模块，并将其与三维安全轨迹生成整合。
3. 在 representative rescue-like scene 及非理想扰动条件下验证成功率、几何安全裕度、规划效率和执行友好性。

交付物：

1. 新版引言。
2. 新版贡献列表。

验收标准：

1. 读者在引言结束前已经清楚本文解决的是工程 bottleneck，而不是单个 learning policy 问题。

### 4.4 方法部分

目标：保留原有数学建模，但补上 CEP 最关心的接口与假设说明。

必须新增一个独立小节：Planner-to-controller interface and deployment assumptions。

该小节建议写清：

1. 上层 planner 输出内容。
输出包括：passing position、planar orientation、opening sequence、3D waypoint 或 minimum-snap trajectory、sparse roll reference。

2. 下层执行系统需要满足的条件。
包括：轨迹跟踪能力、有限姿态调整能力、基本环境几何获取能力。

3. 当前没有显式建模的因素。
包括：perception error、opening reconstruction error、payload swing during execution、wind disturbance、cable deformation。

4. 当前方法在整套系统中的角色。
说明它是 payload-level planning module，不替代底层控制器，只负责提供更合理的 through-bottleneck reference。

建议新增一张接口图：

1. Scene geometry / opening contours -> passing configuration planner -> global path / corridor / minimum-snap -> low-level controller -> execution in Gazebo.

交付物：

1. 新增方法小节。
2. 新增接口图 1 张。

验收标准：

1. CEP 审稿人能明确看出你的模块在系统中的边界，不会误判为“理想化示意规划”。

### 4.5 实验章节重构

目标：把当前“定性展示 + summary table”的组织方式，改成系统验证结构。

建议实验章节改为：

1. Experimental setup
2. Baseline comparison
3. Robustness under non-ideal conditions
4. Planning efficiency and scalability
5. Ablation study
6. Sequential multi-opening traversal

其中当前已有图可以继续保留，但必须重新挂到新的叙事骨架里，不能再作为零散展示。

### 4.6 结论与局限性

目标：从“方法总结”改成“工程适用边界总结”。

建议操作：

1. 结论第一段总结本文解决的工程 bottleneck。
2. 第二段总结在何种非理想条件下完成了验证。
3. 局限性明确写出几何已知、感知闭环未建模、多 UAV 真实协同执行未验证。
4. Future work 不要泛泛写“future hardware experiments”，而是写成 sensor-in-the-loop、geometry uncertainty、controller-coupled validation 三条。

交付物：

1. 重写结论。
2. 单独成段的 limitations and future work。

## 5. 实验补强的具体执行计划

### 5.1 第一优先级：必须补

这部分不补，转 CEP 的说服力不够。

#### A. 基线对比

最低配置必须有 3 类 baseline：

1. 固定姿态基线。
2. 几何启发式基线。
3. 采样搜索或简化优化基线。

现有基础：

1. [inverse_transport_development/experiments/run_planner_export_inference.py](/home/eugene/Payload_Model/RL_Enhanced/inverse_transport_development/experiments/run_planner_export_inference.py) 已有 `--baseline-only`，可直接作为 minimum-norm baseline。
2. 当前缺少“固定姿态”和“几何启发式”两个显式 baseline 入口，需要补脚本或补模式开关。

建议新增脚本：

1. `inverse_transport_development/experiments/run_cep_baseline_suite.py`
功能：统一跑主方法、minimum-norm baseline、fixed-orientation baseline、geometry heuristic baseline、sampling baseline，并输出同格式结果表。

建议输出指标：

1. success rate
2. minimum clearance
3. mean clearance
4. cable collision count
5. door collision count
6. single-opening planning time
7. total planning time
8. passing angle change
9. path length
10. max jerk 或 snap proxy

验收标准：

1. 至少有一张 baseline summary table。
2. 至少有一张 success rate / minimum clearance / planning time 对比图。

#### B. 非理想工况鲁棒性

最低配置必须先补 3 类：

1. opening geometry error
2. payload size mismatch
3. initial orientation error

建议新增脚本：

1. `inverse_transport_development/experiments/run_cep_robustness_suite.py`
功能：统一生成扰动样本、调用主规划入口、汇总统计并生成表图。

建议优先改造的现有输入源：

1. [inverse_transport_development/experiments/generate_simple_passage_scene.py](/home/eugene/Payload_Model/RL_Enhanced/inverse_transport_development/experiments/generate_simple_passage_scene.py)
改造方向：加入 opening vertex perturbation、payload size perturbation、roll / initial orientation perturbation 参数。

建议扰动水平：

1. geometry error：1% / 3% / 5% / 8%
2. payload size mismatch：2% / 5% / 8%
3. initial orientation error：5° / 10° / 15°

建议输出指标：

1. success rate vs disturbance level
2. minimum clearance vs disturbance level
3. collision-after-execution rate
4. infeasible rate

验收标准：

1. 每类扰动至少一张曲线图。
2. 至少一张总表汇总不同扰动等级下的成功率和最小安全裕度。

#### C. 时间分解与复杂度评估

当前不能只报 RL inference time，必须报整个 pipeline。

建议优先改造入口：

1. [inverse_transport_development/experiments/run_planner_export_inference.py](/home/eugene/Payload_Model/RL_Enhanced/inverse_transport_development/experiments/run_planner_export_inference.py)

建议增加的 timing 项：

1. passing configuration selection
2. A* planning
3. corridor generation
4. QP solving
5. sequential refinement
6. total planning time

建议新增脚本：

1. `inverse_transport_development/experiments/summarize_cep_timing.py`
功能：从日志或 npz/json 统计文件中输出时间分解表。

验收标准：

1. 一张 time breakdown table。
2. 一张复杂度趋势图，横轴为 scene complexity 或 openings count。

### 5.2 第二优先级：建议补

#### D. 消融实验

建议至少做 3 个：

1. 去掉 orientation prior。
2. 去掉 center penalty 或 equivalent regularization。
3. 去掉 safety corridor，仅保留 minimum-snap。

建议新增脚本：

1. `inverse_transport_development/experiments/run_cep_ablation_suite.py`

验收标准：

1. 一张 ablation summary table。
2. 至少一个“没有该模块时失败模式如何恶化”的定量图。

#### E. 多 opening 串行任务

建议场景等级：

1. single opening
2. double opening
3. three-to-four openings sequence

建议继续复用或改造：

1. [inverse_transport_development/experiments/generate_simple_passage_scene.py](/home/eugene/Payload_Model/RL_Enhanced/inverse_transport_development/experiments/generate_simple_passage_scene.py)
改造方向：支持不同 opening 数量和组合顺序的批量生成。

验收标准：

1. 一张多 opening 任务对比表。
2. 至少一张展示 cumulative orientation change 或 total path length 的图。

### 5.3 第三优先级：有条件再补

#### F. 执行误差注入

形式可简化，不必立刻做完整 controller redesign：

1. position Gaussian noise
2. local overshoot near opening
3. yaw / roll lag

建议优先挂在 Gazebo 回放链或离线 replay 上，而不是先重构控制器。

#### G. 弱实证增强

如果时间允许，可补：

1. 用真实测量 contour 替代手工 polygon。
2. 用图像/点云重建结果做 planner 输入。
3. 用真实 tracking error profile 做 replay。

## 6. 仓库落实清单

### 6.1 建议新增文件

建议新增下列脚本与文档：

1. `inverse_transport_development/experiments/run_cep_baseline_suite.py`
2. `inverse_transport_development/experiments/run_cep_robustness_suite.py`
3. `inverse_transport_development/experiments/run_cep_ablation_suite.py`
4. `inverse_transport_development/experiments/summarize_cep_timing.py`
5. `inverse_transport_development/results/cep_revision/总结.md`

其中 `总结.md` 用于持续记录：

1. 已完成哪些实验。
2. 哪些结果能直接进论文。
3. 哪些现象需要解释或回避。

### 6.2 建议优先修改文件

1. [inverse_transport_development/experiments/run_planner_export_inference.py](/home/eugene/Payload_Model/RL_Enhanced/inverse_transport_development/experiments/run_planner_export_inference.py)
优先增加：更细的 timing、更多模式开关、结构化结果导出。

2. [inverse_transport_development/experiments/generate_simple_passage_scene.py](/home/eugene/Payload_Model/RL_Enhanced/inverse_transport_development/experiments/generate_simple_passage_scene.py)
优先增加：扰动注入、多 opening 组合、批量参数化生成。

3. [inverse_transport_development/experiments/plot_fixed_scene_uav_paths.py](/home/eugene/Payload_Model/RL_Enhanced/inverse_transport_development/experiments/plot_fixed_scene_uav_paths.py)
优先增加：统一 CEP 图表风格导出、批量对比图导出。

4. [inverse_transport_development/records/总结.md](/home/eugene/Payload_Model/RL_Enhanced/inverse_transport_development/records/总结.md)
继续记录阶段结果，但建议 CEP 重构期另外开独立 revision summary，避免和算法开发日志混在一起。

### 6.3 结果目录建议

建议单独建立一个 CEP 改稿结果目录：

1. `inverse_transport_development/results/cep_revision/baselines/`
2. `inverse_transport_development/results/cep_revision/robustness/`
3. `inverse_transport_development/results/cep_revision/ablation/`
4. `inverse_transport_development/results/cep_revision/timing/`
5. `inverse_transport_development/results/cep_revision/figures_final/`

目的：

1. 避免和当前 exploratory results 混杂。
2. 让最终论文图表和中间实验图区分清楚。

## 7. 图表重构计划

### 7.1 当前可保留图表

以下材料原则上可保留，但要重新服务于 CEP 叙事：

1. scene-to-abstraction 图
2. representative passing configuration 图
3. Gazebo traversal snapshot
4. minimum-snap derivative curves
5. 当前 summary table 的部分指标

### 7.2 必须新增图表

1. Baseline comparison summary table
2. success rate / minimum clearance / planning time 对比图
3. geometry error robustness 曲线图
4. payload size mismatch robustness 曲线图
5. time breakdown table
6. multi-opening sequential traversal result table
7. planner-to-controller interface figure

### 7.3 可选新增图表

1. Gazebo 中的 through-opening snapshot sequence
2. 扰动等级上升时的 failure mode montage
3. cable collision vs point collision 对比图

## 8. 建议执行顺序

### 第 1 阶段：先补证据，再改正文

顺序建议：

1. 跑 baseline suite
2. 跑 robustness suite
3. 跑 timing suite
4. 整理能进论文的表图

原因：

1. 没有这些结果，摘要、引言、实验和结论都写不稳。

### 第 2 阶段：重写论文骨架

顺序建议：

1. 标题
2. 摘要
3. 引言与贡献
4. 方法中的接口小节
5. 实验章节整体重排
6. 结论与 limitations

### 第 3 阶段：补图、补说明、统一语言风格

顺序建议：

1. 统一术语
2. 统一图表标题与 caption
3. 强化 practical relevance 表达
4. 检查是否仍然过度强调 RL novelty

## 9. 两周压缩执行版

如果按“尽量少返工”的策略执行，建议压缩为：

### 第 1 周

1. 增加 baseline suite。
2. 增加 robustness suite。
3. 给主规划脚本补 timing 输出。
4. 产出第一轮表图。

### 第 2 周

1. 重写标题、摘要、引言、贡献。
2. 重写实验章节结构。
3. 补 planner-to-controller interface 图。
4. 补 limitations 与 CEP 风格结论。

## 10. 最小可投版本定义

如果时间有限，以下内容必须完成后再投 CEP：

1. 标题与摘要完成重写。
2. 引言和贡献完成工程化重定位。
3. 至少 3 类 baseline comparison。
4. 至少 3 类鲁棒性实验。
5. 至少 1 张时间分解表。
6. 方法里补上 planner-to-controller interface 小节。
7. 结论里明确 limitations。

如果上述 7 条有 2 条以上没完成，建议不要急着送审。

## 11. 立即可执行的下一步

最合理的下一步不是立刻改论文措辞，而是先在仓库里做 3 件事：

1. 新建 `run_cep_baseline_suite.py`，把当前主方法、minimum-norm baseline、fixed-orientation baseline 接到同一结果格式。
2. 新建 `run_cep_robustness_suite.py`，优先实现 geometry error、payload size mismatch、initial orientation error 三种扰动。
3. 给 `run_planner_export_inference.py` 增加结构化 timing 导出，为后续实验表格直接供数。

完成这 3 件事后，再去改摘要和实验章节，整篇文章会稳很多。