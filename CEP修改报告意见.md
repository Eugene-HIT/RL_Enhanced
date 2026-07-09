# CEP修改报告意见

## 1. 文档目的

本文档用于基于当前论文版本，系统总结若改投 Control Engineering Practice（CEP）时建议开展的修改方向、实验补强内容与论文重构重点。以下意见严格结合现有稿件 [Paper.tex](Paper.tex) 的实际内容展开，不按空白状态重新规划，而是在当前已有结果基础上判断“还缺什么”“应如何补”“补到什么程度更贴合 CEP 审稿偏好”。

## 2. 当前稿件基础与已有内容

从当前稿件 [Paper.tex](Paper.tex) 来看，论文已经具备较完整的任务定义、方法链条与初步实验基础，并不是从零起步。现阶段已经完成的核心内容主要包括以下几个方面。

### 2.1 已完成的方法框架

当前论文已经形成了较清晰的两阶段层级规划流程：

1. 在狭窄开口局部截面上，为刚性吊载求解 passing configuration，也就是通过位置与平面朝向。
2. 在得到各关键 opening 的 passing configuration 后，再进行 A* 引导、safety corridor 构造与 minimum-snap 三维轨迹生成。

这一结构已经不仅仅是单独的 RL 决策模块，而是一个完整的 payload-level planning pipeline。这一点对 CEP 是有价值的，因为 CEP 更看重工程流程和系统链条，而不是单独的算法技巧。

### 2.2 已完成的问题建模

当前稿件已经完成以下建模工作：

1. 给出了载荷层的动力学背景描述，说明问题从多 UAV 吊载运输中抽象而来。
2. 将 narrow-opening traversal 的局部通过问题抽象为二维截面中的 position-orientation 规划问题。
3. 将 passing configuration selection 建模为单步 MDP。
4. 给出了 reward 设计，包含 clearance、中心偏移惩罚和姿态变化惩罚。
5. 将下游轨迹生成组织为 A* + corridor + QP minimum-snap 的三维规划流程。

说明当前稿件的方法部分已经有较完整的数学结构，后续改 CEP 时不需要推翻，只需要调整定位并补足工程验证。

### 2.3 已完成的实验基础

当前稿件已经包含以下实验或结果展示基础。

1. 在 Gazebo 中构建了简化的地铁坍塌救援场景。
2. 从该场景中抽取了四类代表性瓶颈 opening，并在文中给出了 scene-to-abstraction 对应关系。
3. 训练阶段使用随机 opening 分布，测试阶段使用未见 opening，已经具备一定泛化验证基础。
4. 给出了 representative passing configuration 的定性展示。
5. 给出了三维场景中的通过快照结果。
6. 给出了轨迹位置、速度、加速度、jerk、snap 曲线结果。
7. 给出了 summary table，包含：
   - 300 个测试场景
   - 成功率 97%
   - 平均推理时间 0.58 ms
   - 95 分位推理时间 0.66 ms
   - passing angle 分布统计
   - reward 统计

这意味着论文并非缺少实验，而是现有实验更多证明“方法能工作”，尚未充分证明“该方法在工程条件下有足够说服力”。这正是 CEP 修改的核心切入点。

## 3. 当前稿件与 CEP 取向的主要差距

CEP 的关注重点通常不是算法本身是否新奇，而是所提方法是否面向真实工程问题、验证是否体现 practical relevance、结果是否足够支持部署合理性。结合当前稿件，主要差距集中在以下几方面。

### 3.1 对比基线不足

当前稿件主要展示了本方法自身的效果，但缺少与合理替代方案的系统比较。现有结果能说明“方法可行”，但还不能充分回答以下 CEP 审稿人常见问题：

1. 为什么必须做 passing configuration planning？
2. 为什么不能直接使用固定姿态或简单启发式？
3. 为什么要使用当前方法，而不是传统几何搜索或优化？

也就是说，当前论文缺少“比别人强在哪里”的工程证据。

### 3.2 非理想工况验证不足

当前稿件在 [Paper.tex](Paper.tex) 中的 Note to Practitioners 已经明确写出：

1. 当前假设 obstacle geometry 已知。
2. 当前假设下游控制执行兼容。
3. 未来才考虑 perception uncertainty、multi-UAV coupling 和 hardware validation。

这虽然是诚实的边界说明，但从 CEP 视角看，也直接暴露出问题：现阶段验证大多仍处于理想几何和理想执行条件下，缺少误差、扰动和非理想执行下的表现评估。

### 3.3 工程接口说明偏弱

当前稿件已经说明自己是 payload-level planner，但还没有把该层与下层执行系统的接口讲得足够具体。CEP 更偏好看到以下问题被明确回答：

1. 上层到底输出什么给下层？
2. 下层需要满足哪些跟踪或姿态执行条件？
3. 失败模式主要来自哪里？
4. 当前方法在实际部署链条中处于什么位置？

如果这些接口不明确，审稿人容易把工作判断为“理想化规划示意”，而非“工程上可集成的模块”。

### 3.4 论文叙述仍偏 RL 论文风格

当前标题、摘要和正文中的主叙述仍将 reinforcement learning 放在比较核心的位置。对于 CEP，这样的写法容易让论文显得像“一个 RL 规划方法”而不是“一个面向工程任务的分层规划方案”。

CEP 一般更接受如下定位：

1. 面向某类工程任务的系统性解决方案。
2. RL 只是其中一个用于提升在线决策效率的技术模块。
3. 重点放在任务适配性、工程可执行性和非理想条件下的可用性。

## 4. 总体修改方向

如果按 CEP 思路改稿，建议总体上完成一次“论文定位重心”的转移，但不需要推翻现有内容。建议将整篇稿件从“RL 求 through-opening pose 的方法论文”转成“面向狭窄开口吊载运输的工程化分层规划方案”。

具体建议如下：

1. 保留现有 passing configuration 规划与三维轨迹生成的两阶段结构。
2. 弱化 RL 本身的新颖性叙述，将其定位为一个高效求解局部通过构型的实现模块。
3. 强化 narrow-opening transportation 作为工程瓶颈问题的任务背景。
4. 强化从 passing configuration 到 trajectory generation 再到 downstream control 的系统链条叙述。
5. 将实验重点从“方法是否成功”调整为“工程上是否可信、是否鲁棒、是否具有实时可用性”。

## 5. 论文内容层面的具体修改建议

### 5.1 标题修改建议

当前标题突出了 reinforcement learning。若投 CEP，建议在标题层面弱化 RL 的中心地位，更强调任务和系统。

可以考虑的修改方向包括：

1. 面向狭窄开口吊载运输的分层通过构型规划与轨迹生成。
2. 面向受限环境吊载运输的通过姿态选择与安全轨迹规划。
3. 如果保留 RL，可将其放在次级修饰位置，而不是标题主干中心。

目标不是去掉 RL，而是避免标题让审稿人一眼把稿件归类为“纯 RL 方法文”。

### 5.2 摘要修改建议

当前摘要已经完成了以下功能：

1. 说明了狭窄 opening 下的 through-passage 问题。
2. 说明了 low-dimensional geometric abstraction 和 RL-based passing configuration selection。
3. 说明了后续三维 minimum-snap trajectory generation。
4. 报告了 success rate、毫秒级推理时间和 smooth traversal。

若改投 CEP，摘要建议补足两类信息：

1. 增加更明确的工程背景语义，例如 rescue、inspection、confined transportation 中的 narrow-opening bottleneck。
2. 增加与基线比较或非理想验证结果的概述，例如在几何扰动、尺寸误差、执行偏差下仍保持较高成功率与安全裕度。

CEP 风格的摘要不应只说“方法有效”，还应简要说明“在更贴近工程的条件下为何仍可信”。

### 5.3 引言与贡献重写建议

当前引言在相关工作综述上较充分，但 CEP 并不特别需要过长的控制背景罗列。建议缩减纯学术铺垫，将更多篇幅集中在工程问题本身。

引言中建议更突出以下几点：

1. narrow-opening traversal 是吊载运输中的关键 bottleneck。
2. 单纯 tracking、anti-swing 和稳定控制无法自动解决“局部瓶颈处如何选姿态”的问题。
3. passing configuration 是连接环境几何约束与下游控制执行之间的重要中间层变量。
4. 该问题若处理不好，即使下层控制再精确，也可能在瓶颈处失败。

贡献建议改写为更贴合 CEP 的三点：

1. 提出一个面向狭窄开口吊载运输任务的 payload-level hierarchical planning pipeline。
2. 提出一个适用于 irregular polygonal openings 的 passing configuration selection 模块，并将其与三维安全轨迹生成整合。
3. 在 representative rescue-like scene 及非理想扰动条件下验证其实时性、通过成功率、几何安全裕度和轨迹执行友好性。

### 5.4 方法部分补充建议

当前方法部分的主体结构可以保留，但建议增加一个更明确的工程接口说明小节，例如：

1. Planner-to-controller interface
2. Practical execution assumptions
3. Deployment considerations

建议在这一小节中明确写清：

1. 上层 planner 输出的内容：
   - 每个 opening 的 passing position 和 planar orientation
   - 三维 waypoint sequence 或 minimum-snap position trajectory
   - sparse roll reference
2. 下层执行系统需满足的条件：
   - 基本轨迹跟踪能力
   - 有限姿态调整能力
   - 能够获取或估计开口几何
3. 当前未显式建模的因素：
   - perception error
   - geometry reconstruction error
   - load swing during execution
   - wind disturbance 或 cable deformation
4. 当前工作在系统中的角色：
   - 提供任务层 through-opening configuration and trajectory reference
   - 不替代低层控制器，而是为其提供更合理的 through-bottleneck keyframe

这一补充对 CEP 很重要，因为它直接提升论文的系统集成可信度。

## 6. 实验修改与补强建议

这是整篇论文最需要加强的部分。结合当前已有实验基础，建议新增实验分为“必须补”“建议补”“有条件再补”三类。

### 6.1 必须补：基线对比实验

当前稿件最需要补的就是基线比较，而且这部分可以直接建立在现有 Gazebo 场景、随机 opening 生成器和 passing configuration 规划框架上，不需要完全重做实验平台。

建议至少设置以下三类 baseline。

#### 6.1.1 固定姿态基线

设置方式：

1. payload 在 through-opening 时保持固定 orientation。
2. 仅允许位置通过 opening center 或局部可行区域调整。

用途：

1. 证明“通过姿态选择”本身是必要的。
2. 说明单纯平移而不调姿态在 irregular opening 下会显著降低成功率或安全裕度。

#### 6.1.2 几何启发式基线

设置方式：

1. passing position 取 opening center 或 clearance heuristic center。
2. passing orientation 采用简单几何规则，例如 opening 主方向对齐、最小外接矩形主轴对齐或最窄边法向规整。

用途：

1. 证明规则法在 irregular polygonal opening 下不够稳定。
2. 显示当前方法在复杂开口几何下具有更稳定的 clearance 表现。

#### 6.1.3 采样搜索或传统优化基线

设置方式：

1. 对 position + orientation 做网格采样、随机采样或简化数值优化。
2. 以 clearance 最大化为目标选解。

用途：

1. 证明当前 RL 模块的价值不只是“能找到可行解”。
2. 重点体现其在接近优化效果下具有更好的在线推理速度。

#### 6.1.4 建议比较指标

基线对比建议统一报告以下指标：

1. 成功率
2. 最小 clearance
3. 平均 clearance
4. 单 opening 规划时间
5. passing angle 变化幅度
6. 整体路径长度
7. 轨迹平滑度代理量，例如最大 jerk 或 snap

其中最关键的是：成功率、最小 clearance 和规划时间。

### 6.2 必须补：鲁棒性与敏感性实验

这部分是最符合 CEP 审稿取向的补强内容。因为当前论文最大的风险不是“完全没有实验”，而是“实验过于理想化”。在没有硬件实验的情况下，鲁棒性实验是最有效的替代证据。

建议在当前环境基础上增加以下几类非理想因素。

#### 6.2.1 Opening 几何边界误差

实现建议：

1. 对 opening polygon 顶点位置加入随机偏差。
2. 规划使用带误差的 opening geometry，执行碰撞判定使用真实 opening。

建议扰动水平：

1. 小扰动：1% 到 3% characteristic size
2. 中扰动：5%
3. 大扰动：8% 到 10%

评估指标：

1. success rate
2. minimum clearance
3. infeasible rate
4. collision-after-execution rate

该实验可直接对应实际中的 perception error 或 map reconstruction error。

#### 6.2.2 Payload 尺寸估计误差

实现建议：

1. 规划阶段使用带偏差的 payload width/height。
2. 执行阶段使用真实 payload 包络做碰撞判定。

建议误差水平：

1. ±2%
2. ±5%
3. ±8%

评估重点：

1. 方法对尺寸不确定性的敏感程度
2. 最小 clearance 是否迅速塌陷
3. 是否存在明显安全边界不足问题

这类实验非常贴合工程实际，因为载荷外廓和 sling-induced envelope 往往并非完全精确已知。

#### 6.2.3 初始姿态偏差

实现建议：

1. 在 approaching bottleneck 前，令当前 payload orientation 与 state prior 之间存在偏差。
2. 偏差可通过随机角度注入实现。

建议偏差水平：

1. ±5°
2. ±10°
3. ±15°

评估重点：

1. RL policy 是否过度依赖理想 prior
2. orientation regularization 是否仍然能保持合理 through-opening behavior

#### 6.2.4 轨迹执行误差或跟踪偏差

实现建议：

这项不一定需要完整重构下层控制器。可在现有 Gazebo 执行或离线回放层面人为注入简化执行误差，例如：

1. 位置执行高斯噪声
2. 转弯阶段固定滞后
3. opening 附近局部 overshoot
4. yaw 或 roll 跟踪延迟

评估重点：

1. 通过成功率下降程度
2. 最小安全裕度下降幅度
3. 哪类 bottleneck 对执行误差最敏感

这是把“理想轨迹生成”往“执行层可承受性”推进的一步，对 CEP 很有帮助。

#### 6.2.5 场景复杂度变化

实现建议：

在现有 Gazebo rescue-like scene 和随机 opening 框架上增加：

1. 更高 opening irregularity
2. 更小 opening size margin
3. 更高 obstacle density
4. 更多连续 bottleneck 数量

用途：

1. 验证方法不是只对当前 4 个代表性 opening 有效。
2. 体现 scalability 和 generalization。

#### 6.2.6 建议输出形式

建议最终以以下形式呈现鲁棒性结果：

1. success rate vs disturbance level 曲线
2. minimum clearance vs disturbance level 曲线
3. 关键扰动条件下的汇总表

### 6.3 必须补：工程时延与复杂度评估

当前稿件只有 RL 推理时间统计，这只能说明 passing configuration selection 很快，还不能说明整个 pipeline 工程上是否高效。

建议将总耗时拆分为：

1. passing configuration selection time
2. A* planning time
3. safety corridor construction time
4. QP solving time
5. total planning time

然后在不同复杂度场景下分别统计：

1. 单 opening 场景
2. 多 opening 串行场景
3. 低复杂度障碍场景
4. 中高复杂度障碍场景

该部分的核心目的在于回答：

1. 当前方案是否只在局部决策层面快，而整体系统并不快。
2. 当场景复杂度提升时，哪一模块成为主要瓶颈。
3. 是否具备在线或准在线规划潜力。

建议增加一张 time breakdown 表，以及一张 complexity trend 图。

### 6.4 建议补：消融实验

消融实验不是最刚性的门槛，但对增强论文说服力很有价值，而且与你当前方法设计逻辑高度一致，较容易实现。

建议至少做以下三个消融。

#### 6.4.1 去掉 orientation prior

目的：

1. 验证 state 中加入 θ_{k-1} 是否有助于姿态连续性。
2. 检查多 opening 串行场景下的 cumulative orientation change 是否恶化。

#### 6.4.2 去掉 center penalty

目的：

1. 验证 reward 中 position regularization 是否有必要。
2. 检查是否会出现偏置过大、虽然局部 clearance 可接受但全局 through-path 不合理的解。

#### 6.4.3 去掉 safety corridor，仅做 minimum-snap

目的：

1. 与当前文中已有的 naive failure illustration 形成定量对应。
2. 证明 corridor constraints 对 cluttered environment 下的 collision avoidance 是必要的。

建议消融指标：

1. success rate
2. minimum clearance
3. angle variation
4. trajectory feasibility rate
5. collision ratio

### 6.5 建议补：多 opening 串行任务实验

当前稿件虽然已经提到 multiple openings，并在 scene snapshot 中有所体现，但最好形成一个明确的任务组实验，而不是仅停留在示意层面。

建议设置以下任务等级：

1. 单 opening 穿越
2. 双 opening 串行穿越
3. 三到四 opening 串行穿越

在这些任务上建议统计：

1. 成功率
2. 总规划时间
3. 累积姿态变化
4. 总路径长度
5. 最小安全裕度

这部分实验尤其适合体现你当前方法中 orientation prior 的实际价值，也更符合 CEP 对 sequential engineering tasks 的偏好。

### 6.6 有条件再补：弱实证或半实物增强验证

如果后续条件允许，但暂时做不了真机实验，也可以增加一些“弱实证”内容增强 CEP 适配性，例如：

1. 使用真实测量的 opening contour 替代纯手工 polygon。
2. 通过真实图像或点云重建 opening 截面后输入 planner。
3. 用真实采样的 tracking error 模型做离线 replay。

这些内容即使不构成 full hardware experiment，也能显著增强论文的 practice 属性。

## 7. 推荐的实验章节重构方式

建议将当前实验章节从“定性展示 + summary table”的结构，升级为更贴合 CEP 的系统验证结构。推荐组织如下：

1. Experimental setup
2. Baseline comparison
3. Robustness under non-ideal conditions
4. Planning efficiency and scalability
5. Ablation study
6. Sequential multi-opening traversal

这样做的好处在于：

1. 实验逻辑更加完整。
2. 更容易回应 CEP 审稿人关于工程可信性的关注。
3. 有利于将当前已有图表保留并在其基础上扩展比较图和统计图。

## 8. 基于当前论文基础的最实际补实验路线

如果按“工作量尽量可控、增益尽量大”的原则排序，建议优先顺序如下。

### 8.1 第一优先级

这部分最值得先做，因为都可以直接建立在现有环境之上：

1. 三类 baseline comparison
2. opening geometry error robustness
3. payload size mismatch robustness
4. time breakdown 统计

完成这四项后，论文从“方法演示”升级到“有比较、有扰动、有系统时延”的 CEP 初步形态。

### 8.2 第二优先级

如果时间和实验条件允许，建议继续补：

1. orientation prior ablation
2. unconstrained minimum-snap 对比
3. 多 opening 串行任务统计

这三项可以显著增强论文结构完整性，并自然支撑方法设计中的几个关键选择。

### 8.3 第三优先级

有余力时再补：

1. tracking/execution error 注入
2. 弱实证输入数据
3. 更复杂救援场景扩展

## 9. 图表补充建议

当前稿件已经具备以下图表：

1. 问题示意图
2. 方法框架图
3. naive minimum-snap failure 图
4. scene-to-abstraction 图
5. Gazebo 穿越快照图
6. 导数曲线图
7. summary metrics 表

这些图表可以保留，但若投 CEP，建议再增加：

1. baseline comparison 总表
2. robustness 曲线图
3. time breakdown 表
4. 多 opening 任务对比表
5. planner-to-controller interface 图

建议图表目标从“展示方法工作过程”转向“支撑工程判断与比较结论”。

## 10. 结论与讨论部分的修改建议

当前结论已经概括了方法和现有仿真结果，但 CEP 版本建议进一步强调以下三点：

1. 本文解决了什么工程瓶颈问题。
2. 在哪些 realistic but still simulated conditions 下验证了方法。
3. 当前尚未覆盖的边界条件有哪些。

建议 limitation 写得更具体，而不是只说“未来做硬件”。例如可以明确写出：

1. 当前仍假设 opening geometry 可获得。
2. 当前未显式建模 perception-to-planning uncertainty 闭环。
3. 当前未在真机平台上验证多 UAV 吊载协同执行。
4. 后续将扩展至 sensor-in-the-loop 或 hardware validation。

这种写法更符合 CEP 风格，因为它体现了工程边界的明确认识，而不是笼统留给未来工作。

## 11. 对整篇稿件的总修改策略总结

最核心的调整不是重写算法，而是重写论文的重心。

建议将稿件总体定位调整为：

“面向狭窄开口吊载运输的分层规划系统，在 representative rescue-like scene 和多种非理想扰动条件下的工程验证。”

在这个框架下：

1. RL 是实现 passing configuration selection 的技术手段之一。
2. 真正的论文主线应是 through-opening bottleneck 的系统级解决方案。
3. 实验主线应是工程可信性，而不是单纯成功展示。

## 12. 最终建议执行清单

为便于后续和导师讨论，建议将修改任务压缩为如下可执行清单。

1. 重写标题、摘要、贡献，弱化 RL novelty，强化 engineering pipeline 定位。
2. 在方法部分增加 planner-to-controller interface 与 practical assumptions 说明。
3. 增加三类 baseline 对比：固定姿态、几何启发式、采样或传统优化。
4. 增加至少三类鲁棒性实验：opening geometry error、payload size mismatch、initial orientation error。
5. 增加模块级 time breakdown 与复杂度统计。
6. 增加若干消融实验，重点验证 orientation prior、reward regularization 与 safety corridor 的必要性。
7. 增加多 opening 串行任务统计，突出 sequential traversal 能力。
8. 重写结论和 limitation，使其更符合 CEP 的工程表达方式。

## 13. 一句话结论

当前稿件已经具备投 CEP 的雏形，但必须从“方法可行性展示”进一步提升为“工程可信性验证”。最关键的补强不是再扩展算法，而是增加基线对比、非理想条件鲁棒性、系统时延评估和更明确的工程接口说明。