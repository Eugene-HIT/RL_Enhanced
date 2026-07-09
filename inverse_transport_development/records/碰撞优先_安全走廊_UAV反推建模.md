# 碰撞优先安全走廊 UAV 反推建模

## 文档信息 / Document Info

- 创建时间 / Created: 2026-06-10
- 创建者 / Creator: Eugene
- 最后修改 / Last Modified: 2026-06-10
- 目的 / Purpose:
  - 将当前“载荷轨迹 -> wrench -> 挂点力 -> UAV 位置恢复”路线，重构为“碰撞优先、安全走廊约束、张力可行、轨迹可控”的正式数学模型。
  - Provide a research-ready mathematical formulation for collision-first UAV pose inference with safe corridors, tension feasibility, and trackability constraints.
- 主要输入 / Main Inputs:
  - 载荷轨迹与姿态、载荷所需 wrench、挂点几何、障碍物模型、无人机能力边界。
- 主要输出 / Main Outputs:
  - 可直接进入论文总结与后续实现的变量表、约束集合、优化目标与求解流程。

## 1. 研究定位

本文档对应 `inverse_transport_development` 的下一阶段：在已知载荷轨迹和场景障碍条件下，求解三架 UAV 的可行位置轨迹，使其优先满足安全避障和平台能力约束，再保证载荷所需 wrench 在张力上限内可实现，最后在可行解中优化张力与轨迹平滑性。

这一步不再等价于“由最小张力恢复 UAV 位置”。更准确地说，它是一个带几何、安全和动力学约束的多机协同轨迹规划问题，只是其输入来自当前逆向链条中的 payload 轨迹与 wrench。

## 2. 证据基础与参考文献

本文档的模型分层来自以下本地证据源：

1. `K.S2013(326)RSS(经典建模参考).pdf`
   - 用途：多 UAV 绳索悬挂刚体载荷的经典动力学建模基础。
   - 本文采用内容：载荷位姿、挂点、绳索方向、张力与载荷 wrench 的耦合层次。
2. `T.L2017(214)IEEE_TCST(几何控制).pdf`
   - 用途：几何控制与多绳索刚体载荷建模。
   - 本文采用内容：`r_i`、`q_i`、`R_L`、`W_L` 之间的几何关系，以及 `Phi T = W_L` 形式的体坐标映射。
3. `K.W2024(15)IEEE_RAL(待细看，很全).pdf`
   - 用途：给定目标 wrench 后的 cable force allocation。
   - 本文采用内容：正张力、张力上界、冗余自由度与优化分配思想。
4. `J.G2022(17)JAIS（四区，有载荷轨迹规划，无避障）.pdf`
   - 用途：负载分配与轨迹规划之间的接口。
   - 本文采用内容：张力分配不应脱离轨迹可行性单独讨论。
5. `Y.W2026(5)TranOnRobot.pdf`
   - 用途：系统级安全协同运输目标。
   - 本文采用内容：最终目标应是安全且可执行的多机运输，而不是静态分力本身。

本文档还结合了当前仓库中的本地实现证据：

1. `inverse_transport_development/src/common/rigid_body_payload.py`
   - 已实现 `x_i = x_L + R_L (r_i - L_i q_i)` 和 body-frame wrench map `Phi`。
2. `inverse_transport_development/src/cable_force_inference/point_force_allocation.py`
   - 当前仅完成“给定 wrench 的最小范数挂点力分配”，未显式引入障碍、安全区和张力上界。
3. `traj_qp_corridor.py`
   - 已实现二维 `x-z` 走廊的半空间表示 `A x <= b`，适合作为后续 UAV 安全走廊约束的工程接口参考。

## 3. 当前模型与其局限

当前代码的计算链条是：

1. 由载荷轨迹计算 `W_L(t)`。
2. 由 `W_L(t)` 分配挂点三维力 `f_i(t)`。
3. 由 `f_i(t)` 恢复绳索方向与张力：

$$
T_i(t) = \|f_i(t)\|_2,
\qquad
q_i(t) = -\frac{f_i(t)}{\|f_i(t)\|_2}.
$$

4. 由几何关系恢复 UAV 位置：

$$
x_i(t) = x_L(t) + R_L(t)\bigl(r_i - L_i q_i(t)\bigr).
$$

该路线的优点是几何清楚、实现简单，但存在三点根本局限：

1. 未建模障碍与危险区，恢复出的 `x_i(t)` 可能穿越障碍或机间相撞。
2. 未强制 `0 \le T_i \le T_i^{\max}`，只能得到一个几何上可解释的张力，而不是物理上可执行的张力。
3. 未建模速度、加速度、jerk 等能力边界，因此恢复结果未必可被飞控真实跟踪。

因此，当前模型是“力学一致性恢复”，不是“安全可执行规划”。

## 4. 建模假设

第一版建议采用以下假设，以便把问题先闭合成一个可实现、可验证的研究问题：

1. 绳索无质量、始终拉紧、不可压缩。
2. 绳长 `L_i` 已知且固定。
3. 载荷轨迹 `x_L(t), R_L(t)` 已由上游规划器给定。
4. 载荷所需 wrench `W_L(t)` 已由当前 `wrench_models.py` 计算得到。
5. 障碍物在当前规划窗口内静态已知。
6. 无人机之间采用球形安全包络近似。
7. 第一版只考虑 UAV 位置轨迹可跟踪性，不直接进入飞行器姿态控制器细节。

这些假设与当前代码能力一致，也与 `K.S2013`、`T.L2017` 一类建模文献的第一层抽象一致。

## 5. 变量与符号

对每个离散时刻 `k = 0, 1, ..., N`，定义：

- 载荷位置：`x_L^k \in \mathbb{R}^3`
- 载荷姿态旋转矩阵：`R_L^k \in SO(3)`
- 第 `i` 个挂点在载荷体坐标系中的位置：`r_i \in \mathbb{R}^3`
- 第 `i` 根绳长：`L_i > 0`
- 第 `i` 架 UAV 的位置：`x_i^k \in \mathbb{R}^3`
- 第 `i` 根绳索方向：`q_i^k \in \mathbb{S}^2`
- 第 `i` 根绳索张力：`T_i^k \ge 0`
- 载荷所需 wrench：

$$
W_L^k = \begin{bmatrix} F_L^k \\ M_L^k \end{bmatrix} \in \mathbb{R}^6
$$

- 张力向量：

$$
T^k = \begin{bmatrix} T_1^k & T_2^k & T_3^k \end{bmatrix}^\top
$$

- 安全走廊半空间：

$$
A_{i,k} x_i^k \le b_{i,k}
$$

其中 `A_{i,k} \in \mathbb{R}^{m_{i,k} \times 3}`，`b_{i,k} \in \mathbb{R}^{m_{i,k}}`。

## 6. 几何与力学约束

### 6.1 绳索几何关系

由当前项目中已实现的关系式：

$$
x_i^k = x_L^k + R_L^k \left(r_i - L_i q_i^k\right).
$$

等价地，可写为：

$$
q_i^k = \frac{x_L^k + R_L^k r_i - x_i^k}{L_i},
\qquad
\|q_i^k\|_2 = 1.
$$

这说明：

1. 若以 `q_i^k` 为变量，则 UAV 位置由几何直接恢复。
2. 若以 `x_i^k` 为变量，则 `q_i^k` 由几何计算得到。

本文档建议在外层规划中使用 `x_i^k` 作为主变量，因为障碍和碰撞约束写在位置空间中最自然。

### 6.2 载荷 wrench 可实现性

对于给定的 `q_i^k`，body frame 下的张力映射写为：

$$
W_L^k = \Phi\left(q_1^k, q_2^k, q_3^k\right) T^k,
$$

其中

$$
\Phi(q) =
\begin{bmatrix}
q_1^k & q_2^k & q_3^k \\
r_1 \times q_1^k & r_2 \times q_2^k & r_3 \times q_3^k
\end{bmatrix}
\in \mathbb{R}^{6 \times 3}.
$$

若考虑更多绳索，该式保持不变，只是列数扩展到绳索数量 `n`。当前三机三绳情况下，该映射一般并不总能对任意 `W_L^k` 精确实现，因此必须显式检查可行性。

### 6.3 张力边界

每根绳索应满足：

$$
0 \le T_i^k \le T_i^{\max}.
$$

若某一时刻不存在满足上式且满足 `\Phi(q^k) T^k = W_L^k` 的 `T^k`，则该时刻 UAV 布局不可执行。

## 7. 碰撞优先的安全约束

### 7.1 障碍膨胀与危险区

设原始障碍物集合为 `\mathcal{O}`，则第 `i` 架 UAV 的膨胀危险区定义为：

$$
\mathcal{O}_i^{\mathrm{infl}} = \mathcal{O} \oplus \mathcal{B}(r_i^{\mathrm{safe}}),
$$

其中 `\oplus` 表示 Minkowski 和，`\mathcal{B}(r)` 是半径为 `r` 的球，安全半径可拆为：

$$
r_i^{\mathrm{safe}} = r_i^{\mathrm{body}} + r_i^{\mathrm{tracking}} + r_i^{\mathrm{margin}}.
$$

这一步对应“先把几何障碍转成 UAV 不可进入的危险区”。

### 7.2 安全走廊约束

在每个时刻或时间段，为每架 UAV 构造一个局部凸安全走廊：

$$
\mathcal{C}_{i,k} = \{x \in \mathbb{R}^3 \mid A_{i,k} x \le b_{i,k} \}.
$$

对应的硬约束写为：

$$
A_{i,k} x_i^k \le b_{i,k}.
$$

这与 `traj_qp_corridor.py` 中二维 `x-z` 走廊的半空间格式一致，只是本文建议推广到三维，或在第一版中沿用 `x-z` 约束并将 `y` 方向单独限幅。

当前实现状态需要单独说明：第一版代码已支持“局部盒走廊 + 膨胀球障碍切平面裁剪”的三维近似，即对每个参考点和每个膨胀球障碍构造一个保持参考点可行的支撑半空间，再与局部盒走廊求交。这仍不是完整的多面体自由空间分解，但已经比单纯参考盒更接近“危险区驱动”的 corridor 生成。

### 7.3 机间防碰撞约束

严格形式是：

$$
\|x_i^k - x_j^k\|_2 \ge d_{ij}^{\min}, \qquad i \ne j.
$$

该约束是非凸的。第一版建议在参考轨迹 `\bar{x}_i^k` 附近线性化，写成：

$$
\hat{n}_{ij}^{k\top}(x_i^k - x_j^k) \ge d_{ij}^{\min},
$$

其中

$$
\hat{n}_{ij}^k = \frac{\bar{x}_i^k - \bar{x}_j^k}{\|\bar{x}_i^k - \bar{x}_j^k\|_2}.
$$

这就是后续序列凸化中的局部分离约束。

## 8. UAV 轨迹可控性约束

为了保证恢复出的轨迹能被飞控真实跟踪，至少应加入速度、加速度与高阶平滑性约束。

### 8.1 离散速度约束

令采样周期为 `\Delta t_k = t_{k+1} - t_k`，则可写为：

$$
\left\|\frac{x_i^{k+1} - x_i^k}{\Delta t_k}\right\|_2 \le v_i^{\max}.
$$

在 QP 中可先用逐分量上界近似：

$$
-v_i^{\max} \Delta t_k \le x_{i,\alpha}^{k+1} - x_{i,\alpha}^k \le v_i^{\max} \Delta t_k,
$$

其中 `\alpha \in \{x,y,z\}`。

### 8.2 离散加速度约束

可用二阶差分近似：

$$
\left\|\frac{x_i^{k+1} - 2x_i^k + x_i^{k-1}}{\Delta t_k^2}\right\|_2 \le a_i^{\max}.
$$

同样可先采用逐分量线性界近似。

### 8.3 jerk 或 snap 平滑项

若轨迹最终以多项式形式表示，可延续 minimum-snap 路线；若以离散点形式优化，则建议加入三阶差分二次项：

$$
J_{\mathrm{smooth}} = \sum_{k=1}^{N-2} \left\|x_i^{k+2} - 3x_i^{k+1} + 3x_i^k - x_i^{k-1}\right\|_2^2.
$$

该项不是安全硬约束，但对于“轨迹可控”非常关键。

## 9. 分层优化模型

## 9.1 核心原则

本问题不应再以“张力最小”作为一级目标，而应采用以下优先级：

1. 碰撞与安全区约束。
2. 张力可行性与张力上界。
3. UAV 轨迹可跟踪性。
4. 在上述可行前提下，再优化张力大小、偏离参考程度与平滑性。

## 9.2 内层张力可行性问题

给定外层位置 `x_i^k` 后，由几何关系得到 `q_i^k`，再求解内层问题：

$$
\begin{aligned}
\min_{T^k} \quad & \|T^k\|_2^2 \\
\text{s.t.} \quad & \Phi(q^k) T^k = W_L^k, \\
& 0 \le T^k \le T^{\max}.
\end{aligned}
$$

这里的目标只是为了在可行集中选一个最小二范数张力分配，不再承担安全优先的角色。若该问题无解，则当前外层 UAV 布局无效。

在工程实现中，也可以引入高惩罚 slack `s_W^k` 做诊断：

$$
\Phi(q^k) T^k + s_W^k = W_L^k,
$$

但最终有效解应满足 `s_W^k = 0` 或足够接近零。

对当前三机三绳第一版实现，需要额外说明一条工程放宽原则：由于 `\Phi(q) \in \mathbb{R}^{6 \times 3}`，标量张力层通常无法对任意六维 `W_L^k` 做严格精确重构，因此实现中应保留两级判定：

1. 严格可行：要求 `\|W_L^k - \Phi(q^k)T^k\|` 进入很小的绝对阈值。
2. 初始放宽可行：分别检查力残差和力矩残差的绝对量级与相对量级，例如

$$
\frac{\|F_L^k - \hat{F}_L^k\|_2}{\max(\|F_L^k\|_2, \varepsilon)} \le \eta_F,
\qquad
\frac{\|M_L^k - \hat{M}_L^k\|_2}{\max(\|M_L^k\|_2, \varepsilon)} \le \eta_M,
$$

并同时满足

$$
\|F_L^k - \hat{F}_L^k\|_2 \le \delta_F,
\qquad
\|M_L^k - \hat{M}_L^k\|_2 \le \delta_M.
$$

其中 `\hat{W}_L^k = \Phi(q^k)T^k`。这里的 `\varepsilon` 不应取机器精度，而应取工程尺度下限，例如对力和力矩分别设置 `\varepsilon_F` 与 `\varepsilon_M`，避免在目标力矩本来很小的时候被相对误差虚高放大。对第一版顺序规划，放宽阈值可以明显偏保守，例如令 `\eta_F` 处于几个百分点量级，而 `\eta_M` 与 `\delta_M` 先取更宽的工程阈值，只要力矩残差仍被一个可解释的绝对上界控制即可。这样做的目的不是把张力误差永久放宽，而是先让“安全走廊 + 张力近似可实现”链路闭合，再在后续迭代里逐步收紧到严格可行。

## 9.3 外层碰撞优先轨迹优化

以整段 UAV 位置轨迹 `X = \{x_i^k\}` 为变量，建议外层写成：

$$
\begin{aligned}
\min_X \quad &
w_{\mathrm{ref}} J_{\mathrm{ref}}(X)
+ w_{\mathrm{smooth}} J_{\mathrm{smooth}}(X)
+ w_{\mathrm{shape}} J_{\mathrm{shape}}(X)
+ w_{\mathrm{tension}} J_{\mathrm{tension}}(X) \\
\text{s.t.} \quad & A_{i,k} x_i^k \le b_{i,k}, \\
& \hat{n}_{ij}^{k\top}(x_i^k - x_j^k) \ge d_{ij}^{\min}, \\
& \|v_i^k\| \le v_i^{\max}, \\
& \|a_i^k\| \le a_i^{\max}, \\
& \text{inner-tension-feasible}(x_i^k, W_L^k) = \text{true}.
\end{aligned}
$$

其中：

- `J_ref`：偏离参考轨迹的代价，参考解可来自当前最小范数恢复结果。
- `J_shape`：编队形状正则项，用于避免三机布局退化。
- `J_tension`：张力 surrogate，只有在前述可行后才作为次级优化项。

对当前仓库中的第一版局部 QP，实现上可具体写成：

$$
J_{\mathrm{ref}}(X)
= \sum_{k=1}^{N} \sum_{i=1}^{M} \|x_i^k - \bar{x}_i^k\|_2^2,
$$

$$
J_{\mathrm{smooth}}(X)
= \sum_{k=2}^{N-1} \sum_{i=1}^{M} \|x_i^{k+1} - 2x_i^k + x_i^{k-1}\|_2^2,
$$

$$
J_{\mathrm{shape}}(X)
= \sum_{k=1}^{N} \sum_{1 \le i < j \le M}
\left\|
(x_i^k - x_j^k) - (\bar{x}_i^k - \bar{x}_j^k)
\right\|_2^2,
$$

其中 `J_shape` 直接约束 pairwise 相对位形偏离参考编队的程度，用于抑制三机布局在局部走廊内退化。

对 `J_tension`，当前第一版实现不直接对 `\Phi(q(x))T-W_L` 做单层非线性优化，而是采用顺序凸化下的局部 surrogate：

1. 在当前参考轨迹 `\bar{X}` 上先求一次内层张力可行性，得到每个时刻的残差 `r^k = W_L^k - \Phi(q(\bar{X}^k))T^k`。
2. 用该残差通过几何反馈构造一个 surrogate 位置 `\tilde{x}_i^k`。
3. 用残差大小构造样本权重 `\alpha_k \ge 0`，再在外层 QP 中加入

$$
J_{\mathrm{tension}}(X)
= \sum_{k=1}^{N} \alpha_k \sum_{i=1}^{M}
\|x_i^k - \tilde{x}_i^k\|_2^2.
$$

当前代码里，`\alpha_k` 采用按当前参考残差范数归一化后的权重，即残差越大的时刻，对 surrogate 的吸引越强。这仍是工程近似，但比“所有时刻统一同权重的位置吸引”更接近文档中 `J_tension` 只在张力困难样本上起作用的原意。

在 2026-06-16 的进一步收紧实现中，外层局部 QP 又额外加入了一层更接近正式 SCP 的局部模型：固定当前参考点处的张力解 `\bar{T}^k`，把残差

$$
r^k(x) = W_L^k - \Phi(q(x))\bar{T}^k
$$

在参考位置 `\bar{X}` 处做一阶线性化：

$$
r^k(x) \approx \bar{r}^k + J_r^k (x^k - \bar{x}^k),
$$

其中 `\bar{r}^k = r^k(\bar{x}^k)`，`J_r^k` 为对 UAV 位置的局部雅可比。于是额外引入一项局部 Gauss-Newton 型 surrogate：

$$
J_{\mathrm{tension,lin}}(X)
= \sum_{k=1}^{N} \alpha_k
\left\|
\bar{r}^k + J_r^k (x^k - \bar{x}^k)
\right\|_2^2.
$$

这一步仍然不是把 `\Phi(q(x))T=W_L` 作为单层硬约束直接并入 QP，但已经从“位置启发式 surrogate”收紧为“围绕当前参考点的残差一阶模型”。

为了保证该线性化只在局部有效域内使用，当前实现同时加入一个盒式 trust region：

$$
\|x_i^k - \bar{x}_i^k\|_{\infty} \le \Delta_{\mathrm{tr}},
$$

其中 `\Delta_{\mathrm{tr}}` 为每轮局部 QP 的信赖域半宽。这样做的目的，是避免为了压低线性化残差而一步跳出当前几何近似和张力近似的有效区域。

由于其中包含线性化防碰撞和内层可行性判断，整体上更接近 sequential convex programming，而不是单次闭式 QP。

## 10. 为什么不建议直接做“单层纯 QP”

如果把 `x_i^k` 直接作为变量，则 `q_i^k` 通过几何关系依赖于 `x_i^k`，进而 `\Phi(q^k)` 也依赖 `x_i^k`。这样张力约束

$$
\Phi(q(x)) T = W_L
$$

会变成非线性耦合。再叠加严格的机间距离约束 `\|x_i - x_j\|_2 \ge d_{\min}`，问题天然不是纯 QP。

因此，合理路线不是强行把全部约束塞进一个单层 QP，而是：

1. 以当前恢复结果为参考解。
2. 在参考解附近构造局部凸走廊和线性化分离约束。
3. 解一个局部 QP 更新 UAV 轨迹。
4. 重新检查张力可行性并更新参考。
5. 重复至收敛。

## 11. 建议的求解流程

### 步骤 1：生成参考轨迹

使用当前 `point_force_allocation.py` 的恢复结果作为初始参考轨迹 `\bar{x}_i^k`。

### 步骤 2：构造局部安全走廊

根据膨胀障碍和参考轨迹，为每架 UAV 在每个时间段生成 `A_{i,k}, b_{i,k}`。

### 步骤 3：线性化机间防碰撞

围绕 `\bar{x}_i^k` 生成分离超平面法向 `\hat{n}_{ij}^k`。

### 步骤 4：求解外层局部 QP

在速度、加速度、走廊和分离约束下，优化新的 UAV 位置轨迹。

### 步骤 5：逐时刻做张力可行性检查

对每个 `k`：

1. 由 `x_i^k` 计算 `q_i^k`
2. 解张力分配问题
3. 检查 `0 \le T_i^k \le T_i^{\max}` 是否成立

若失败，则缩小步长、收紧 corridor 或调整参考解。

### 步骤 6：收敛判据

当下列量同时满足时停止：

1. 轨迹增量小于阈值。
2. 所有时刻张力可行。
3. 所有走廊和机间碰撞约束满足。
4. 速度、加速度约束满足。

## 12. 与仓库代码的模块接口建议

建议在 `src/uav_pose_inference/` 下按以下层次实现：

1. `safe_corridor.py`
   - 输入：障碍物、UAV 安全半径、参考轨迹
   - 输出：`A_{i,k}, b_{i,k}`
2. `tension_feasibility.py`
   - 输入：`x_L^k, R_L^k, r_i, L_i, x_i^k, W_L^k, T_i^{max}`
   - 输出：`q_i^k, T_i^k, feasible, margin`
3. `uav_corridor_qp.py`
   - 输入：参考 UAV 轨迹、走廊约束、分离约束、速度加速度边界
   - 输出：更新后的 UAV 轨迹
4. `sequential_planner.py`
   - 输入：载荷轨迹、载荷 wrench、障碍物与系统参数
   - 输出：满足安全和张力约束的 UAV 位置轨迹

与现有代码的衔接关系为：

1. `wrench_models.py` 继续负责 `W_L^k`。
2. `rigid_body_payload.py` 继续提供 `r_i`、`L_i`、`Phi` 与几何关系。
3. `point_force_allocation.py` 不再作为最终 UAV 轨迹生成器，而转为初值生成或对照基线。

## 13. 论文表述建议

若后续写论文，本节建议按以下逻辑组织：

1. 先指出传统最小范数张力恢复的局限：缺少障碍、安全和可执行性约束。
2. 再给出碰撞优先的分层优化框架：安全约束优先，张力可行性次之，张力最小化最后。
3. 明确说明本方法采用外层安全走廊 QP 与内层张力可行性检查的分层架构。
4. 最后说明该架构与现有 payload 轨迹规划器和 wrench 估计器的接口天然兼容。

## 14. 当前仍待验证的问题

以下问题尚未在当前仓库中落地，需要在实现前继续确认：

1. 三机三绳在目标场景下能否对全部 `W_L^k` 提供足够的 wrench 可实现性。
2. 走廊是采用逐采样点离散约束，还是采用按时间段绑定的多项式控制点约束。
3. 机间防碰撞是使用逐轮线性化，还是引入更保守的固定分离模板。
4. `y` 方向是否需要与 `x-z` 平面 corridor 解耦，还是直接建立三维多面体。
5. 若张力可行域过小，是否需要把“载荷轨迹微调”纳入更外层共同优化。

## 15. 本文档结论

对当前项目，下一阶段的正确问题不是“依据最小张力恢复 UAV 位置”，而是：

$$
\text{在安全走廊与平台能力约束下，规划三机可行位置轨迹，使载荷所需 wrench 在张力上限内可实现，并在可行解中进一步优化张力与平滑性。}
$$

从实现角度看，这一问题最适合采用“外层局部凸安全轨迹优化 + 内层张力可行性检查”的分层序列凸化框架，而不是试图一次性写成单个纯 QP。