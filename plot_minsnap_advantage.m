%% plot_minsnap_advantage.m
% 三联优势对比图：位置 / 加速度 / Snap
% 读取 corridor_export.mat 中的轨迹，
% 与两种缺陷拟合方式对比（稀疏线性插值 + 低阶全局多项式拟合）。

clear; clc; close all;

%% -------- Load data --------
S = load('corridor_export.mat');
traj = S.traj;
keyframes = S.keyframes;

if isfield(traj, 'sample_t') && ~isempty(traj.sample_t)
    t = traj.sample_t(:);
else
    M = size(traj.sample_xyz, 1);
    t = linspace(0, 1, M)';
end

xyz = traj.sample_xyz;

% 选一个方向（默认用 y，若想用 x 或 z 请改索引）
axis_id = 2; % 1:x, 2:y, 3:z
p_ms = xyz(:, axis_id);

% 处理重复时间戳，保证单调唯一
[t, uniq_idx] = unique(t, 'stable');
p_ms = p_ms(uniq_idx);

% 统一到等时间间隔，降低数值求导尖刺
p_count = numel(t);
t_uniform = linspace(t(1), t(end), p_count)';
p_ms = interp1(t, p_ms, t_uniform, 'linear', 'extrap');
t = t_uniform;

assert(numel(t) == numel(p_ms), 't and position size mismatch.');

%% -------- Build baselines from A* points (intentionally imperfect) --------
tags = string(keyframes.tags(:));
P = keyframes.P_wp;

astar_pts = P(tags == "astar", :);
start_pt = P(tags == "start", :);
goal_pt  = P(tags == "goal", :);

if isempty(astar_pts)
    astar_pts = P;
end

% 取较少的 A* 点以放大缺陷（同时补上起点/终点）
M = size(astar_pts, 1);
target_num = min(12, max(6, M));
step = max(1, ceil(M / target_num));
astar_sub = astar_pts(1:step:end, :);

pts = [start_pt; astar_sub; goal_pt];
if isempty(start_pt)
    pts = [astar_sub(1, :); pts];
end
if isempty(goal_pt)
    pts = [pts; astar_sub(end, :)];
end

% 按 y 方向排序（场景中 y 单调）并去重
[~, ord] = sort(pts(:,2));
pts = pts(ord, :);
[~, ia] = unique(pts(:,2), 'stable');
pts = pts(ia, :);

% 以路径弧长参数化，并映射到时间 [0,1]
ds = sqrt(sum(diff(pts, 1, 1).^2, 2));
s = [0; cumsum(ds)];
s = s / max(s(end), 1e-9);

t_norm = (t - t(1)) / max(t(end) - t(1), 1e-9);
t_norm = min(max(t_norm, 0), 1);
p_astar = pts(:, axis_id);

% Baseline-1: 稀疏线性插值（转折处明显、不够平滑）
p_lin = interp1(s, p_astar, t_norm, 'linear', 'extrap');

% Baseline-2: PCHIP 插值（避免过冲但仍可能欠平滑）
p_spline = interp1(s, p_astar, t_norm, 'pchip');

%% -------- Numerical derivatives (robust) --------
if numel(t) > 2
    dt = mean(diff(t));
else
    dt = 1.0;
end

% 轻微平滑后再求导，减弱数值尖刺
smooth_k = 7; % 奇数窗口
smooth = @(s) movmean(s, smooth_k);

d1 = @(s) gradient(s, dt);
d2 = @(s) d1(d1(s));
d4 = @(s) d2(d2(s));

sm_ms  = smooth(p_ms);
sm_lin = smooth(p_lin);
sm_spl = smooth(p_spline);

snap_ms  = d4(sm_ms);
snap_lin = d4(sm_lin);
snap_spl = d4(sm_spl);

%% -------- Plot (Single panel: Snap) --------
fig = figure('Color','w','Position',[200 80 850 350]);

hold on; grid off; box on; 
set(gca,'LineWidth',0.8,'TickDir','in','TickLength',[0.005 0.005]);

% IEEE颜色盘: Blue, Red/Orange, Yellow/Orange
c_ms  = [0 0.4470 0.7410];
c_lin = [0.8500 0.3250 0.0980];
c_spl = [0.9290 0.6940 0.1250];

p1 = plot(t, snap_ms, '-', 'Color', c_ms, 'LineWidth', 2.0);
p2 = plot(t, snap_lin, '-', 'Color', c_lin, 'LineWidth', 1.8);
p3 = plot(t, snap_spl, '-', 'Color', c_spl, 'LineWidth', 1.5);

ylabel('Snap (/s^4$)','Interpreter','latex', 'FontSize', 12);
xlabel('Time ($)','Interpreter','latex', 'FontSize', 12);

% 图例设置: 紧凑有边框的样式，如参考图所示
lgd = legend([p1, p2, p3], {'MinSnap (planned)', 'A* linear fit', 'A* cubic spline'}, 'Location', 'northeast');
set(lgd, 'Interpreter', 'latex', 'FontSize', 11, 'Box', 'on', 'EdgeColor', [0.4 0.4 0.4]);

t_range = t(end) - t(1);
xlim([t(1) + 0.05*t_range, t(end) - 0.05*t_range]);

% 动态拉高y轴顶端，防止曲线遮挡到右上角的legend
y_lims = ylim;
ylim([y_lims(1)*1.05, y_lims(2)*1.35]);

% 坐标轴数字字体设置
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

%% -------- Export --------
exportgraphics(fig, 'minsnap_advantage.pdf', 'ContentType','vector');
exportgraphics(fig, 'minsnap_advantage.png', 'Resolution', 300);
disp('Saved: minsnap_advantage.pdf / minsnap_advantage.png');
