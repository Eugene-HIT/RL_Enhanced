function plot_traj_derivatives_overview(matfile)
% plot_traj_derivatives_overview
% Plot x/y/z/roll and their 1st-4th derivatives in a compact 4x5 layout.

if nargin < 1
    matfile = 'corridor_export.mat';
end

S = load(matfile);
assert(isfield(S, 'traj'), 'Missing traj in %s.', matfile);

traj = S.traj;
assert(isfield(traj, 'sample_xyz') && ~isempty(traj.sample_xyz), 'traj.sample_xyz is empty.');

if isfield(traj, 'sample_t') && ~isempty(traj.sample_t)
    t_raw = traj.sample_t(:);
else
    t_raw = linspace(0, 1, size(traj.sample_xyz, 1))';
end

xyz_raw = traj.sample_xyz;

[t_unique, uniq_idx] = unique(t_raw, 'stable');
xyz_unique = xyz_raw(uniq_idx, :);

sample_count = numel(t_unique);
assert(sample_count >= 8, 'Not enough samples to compute higher derivatives.');

t = linspace(t_unique(1), t_unique(end), sample_count)';
xyz = interp1(t_unique, xyz_unique, t, 'pchip', 'extrap');
dt = mean(diff(t));

[roll_deg, ~] = build_roll_reference_deg(S, traj, t);

signals = {
    xyz(:, 1), 'x';
    xyz(:, 2), 'y';
    xyz(:, 3), 'z';
    roll_deg,  'roll'
};

col_titles = {
    'Position (xyz: m, roll: deg)', ...
    'Velocity (xyz: m/s, roll: deg/s)', ...
    'Acceleration (xyz: m/s^2, roll: deg/s^2)', ...
    'Jerk (xyz: m/s^3, roll: deg/s^3)', ...
    'Snap (xyz: m/s^4, roll: deg/s^4)'
};
row_colors = [
    0.07 0.45 0.74;
    0.85 0.33 0.10;
    0.47 0.67 0.19;
    0.49 0.18 0.56
];

fig = figure('Color', 'w', 'Position', [40 40 1680 860]);
tlo = tiledlayout(fig, 4, 5, 'Padding', 'compact', 'TileSpacing', 'compact');

for row = 1:4
    base_signal = signals{row, 1};
    label_name = signals{row, 2};
    chain = derivative_chain(base_signal, dt);

    for col = 1:5
        ax = nexttile(tlo, (row - 1) * 5 + col);
        hold(ax, 'on');
        box(ax, 'on');
        ax.LineWidth = 0.8;
        ax.TickDir = 'in';
        ax.FontName = 'Times New Roman';
        ax.FontSize = 10;

        plot(ax, t, chain{col}, 'LineWidth', 1.9, 'Color', row_colors(row, :));

        if row == 1
            title(ax, col_titles{col}, 'FontName', 'Times New Roman', 'FontSize', 10.5, 'FontWeight', 'bold');
        end

        if col == 1
            ylabel(ax, label_name, 'FontName', 'Times New Roman', 'FontSize', 11, 'FontWeight', 'bold');
        else
            ylabel(ax, '');
        end

        if row < 4
            ax.XTickLabel = [];
        end

        xlim(ax, [t(1), t(end)]);
        apply_padded_ylim(ax, chain{col});
    end
end

xlabel(tlo, 'Time(s)', 'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'bold');

exportgraphics(fig, 'traj_derivatives_overview.pdf', 'ContentType', 'vector');
exportgraphics(fig, 'traj_derivatives_overview.png', 'Resolution', 300);
disp('Saved: traj_derivatives_overview.pdf / traj_derivatives_overview.png');

end


function chain = derivative_chain(signal_in, dt)
window = choose_smooth_window(numel(signal_in));

s0 = smoothdata(signal_in(:), 'movmean', window);
s1 = smoothdata(gradient(s0, dt), 'movmean', window);
s2 = smoothdata(gradient(s1, dt), 'movmean', window);
s3 = smoothdata(gradient(s2, dt), 'movmean', window);
s4 = smoothdata(gradient(s3, dt), 'movmean', window);

chain = {s0, s1, s2, s3, s4};
end


function window = choose_smooth_window(n)
window = min( nine_or_less(n), 15 );
if mod(window, 2) == 0
    window = window - 1;
end
window = max(window, 5);
end


function value = nine_or_less(n)
value = min(n - 1, 9);
if value < 5
    value = 5;
end
end


function [roll_deg, note] = build_roll_reference_deg(S, traj, t)
roll_deg = zeros(size(t));
note = 'Roll reference unavailable; using zero roll.';

if ~isfield(S, 'keyframes') || ~isfield(S.keyframes, 'roll_wp')
    return;
end
if ~isfield(traj, 'T_per_seg') || isempty(traj.T_per_seg)
    return;
end

roll_wp_all = S.keyframes.roll_wp(:);
T_per_seg = traj.T_per_seg(:);
waypoint_times_all = [0; cumsum(T_per_seg)];

n_all = min(numel(roll_wp_all), numel(waypoint_times_all));
roll_wp_all = roll_wp_all(1:n_all);
waypoint_times_all = waypoint_times_all(1:n_all);

[roll_wp, waypoint_times] = select_roll_control_points(S, roll_wp_all, waypoint_times_all);

n = min(numel(roll_wp), numel(waypoint_times));
roll_wp = roll_wp(1:n);
waypoint_times = waypoint_times(1:n);

roll_deg_ctrl = rad2deg(roll_wp);
roll_deg = solve_sparse_roll_reference(t, waypoint_times, roll_deg_ctrl);
roll_deg = smoothdata(roll_deg, 'gaussian', choose_roll_window(numel(t)));
note = 'Roll row uses sparse control points with low-snap regularized smoothing.';
end


function apply_padded_ylim(ax, y)
y = y(isfinite(y));
if isempty(y)
    return;
end

ymin = min(y);
ymax = max(y);

if abs(ymax - ymin) < 1e-9
    delta = max(1.0, abs(ymax) * 0.1 + 1e-3);
    ylim(ax, [ymin - delta, ymax + delta]);
    return;
end

pad = 0.10 * (ymax - ymin);
ylim(ax, [ymin - pad, ymax + pad]);
end


function window = choose_roll_window(n)
window = max(21, 2 * floor(max(21, n / 10) / 2) + 1);
window = min(window, max(21, n - mod(n + 1, 2) - 1));
if mod(window, 2) == 0
    window = window - 1;
end
window = max(window, 21);
end


function lambda = choose_roll_lambda(n)
lambda = max(2e4, 1.8 * n^3);
end


function y_smooth = smooth_by_snap_penalty(y, lambda)
y = y(:);
n = numel(y);

if n < 6
    y_smooth = y;
    return;
end

e = ones(n, 1);
D4 = spdiags([e, -4 * e, 6 * e, -4 * e, e], 0:4, n - 4, n);
A = speye(n) + lambda * (D4' * D4);
y_smooth = full(A \ y);
end


function [roll_wp_sel, waypoint_times_sel] = select_roll_control_points(S, roll_wp_all, waypoint_times_all)
roll_wp_sel = roll_wp_all;
waypoint_times_sel = waypoint_times_all;

if ~isfield(S.keyframes, 'tags') || isempty(S.keyframes.tags)
    return;
end

tags = string(S.keyframes.tags(:));
tags = tags(1:min(numel(tags), numel(roll_wp_all)));

keep = (tags == "start") | (tags == "door_mid") | (tags == "goal");
keep = keep(:);

if nnz(keep) < 3
    return;
end

keep = keep(1:min(numel(keep), numel(waypoint_times_all)));
roll_wp_sel = roll_wp_all(keep);
waypoint_times_sel = waypoint_times_all(keep);

[waypoint_times_sel, uniq_idx] = unique(waypoint_times_sel, 'stable');
roll_wp_sel = roll_wp_sel(uniq_idx);
end


function y_out = taper_roll_endpoints(y_in, span)
y_in = y_in(:);
n = numel(y_in);

if n < 2 * span + 1
    y_out = y_in;
    return;
end

y_out = y_in;
w = 0.5 - 0.5 * cos(linspace(0, pi, span)');

start_ref = y_in(1);
end_ref = y_in(end);

y_out(1:span) = (1 - w) * start_ref + w .* y_in(1:span);
y_out(end - span + 1:end) = flipud(w) .* y_in(end - span + 1:end) + (1 - flipud(w)) * end_ref;
end


function span = choose_roll_taper(n)
span = max(9, round(0.08 * n));
span = min(span, floor((n - 1) / 3));
end


function y = solve_sparse_roll_reference(t_dense, t_ctrl, y_ctrl)
t_dense = t_dense(:);
t_ctrl = t_ctrl(:);
y_ctrl = y_ctrl(:);
n = numel(t_dense);

ctrl_idx = interp1(t_dense, 1:n, t_ctrl, 'nearest', 'extrap');
ctrl_idx = max(1, min(n, round(ctrl_idx)));

w = choose_roll_obs_weights(numel(ctrl_idx));
W = sparse(ctrl_idx, ctrl_idx, w, n, n);
b = zeros(n, 1);
b(ctrl_idx) = w .* y_ctrl;

e = ones(n, 1);
D4 = spdiags([e, -4 * e, 6 * e, -4 * e, e], 0:4, n - 4, n);

lambda_snap = choose_roll_lambda(n);
lambda_mag = choose_roll_magnitude_lambda(n);

A = W + lambda_snap * (D4' * D4) + lambda_mag * speye(n);
y = full(A \ b);
end


function w = choose_roll_obs_weights(n_ctrl)
w = 8 * ones(n_ctrl, 1);
if n_ctrl >= 1
    w(1) = 50;
end
if n_ctrl >= 2
    w(end) = 50;
end
end


function lambda_mag = choose_roll_magnitude_lambda(n)
lambda_mag = max(2.0, 0.015 * n);
end