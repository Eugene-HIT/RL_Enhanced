function plot_corridor_export(matfile)
%PLOT_CORRIDOR_EXPORT Visualize exported corridor/keyframes/trajectory on XZ and YZ planes.
%   plot_corridor_export('corridor_export.mat')

if nargin < 1
    matfile = 'corridor_export.mat';
end

S = load(matfile);

% MATLAB loads variables directly (traj, keyframes, doors, forbidden, segments, astar_paths)
traj = S.traj;
keyframes = S.keyframes;
doors = S.doors;
forbidden = S.forbidden;
segments = S.segments;

% Normalize struct arrays possibly stored as cells
doors = unwrap_cell_struct(doors);
forbidden = unwrap_cell_struct(forbidden);
segments = unwrap_cell_struct(segments);

P = keyframes.P_wp;                 % (N,3)
roll = keyframes.roll_wp; %#ok<NASGU>
tags = string(keyframes.tags(:));  % (N,1)
hard = logical(keyframes.hard(:));

% Identify points
is_astar = tags == "astar";
is_door_hard = hard & (tags == "door_pre" | tags == "door_mid" | tags == "door_post");
is_start_goal = tags == "start" | tags == "goal";

sample_xyz = traj.sample_xyz;       % (M,3)

% -------- YX / YZ plots (stacked, tight) --------
figure('Name','YX & YZ Planes','Color','w');
tiledlayout(2,1,'TileSpacing','none','Padding','compact');

% -------- YX plot (y as horizontal axis) --------
nexttile; hold on; grid off; axis equal;

% Forbidden obstacles (XY) from poly_xz bounds + y-range
plot_forbidden_xy(forbidden);

% Door openings (XY) from poly_xz bounds + y-range
plot_doors_xy(doors);

% Corridor rectangles (XY, oriented by trajectory direction)
plot_corridors_xy(segments);

% Trajectory (y vs x)
plot(sample_xyz(:,2), sample_xyz(:,1), 'k-', 'LineWidth', 1.5, 'DisplayName','Trajectory');

% Keyframes
scatter(P(is_astar,2), P(is_astar,1), 18, [0.35 0.00 0.55], 'filled', 'DisplayName','A* points');
scatter(P(is_door_hard,2), P(is_door_hard,1), 55, [0.9 0.2 0.2], 'filled', 'DisplayName','Door hard points');
scatter(P(is_start_goal,2), P(is_start_goal,1), 40, [0.2 0.2 0.9], 'filled', 'DisplayName','Start/Goal');

[xlim_y, ylim_xz] = compute_global_limits(sample_xyz, P);
xlim(xlim_y); ylim(ylim_xz);

xlabel('$y\,(m)$','Interpreter','latex');
ylabel('$x\,(m)$','Interpreter','latex');
title('YX');
style_axes_latex(gca);
legend('Location','bestoutside');

% -------- YZ plot (y as horizontal axis) --------
nexttile; hold on; grid off; axis equal;

% Forbidden obstacles (ZY) from poly_xz bounds + y-range
plot_forbidden_zy(forbidden);

% Door openings (ZY) from poly_xz bounds + y-range
plot_doors_zy(doors);

% Corridor rectangles (ZY, oriented by trajectory direction)
plot_corridors_zy(segments);

% Trajectory (y vs z)
plot(sample_xyz(:,2), sample_xyz(:,3), 'k-', 'LineWidth', 1.5, 'DisplayName','Trajectory');

% Keyframes
scatter(P(is_astar,2), P(is_astar,3), 18, [0.35 0.00 0.55], 'filled', 'DisplayName','A* points');
scatter(P(is_door_hard,2), P(is_door_hard,3), 55, [0.9 0.2 0.2], 'filled', 'DisplayName','Door hard points');
scatter(P(is_start_goal,2), P(is_start_goal,3), 40, [0.2 0.2 0.9], 'filled', 'DisplayName','Start/Goal');

xlim(xlim_y); ylim(ylim_xz);

xlabel('$y\,(m)$','Interpreter','latex');
ylabel('$z\,(m)$','Interpreter','latex');
title('YZ');
style_axes_latex(gca);
legend('Location','bestoutside');

end

% ===== Helper functions =====
function out = unwrap_cell_struct(x)
    if iscell(x)
        out = [x{:}];
    else
        out = x;
    end
end

function plot_forbidden_xy(forbidden)
    if isempty(forbidden); return; end
    shown = false;
    for i = 1:numel(forbidden)
        f = forbidden(i);
        if ~isfield(f, 'poly_xz'); continue; end
        poly = f.poly_xz;
        if isempty(poly); continue; end
        if size(poly,2) ~= 2; continue; end
        x0 = min(poly(:,1));
        x1 = max(poly(:,1));
        y0 = f.y_min; y1 = f.y_max;
        % y is horizontal axis, x is vertical axis
        x = [y0 y1 y1 y0];
        y = [x0 x0 x1 x1];
        if ~shown
            patch(x, y, [1 0 0], 'FaceAlpha',0.10, 'EdgeColor',[0.8 0 0], 'LineWidth',1.0, 'DisplayName','Forbidden');
            shown = true;
        else
            patch(x, y, [1 0 0], 'FaceAlpha',0.10, 'EdgeColor',[0.8 0 0], 'LineWidth',1.0, 'HandleVisibility','off');
        end
    end
end

function plot_doors_xy(doors)
    if isempty(doors); return; end
    shown = false;
    for i = 1:numel(doors)
        d = doors(i);
        if ~isfield(d, 'poly_xz'); continue; end
        poly = d.poly_xz;
        if isempty(poly); continue; end
        if size(poly,2) ~= 2; continue; end
        x0 = min(poly(:,1));
        x1 = max(poly(:,1));
        y0 = d.y_min; y1 = d.y_max;
        % y is horizontal axis, x is vertical axis
        x = [y0 y1 y1 y0];
        y = [x0 x0 x1 x1];
        if ~shown
            patch(x, y, [0.95 0.95 0.85], 'FaceAlpha',0.35, 'EdgeColor',[0.8 0.8 0.6], 'LineWidth',0.8, 'DisplayName','Door');
            shown = true;
        else
            patch(x, y, [0.95 0.95 0.85], 'FaceAlpha',0.35, 'EdgeColor',[0.8 0.8 0.6], 'LineWidth',0.8, 'HandleVisibility','off');
        end
    end
end

function plot_forbidden_zy(forbidden)
    if isempty(forbidden); return; end
    shown = false;
    for i = 1:numel(forbidden)
        f = forbidden(i);
        if ~isfield(f, 'poly_xz'); continue; end
        poly = f.poly_xz;
        if isempty(poly); continue; end
        if size(poly,2) ~= 2; continue; end
        zmin = min(poly(:,2));
        zmax = max(poly(:,2));
        y0 = f.y_min; y1 = f.y_max;
        % y is horizontal axis, z is vertical axis
        z = [y0 y1 y1 y0];
        y = [zmin zmin zmax zmax];
        if ~shown
            patch(z, y, [1 0 0], 'FaceAlpha',0.10, 'EdgeColor',[0.8 0 0], 'LineWidth',1.0, 'DisplayName','Forbidden');
            shown = true;
        else
            patch(z, y, [1 0 0], 'FaceAlpha',0.10, 'EdgeColor',[0.8 0 0], 'LineWidth',1.0, 'HandleVisibility','off');
        end
    end
end

function plot_doors_zy(doors)
    if isempty(doors); return; end
    shown = false;
    for i = 1:numel(doors)
        d = doors(i);
        if ~isfield(d, 'poly_xz'); continue; end
        poly = d.poly_xz;
        if isempty(poly); continue; end
        if size(poly,2) ~= 2; continue; end
        zmin = min(poly(:,2));
        zmax = max(poly(:,2));
        y0 = d.y_min; y1 = d.y_max;
        % y is horizontal axis, z is vertical axis
        z = [y0 y1 y1 y0];
        y = [zmin zmin zmax zmax];
        if ~shown
            patch(z, y, [0.95 0.95 0.85], 'FaceAlpha',0.35, 'EdgeColor',[0.8 0.8 0.6], 'LineWidth',0.8, 'DisplayName','Door');
            shown = true;
        else
            patch(z, y, [0.95 0.95 0.85], 'FaceAlpha',0.35, 'EdgeColor',[0.8 0.8 0.6], 'LineWidth',0.8, 'HandleVisibility','off');
        end
    end
end

function plot_corridors_xy(segments)
    if isempty(segments); return; end
    shown = false;
    for i = 1:numel(segments)
        seg = segments(i);
        if isfield(seg, 'enabled') && ~seg.enabled
            continue;
        end
        if ~isfield(seg, 'obb_corners'); continue; end
        C = seg.obb_corners; % 8x3
        if isempty(C) || size(C,2) ~= 3; continue; end
        % y is horizontal axis, x is vertical axis
        pts = C(:,[2 1]);
        k = convhull(pts(:,1), pts(:,2));
        if ~shown
            patch(pts(k,1), pts(k,2), [0.65 0.90 0.65], 'FaceAlpha',0.18, 'EdgeColor',[0.2 0.6 0.2], 'LineWidth',1.0, 'DisplayName','Corridor');
            shown = true;
        else
            patch(pts(k,1), pts(k,2), [0.65 0.90 0.65], 'FaceAlpha',0.18, 'EdgeColor',[0.2 0.6 0.2], 'LineWidth',1.0, 'HandleVisibility','off');
        end
    end
end

function plot_corridors_zy(segments)
    if isempty(segments); return; end
    shown = false;
    for i = 1:numel(segments)
        seg = segments(i);
        if isfield(seg, 'enabled') && ~seg.enabled
            continue;
        end
        if ~isfield(seg, 'obb_corners'); continue; end
        C = seg.obb_corners; % 8x3
        if isempty(C) || size(C,2) ~= 3; continue; end
        % y is horizontal axis, z is vertical axis
        pts = C(:,[2 3]);
        k = convhull(pts(:,1), pts(:,2));
        if ~shown
            patch(pts(k,1), pts(k,2), [0.65 0.90 0.65], 'FaceAlpha',0.18, 'EdgeColor',[0.2 0.6 0.2], 'LineWidth',1.0, 'DisplayName','Corridor');
            shown = true;
        else
            patch(pts(k,1), pts(k,2), [0.65 0.90 0.65], 'FaceAlpha',0.18, 'EdgeColor',[0.2 0.6 0.2], 'LineWidth',1.0, 'HandleVisibility','off');
        end
    end
end

function [xlim_y, ylim_xz] = compute_global_limits(sample_xyz, P)
    y_all = [sample_xyz(:,2); P(:,2)];
    x_all = P(:,1); z_all = P(:,3);
    xz_all = [sample_xyz(:,1); sample_xyz(:,3); x_all; z_all];

    [xlim_y, ~] = compute_limits(y_all, y_all);
    [~, ylim_xz] = compute_limits(xz_all, xz_all);
end

function style_axes_latex(ax)
    set(ax, 'TickLabelInterpreter','latex', 'FontName','Times New Roman');
end

function [xlim_out, ylim_out] = compute_limits(x, y)
    x = x(:); y = y(:);
    xmin = min(x); xmax = max(x);
    ymin = min(y); ymax = max(y);
    dx = max(xmax - xmin, 1e-6);
    dy = max(ymax - ymin, 1e-6);
    pad = 0.05;
    xlim_out = [xmin - pad*dx, xmax + pad*dx];
    ylim_out = [ymin - pad*dy, ymax + pad*dy];
end
