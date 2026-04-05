function plot_export_scene_3d_clean()
clc; clear; close all;

MAT_PATH = 'corridor_export.mat';

% =========================================================
% 1) Robust loading
% =========================================================
S = load(MAT_PATH);

if isfield(S, 'traj')
    traj = S.traj;
    keyframes = getFieldOr(S, 'keyframes', []);
    doors = getFieldOr(S, 'doors', []);
    forbidden = getFieldOr(S, 'forbidden', []);
elseif isfield(S, 'export')
    export = S.export;
    traj = export.traj;
    keyframes = getFieldOr(export, 'keyframes', []);
    doors = getFieldOr(export, 'doors', []);
    forbidden = getFieldOr(export, 'forbidden', []);
else
    error('MAT file has neither top-level "traj" nor "export".');
end

% unwrap possible cell arrays
keyframes = unwrap_cell_struct(keyframes);
doors = unwrap_cell_struct(doors);
forbidden = unwrap_cell_struct(forbidden);

% =========================================================
% 2) Extract trajectory arrays
% =========================================================
xyz = getFieldOr(traj, 'sample_xyz', []);
if isempty(xyz)
    error('traj.sample_xyz is empty.');
end
x = xyz(:,1); y = xyz(:,2); z = xyz(:,3);

% =========================================================
% 3) Doors info
% =========================================================
door_polys = {};
door_ymin  = [];
door_ymax  = [];
door_cx    = [];
door_cz    = [];

if ~isempty(doors)
    nDoors = numel(doors);
    door_polys = cell(nDoors,1);
    door_ymin  = zeros(nDoors,1);
    door_ymax  = zeros(nDoors,1);
    door_cx    = zeros(nDoors,1);
    door_cz    = zeros(nDoors,1);

    for i = 1:nDoors
        di = doors(i);
        poly_xz = getFieldOr(di, 'poly_xz', []);
        if isempty(poly_xz); poly_xz = zeros(0,2); end
        door_polys{i} = poly_xz;

        door_ymin(i) = getFieldOr(di, 'y_min', 0);
        door_ymax(i) = getFieldOr(di, 'y_max', 0);

        cx_i = getFieldOr(di, 'door_cx', NaN);
        cz_i = getFieldOr(di, 'door_cz', NaN);
        if ~isfinite(cx_i) || ~isfinite(cz_i)
            [cx_i, cz_i] = poly_center_xz(poly_xz);
        end
        door_cx(i) = cx_i;
        door_cz(i) = cz_i;
    end
end

% =========================================================
% 4) Plot setup
% =========================================================
fig = figure('Color','w','Position',[80 80 980 720]);
ax = axes('Parent',fig); hold(ax,'on');
set(ax,'FontSize',12,'LineWidth',1.0);
box(ax,'on'); grid(ax,'off');
ax.TickDir = 'out';
ax.Projection = 'orthographic';

xlabel('$x\,(m)$','Interpreter','latex');
ylabel('$y\,(m)$','Interpreter','latex');
zlabel('$z\,(m)$','Interpreter','latex');
view(ax, 38, 22);
axis(ax,'equal');
pbaspect(ax, [1 1 0.55]);

% axis limits with padding
pad = 0.08;
lims = [min(x) max(x) min(y) max(y) min(z) max(z)];
lims(2) = lims(2) + pad * max(1e-6, lims(2)-lims(1));
lims(1) = lims(1) - pad * max(1e-6, lims(2)-lims(1));
lims(4) = lims(4) + pad * max(1e-6, lims(4)-lims(3));
lims(3) = lims(3) - pad * max(1e-6, lims(4)-lims(3));
lims(6) = lims(6) + pad * max(1e-6, lims(6)-lims(5));
lims(5) = lims(5) - pad * max(1e-6, lims(6)-lims(5));

xlim([lims(1) lims(2)]);
ylim([lims(3) lims(4)]);
zlim([lims(5) lims(6)]);

% =========================================================
% 5) Draw door walls
% =========================================================
ringMargin = 0.55;
wallAlpha  = 0.95;
wallColor  = [0.63 0.76 0.92];
edgeLW     = 1.0;
edgeColor  = [0.20 0.20 0.20];

for i = 1:numel(door_polys)
    poly = door_polys{i};
    if size(poly,1) < 3
        continue;
    end
    drawDoorRingWall(poly, door_ymin(i), door_ymax(i), ringMargin, ...
        wallColor, wallAlpha, edgeColor, edgeLW);
end

% =========================================================
% 6) Draw passing rectangles on door faces
% =========================================================
passRect_W = 0.55;
passRect_H = 0.35;
passColor  = [0.15 0.40 0.95];
passAlpha  = 0.90;
passEdge   = [0.05 0.05 0.10];
passEdgeLW = 1.2;

% compute per-door pose from keyframes when possible
[theta_per_door, cx_per_door, cz_per_door] = compute_door_pose(keyframes, numel(door_polys), door_cx, door_cz);

for i = 1:numel(door_polys)
    if size(door_polys{i},1) < 3
        continue;
    end
    yFront = door_ymin(i) - 1e-3;
    c = [cx_per_door(i), yFront, cz_per_door(i)];
    theta_i = theta_per_door(i);
    drawPassingRectOnDoorFace(c, theta_i, passRect_W, passRect_H, passColor, passAlpha, passEdge, passEdgeLW);
end

% =========================================================
% 7) Forbidden prisms (optional, very light)
% =========================================================
if ~isempty(forbidden)
    forbColor = [0.20 0.20 0.20];
    forbAlpha = 0.08;
    forbEdge  = [0.25 0.25 0.25];
    forbLW    = 0.8;

    for i = 1:numel(forbidden)
        fi = forbidden(i);
        polyXZ = getFieldOr(fi,'poly_xz',[]);
        if isempty(polyXZ) || size(polyXZ,1) < 3
            continue;
        end
        fy0 = getFieldOr(fi,'y_min',0);
        fy1 = getFieldOr(fi,'y_max',0);
        drawPrismSolid(polyXZ, fy0, fy1, forbColor, forbAlpha, forbEdge, forbLW);
    end
end

% =========================================================
% 8) Trajectory
% =========================================================
plot3(ax, x, y, z, '-', 'LineWidth', 2.6, 'Color', [0.10 0.10 0.10]);
plot3(ax, x(1), y(1), z(1), 'o', 'MarkerSize',7, 'LineWidth',1.4, 'Color', [0.10 0.10 0.10]);
plot3(ax, x(end), y(end), z(end), 's', 'MarkerSize',7, 'LineWidth',1.4, 'Color', [0.10 0.10 0.10]);

% Keyframes (light)
if ~isempty(keyframes) && isfield(keyframes,'P_wp')
    P_wp = keyframes.P_wp;
    if ~isempty(P_wp)
        plot3(ax, P_wp(:,1), P_wp(:,2), P_wp(:,3), '.', 'MarkerSize',10, 'Color', [0.15 0.15 0.15]);
    end
end

% lighting for nicer look
camlight headlight;
material(ax, 'dull');

exportgraphics(fig,'scene_3d_result_clean.pdf','ContentType','vector');
disp('[ok] saved scene_3d_result_clean.pdf');

end

% ===================== helpers ============================

function v = getFieldOr(s, name, defaultVal)
v = defaultVal;
try
    if isstruct(s)
        if isfield(s, name)
            v = s.(name);
            return;
        end
    end
    if iscell(s) && numel(s)==1 && isstruct(s{1}) && isfield(s{1},name)
        v = s{1}.(name);
        return;
    end
catch
    v = defaultVal;
end
end

function out = unwrap_cell_struct(x)
if iscell(x)
    out = [x{:}];
else
    out = x;
end
end

function [cx, cz] = poly_center_xz(polyXZ)
cx = NaN; cz = NaN;
try
    if isempty(polyXZ) || size(polyXZ,1) < 3
        return;
    end
    X = polyXZ(:,1); Z = polyXZ(:,2);
    if X(1) ~= X(end) || Z(1) ~= Z(end)
        X = [X; X(1)];
        Z = [Z; Z(1)];
    end
    pg = polyshape(X, Z);
    if pg.NumRegions < 1 || area(pg) < 1e-12
        return;
    end
    [cx, cz] = centroid(pg);
catch
    cx = NaN; cz = NaN;
end
end

function [theta_per_door, cx_per_door, cz_per_door] = compute_door_pose(keyframes, nDoors, door_cx, door_cz)
% prefer roll_wp + P_wp at door_mid keyframes
theta_per_door = zeros(nDoors,1);
cx_per_door = door_cx(:);
cz_per_door = door_cz(:);

if isempty(keyframes) || ~isfield(keyframes,'roll_wp') || ~isfield(keyframes,'tags') || ~isfield(keyframes,'door_ids') || ~isfield(keyframes,'P_wp')
    return;
end

roll_wp = keyframes.roll_wp(:);
tags = string(keyframes.tags(:));
door_ids = keyframes.door_ids(:);
P_wp = keyframes.P_wp;

for i = 1:nDoors
    did = i - 1;
    idx = find(tags == "door_mid" & door_ids == did, 1, 'first');
    if isempty(idx)
        % fallback to pre/post if mid not found
        idx = find((tags == "door_pre" | tags == "door_post") & door_ids == did, 1, 'first');
    end
    if ~isempty(idx)
        if idx <= size(P_wp,1)
            cx_per_door(i) = P_wp(idx,1);
            cz_per_door(i) = P_wp(idx,3);
        end
        if idx <= numel(roll_wp) && isfinite(roll_wp(idx))
            theta_per_door(i) = roll_wp(idx);
        end
    end
end
end

function drawPassingRectOnDoorFace(center, theta, w, h, faceColor, faceAlpha, edgeColor, edgeLW)
% rotated rectangle in x-z plane at y = center(2), rotated by theta about y-axis
cx = center(1); cy = center(2); cz = center(3);

hx = w/2; hz = h/2;
P_local = [
    -hx, 0, -hz;
     hx, 0, -hz;
     hx, 0,  hz;
    -hx, 0,  hz
];

R = [ cos(theta), 0, -sin(theta);
      0,          1,  0;
      sin(theta), 0,  cos(theta) ];

P = (R * P_local')';
P(:,1) = P(:,1) + cx;
P(:,2) = P(:,2) + cy;
P(:,3) = P(:,3) + cz;

patch('Vertices', P, 'Faces', [1 2 3 4], ...
    'FaceColor', faceColor, ...
    'FaceAlpha', faceAlpha, ...
    'EdgeColor', edgeColor, ...
    'LineWidth', edgeLW);
end

% ----------------------------------------------------------
% Draw "donut wall": outer buffer minus hole (door poly)
% ----------------------------------------------------------
function drawDoorRingWall(doorPolyXZ, ymin, ymax, margin, faceColor, faceAlpha, edgeColor, edgeLW)
x = doorPolyXZ(:,1);
z = doorPolyXZ(:,2);
if x(1) ~= x(end) || z(1) ~= z(end)
    x = [x; x(1)];
    z = [z; z(1)];
end

hole = polyshape(x, z, 'Simplify', true);
if hole.NumRegions < 1 || area(hole) < 1e-10
    return;
end

outer = polybuffer(hole, margin);
ring  = subtract(outer, hole);

if ring.NumRegions < 1 || area(ring) < 1e-10
    return;
end

TR = triangulation(ring);
pts = TR.Points;
tri = TR.ConnectivityList;

p = size(pts,1);
Vtop    = [pts(:,1), ymin*ones(p,1), pts(:,2)];
Vbottom = [pts(:,1), ymax*ones(p,1), pts(:,2)];
V = [Vtop; Vbottom];

Ftop = tri;
Fbot = tri(:, [1 3 2]) + p;
F = [Ftop; Fbot];

[bx, bz] = boundary(ring);
loops = split_nan_loops(bx, bz);

for k = 1:numel(loops)
    loop = loops{k};
    if size(loop,1) < 3
        continue;
    end
    if loop(1,1) ~= loop(end,1) || loop(1,2) ~= loop(end,2)
        loop = [loop; loop(1,:)];
    end

    for i = 1:(size(loop,1)-1)
        p1 = loop(i,:);
        p2 = loop(i+1,:);

        v1 = [p1(1), ymin, p1(2)];
        v2 = [p2(1), ymin, p2(2)];
        v3 = [p2(1), ymax, p2(2)];
        v4 = [p1(1), ymax, p1(2)];

        idx0 = size(V,1);
        V = [V; v1; v2; v3; v4];

        F = [F;
             idx0+1 idx0+2 idx0+3;
             idx0+1 idx0+3 idx0+4];
    end
end

patch('Vertices',V,'Faces',F, ...
    'FaceColor',faceColor, ...
    'FaceAlpha',faceAlpha, ...
    'EdgeColor','none');

for k = 1:numel(loops)
    loop = loops{k};
    if size(loop,1) < 3, continue; end
    if loop(1,1) ~= loop(end,1) || loop(1,2) ~= loop(end,2)
        loop = [loop; loop(1,:)];
    end
    xb = loop(:,1);
    zb = loop(:,2);

    plot3(xb, ymin*ones(size(xb)), zb, '-', 'Color', edgeColor, 'LineWidth', edgeLW);
    plot3(xb, ymax*ones(size(xb)), zb, '-', 'Color', edgeColor, 'LineWidth', edgeLW);
end
end

% split NaN-separated boundary vectors into loops (cell)
function loops = split_nan_loops(bx, bz)
bx = bx(:); bz = bz(:);
isnanv = isnan(bx) | isnan(bz);

sep = find(isnanv);
cuts = [0; sep; numel(bx)+1];

loops = {};
for i = 1:(numel(cuts)-1)
    a = cuts(i)+1;
    b = cuts(i+1)-1;
    if b - a + 1 < 3
        continue;
    end
    xx = bx(a:b);
    zz = bz(a:b);

    m = ~(isnan(xx) | isnan(zz));
    xx = xx(m); zz = zz(m);
    if numel(xx) < 3
        continue;
    end
    loops{end+1} = [xx(:), zz(:)]; %#ok<AGROW>
end
end

% ----------------------------------------------------------
% Solid prism for forbidden (no holes)
% ----------------------------------------------------------
function drawPrismSolid(polyXZ, ymin, ymax, faceColor, faceAlpha, edgeColor, edgeLW)
X = polyXZ(:,1);
Z = polyXZ(:,2);
if X(1) ~= X(end) || Z(1) ~= Z(end)
    Xc = [X; X(1)];
    Zc = [Z; Z(1)];
else
    Xc = X; Zc = Z;
end

pg = polyshape(Xc, Zc);
if pg.NumRegions < 1 || area(pg) < 1e-10
    return;
end

TR = triangulation(pg);
tri = TR.ConnectivityList;
pts = TR.Points;

p = size(pts,1);
Vtop    = [pts(:,1), ymin*ones(p,1), pts(:,2)];
Vbottom = [pts(:,1), ymax*ones(p,1), pts(:,2)];
V = [Vtop; Vbottom];

Ftop = tri;
Fbot = tri(:, [1 3 2]) + p;
F = [Ftop; Fbot];

[bx, bz] = boundary(pg);
loops = split_nan_loops(bx, bz);

for k = 1:numel(loops)
    loop = loops{k};
    if size(loop,1) < 3, continue; end
    if loop(1,1) ~= loop(end,1) || loop(1,2) ~= loop(end,2)
        loop = [loop; loop(1,:)];
    end

    for i = 1:(size(loop,1)-1)
        p1 = loop(i,:);
        p2 = loop(i+1,:);

        v1 = [p1(1), ymin, p1(2)];
        v2 = [p2(1), ymin, p2(2)];
        v3 = [p2(1), ymax, p2(2)];
        v4 = [p1(1), ymax, p1(2)];

        idx0 = size(V,1);
        V = [V; v1; v2; v3; v4];
        F = [F;
             idx0+1 idx0+2 idx0+3;
             idx0+1 idx0+3 idx0+4];
    end
end

patch('Vertices',V,'Faces',F, ...
    'FaceColor',faceColor, ...
    'FaceAlpha',faceAlpha, ...
    'EdgeColor','none');

for k = 1:numel(loops)
    loop = loops{k};
    if size(loop,1) < 3, continue; end
    if loop(1,1) ~= loop(end,1) || loop(1,2) ~= loop(end,2)
        loop = [loop; loop(1,:)];
    end
    xb = loop(:,1); zb = loop(:,2);
    plot3(xb, ymin*ones(size(xb)), zb, '-', 'Color', edgeColor, 'LineWidth', edgeLW);
    plot3(xb, ymax*ones(size(xb)), zb, '-', 'Color', edgeColor, 'LineWidth', edgeLW);
end
end
