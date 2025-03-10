clc;
clear;
close all;

image_path = 'floods.jpg'; 
img = imread(image_path);
img = imresize(img, [300, 300]); 

grayImage = rgb2gray(img);
grayImage = imadjust(grayImage); 
edges = edge(grayImage, 'Canny', [0.02, 0.2]);

binary_map = ~edges;

skeleton_map = bwmorph(binary_map, 'skel', Inf);

figure, imshow(img); title('Select FIRST START Point');
[start1_x, start1_y] = ginput(1);
start1 = round([start1_y, start1_x]);

figure, imshow(img); title('Select SECOND START Point');
[start2_x, start2_y] = ginput(1);
start2 = round([start2_y, start2_x]);

figure, imshow(img); title('Select GOAL Point');
[goal_x, goal_y] = ginput(1);
goal = round([goal_y, goal_x]);

start1 = find_nearest_valid_point(skeleton_map, start1);
start2 = find_nearest_valid_point(skeleton_map, start2);
goal = find_nearest_valid_point(skeleton_map, goal);

path1 = a_star(skeleton_map, start1, goal);
path2 = a_star(skeleton_map, start2, goal);

distance1_pixels = sum(sqrt(sum(diff(path1).^2, 2)));
distance2_pixels = sum(sqrt(sum(diff(path2).^2, 2)));

% Convert pixel distances to km (3km x 2km for 300x300 pixels)
pixel_to_km_x = 3 / 300;
pixel_to_km_y = 2 / 300;
distance1_km = distance1_pixels * sqrt(pixel_to_km_x^2 + pixel_to_km_y^2);
distance2_km = distance2_pixels * sqrt(pixel_to_km_x^2 + pixel_to_km_y^2);

if distance1_km < distance2_km
    best_path = path1;
    best_start = start1;
    best_distance = distance1_km;
    best_color = 'r';
else
    best_path = path2;
    best_start = start2;
    best_distance = distance2_km;
    best_color = 'b';
end

figure, imshow(img), hold on;
scatter(start1(2), start1(1), 50, 'g', 'filled'); % Start 1 (green)
scatter(start2(2), start2(1), 50, 'y', 'filled'); % Start 2 (yellow)
scatter(goal(2), goal(1), 50, 'b', 'filled'); % Goal (blue)
plot(path1(:,2), path1(:,1), 'r-', 'LineWidth', 2); % Path 1 (red)
plot(path2(:,2), path2(:,1), 'b-', 'LineWidth', 2); % Path 2 (blue)

% Display distances
mid1 = round(size(path1, 1) / 2);
mid2 = round(size(path2, 1) / 2);
text(path1(mid1,2), path1(mid1,1), sprintf('%.2f km', distance1_km), 'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold');
text(path2(mid2,2), path2(mid2,1), sprintf('%.2f km', distance2_km), 'Color', 'b', 'FontSize', 10, 'FontWeight', 'bold');
title(sprintf('Path Comparison: Best Path (%.2f km) Highlighted', best_distance));
hold off;

[X, Y] = meshgrid(1:size(img,2), 1:size(img,1)); 
Z = double(imresize(edges, size(X))) * 50; % Scale for visibility

figure;
surf(X, Y, Z, img, 'EdgeColor', 'none'); % Proper 3D Model
xlabel('X'); ylabel('Y'); zlabel('Depth');
title('3D Model with Path Projection');
view(3); camlight; lighting phong;
hold on;

scatter3(start1(2), start1(1), 50, 'g', 'filled'); % Start 1
scatter3(start2(2), start2(1), 50, 'y', 'filled'); % Start 2
scatter3(goal(2), goal(1), 50, 'b', 'filled'); % Goal

% Project paths onto the 3D model
path1_z = interp2(X, Y, Z, path1(:,2), path1(:,1)); % Get height values
path2_z = interp2(X, Y, Z, path2(:,2), path2(:,1));
plot3(path1(:,2), path1(:,1), path1_z, 'r-', 'LineWidth', 2);
plot3(path2(:,2), path2(:,1), path2_z, 'b-', 'LineWidth', 2);

hold off;

%% Helper Functions
function nearest = find_nearest_valid_point(skeleton_map, point)
    [rows, cols] = find(skeleton_map);
    distances = sqrt((rows - point(1)).^2 + (cols - point(2)).^2);
    [~, min_idx] = min(distances);
    nearest = [rows(min_idx), cols(min_idx)];
end

function path = a_star(map, start, goal)
    neighbors = [-1, 0; 1, 0; 0, -1; 0, 1; -1, -1; -1, 1; 1, -1; 1, 1];
    cost_move = [1; 1; 1; 1; sqrt(2); sqrt(2); sqrt(2); sqrt(2)];
    openList = containers.Map();
    openList(mat2str(start)) = struct('pos', start, 'g', 0, 'h', norm(start - goal), 'f', norm(start - goal), 'parent', []);
    closedList = false(size(map));
    while ~isempty(openList)
        keysList = keys(openList);
        f_values = cellfun(@(k) openList(k).f, keysList);
        [~, idx] = min(f_values);
        currentKey = keysList{idx};
        current = openList(currentKey);
        remove(openList, currentKey);
        closedList(current.pos(1), current.pos(2)) = true;
        if isequal(current.pos, goal)
            path = reconstruct_path(current);
            return;
        end
        for i = 1:size(neighbors, 1)
            neighbor_pos = current.pos + neighbors(i, :);
            if neighbor_pos(1) < 1 || neighbor_pos(2) < 1 || neighbor_pos(1) > size(map, 1) || neighbor_pos(2) > size(map, 2)
                continue;
            end
            if ~map(neighbor_pos(1), neighbor_pos(2)) || closedList(neighbor_pos(1), neighbor_pos(2))
                continue;
            end
            g_new = current.g + cost_move(i);
            h_new = norm(neighbor_pos - goal);
            f_new = g_new + h_new;
            neighborKey = mat2str(neighbor_pos);
            if ~isKey(openList, neighborKey) || g_new < openList(neighborKey).g
                openList(neighborKey) = struct('pos', neighbor_pos, 'g', g_new, 'h', h_new, 'f', f_new, 'parent', current);
            end
        end
    end
    path = [];
end

function path = reconstruct_path(node)
    path = [];
    while ~isempty(node)
        path = [node.pos; path];
        node = node.parent;
    end
end