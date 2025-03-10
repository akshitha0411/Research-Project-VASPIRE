clc;
clear;
close all;

image_path = 'floods.jpg'; 
img = imread(image_path);
img = imresize(img, [500, 500]); 

grayImage = rgb2gray(img);
grayImage = imadjust(grayImage); 

edgesCanny = edge(grayImage, 'Canny', [0.02, 0.2]);
edgesSobel = edge(grayImage, 'Sobel');
edgesPrewitt = edge(grayImage, 'Prewitt');
edgesLoG = edge(grayImage, 'log'); 

combinedEdges = edgesCanny | edgesSobel | edgesPrewitt | edgesLoG;

h = fspecial('laplacian', 0.2);
laplacianEdges = imfilter(grayImage, h);
laplacianEdges = edge(laplacianEdges, 'zerocross');
finalEdges = combinedEdges | laplacianEdges;

binary_map = ~finalEdges;

skeleton_map = bwmorph(binary_map, 'skel', Inf);

figure, imshow(img), title('Original Image');
figure, imshow(finalEdges), title('Detected Edges');
figure, imshow(skeleton_map), title('Skeletonized Path');

figure, imshow(img); title('Select START Point');
[start_x, start_y] = ginput(1);
start = round([start_y, start_x]);

figure, imshow(img); title('Select GOAL Point');
[goal_x, goal_y] = ginput(1);
goal = round([goal_y, goal_x]);

% Adjust start and goal points if not on a valid path
if ~skeleton_map(start(1), start(2))
    start = find_nearest_valid_point(skeleton_map, start);
end
if ~skeleton_map(goal(1), goal(2))
    goal = find_nearest_valid_point(skeleton_map, goal);
end


path = a_star(skeleton_map, start, goal);

euclidian_distance = sum(sqrt(sum(diff(path).^2, 2)));
disp(['Euclidean Distance Along Path: ', num2str(euclidian_distance)]);


[X, Y] = meshgrid(1:size(finalEdges, 2), 1:size(finalEdges, 1));
Z = double(finalEdges) * 50; 

figure;
surf(X, Y, Z, img, 'EdgeColor', 'none'); 
colormap jet;
xlabel('X'); ylabel('Y'); zlabel('Depth');
title('Generated 3D Model from Image');
view(3); camlight; lighting phong;
hold on;

scatter3(start(2), start(1), 50, 'g', 'filled'); 
scatter3(goal(2), goal(1), 50, 'b', 'filled');   

if ~isempty(path)
    path_z = interp2(X, Y, Z, path(:,2), path(:,1)); 
    plot3(path(:,2), path(:,1), path_z, 'r-', 'LineWidth', 2);


    for i = 1:length(path)
        scatter3(path(i,2), path(i,1), path_z(i), 100, 'm', 'filled');
        pause(0.1); 
    end
end
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
    openList(1).pos = start;
    openList(1).g = 0;
    openList(1).h = norm(start - goal);
    openList(1).f = openList(1).g + openList(1).h;
    openList(1).parent = [];
    closedList = false(size(map));
    while ~isempty(openList)
        f_values = [openList.f];
        [~, idx] = min(f_values);
        current = openList(idx);
        openList(idx) = [];
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
            existingNodeIndex = find(arrayfun(@(node) isequal(node.pos, neighbor_pos), openList), 1);
            if isempty(existingNodeIndex)
                newNode.pos = neighbor_pos;
                newNode.g = g_new;
                newNode.h = h_new;
                newNode.f = f_new;
                newNode.parent = current;
                openList = [openList, newNode];
            elseif g_new < openList(existingNodeIndex).g
                openList(existingNodeIndex).g = g_new;
                openList(existingNodeIndex).f = f_new;
                openList(existingNodeIndex).parent = current;
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