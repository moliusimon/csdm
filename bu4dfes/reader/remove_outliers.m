function [ face ] = remove_ouliers( face, landmarks )
%EXTRACT_BCKG Summary of this function goes here
%   Detailed explanation goes here

%% Filter statistical outliers
miu = mean(face(:,3));
dev = std(face(:,3));
n = 2;

lower = face(:,3)<miu-abs(n*dev);
higher = face(:,3)>miu+abs(n*dev);

face(lower|higher, :) = [];

%% Filter by euclidean distance (sphere)
center = median(face(:,1:3));

dists = sqrt((repmat(center(:,1), length(face), 1) - face(:,1)).^2 + ...
             (repmat(center(:,2), length(face), 1) - face(:,2)).^2 + ...
             (repmat(center(:,3), length(face), 1) - face(:,3)).^2);
[~,idx] = sort(dists);

dists_landmarks = sqrt((repmat(center(:,1), length(landmarks), 1) - landmarks(:,1)).^2 + ...
             (repmat(center(:,2), length(landmarks), 1) - landmarks(:,2)).^2 + ...
             (repmat(center(:,3), length(landmarks), 1) - landmarks(:,3)).^2);
R = max(dists_landmarks);
    
% Remove all points further away than furthest landmark
face(dists>R,:) = [];

%% Filter by euclidean distance (rectangular)
min_mesh = min(landmarks);  
max_mesh = max(landmarks);
delta = max_mesh - min_mesh;
p = 0.1;
border = face(:,1) < (min_mesh(1) - p*delta(1)) | ...
         face(:,1) > (max_mesh(1) + p*delta(1));
face(border, :) = [];

end

