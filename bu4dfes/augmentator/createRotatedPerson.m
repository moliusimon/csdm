function [ rotatedperson ] = createRotatedPerson( persondata, pitch, yaw, roll, background, showimg )
%CREATEROTATEDPERSON Summary of this function goes here
%   Detailed explanation goes here
%  e.g. dataperson = frontal(1); A struct with raw_depth, depth, rgb,
%  context
 
    %% Define input
    S = 5; % Subsample each S depth coordinates
    rgb = persondata.rgb;
    [h,w,~] = size(rgb);

    raw_depth = persondata.depth;

    %% Create rotation matrix
    anglex =  pitch; angley = yaw; anglez= roll; % pitch, yaw, roll
    Rx = [1 0 0; 0 cos(anglex) -sin(anglex); 0 sin(anglex) cos(anglex)]; % Rotation matrix x
    Ry = [cos(angley) 0 sin(angley); 0 1 0; -sin(angley) 0 cos(angley)]; % Rotation matrix y 
    Rz = [cos(anglez) -sin(anglez) 0; sin(anglez) cos(anglez) 0; 0 0 1]; % Rotation matrix z
    R = Rz*Ry*Rx;
       
    %% Triangularization
    % Based on http://es.mathworks.com/matlabcentral/fileexchange/44651-active-appearance-models--aams-/content/tzimiro_ICCV2013_code/functions/warp_image.m
    [h, w, ~] = size(rgb);
      
    %% Sample raw depth
    raw_depth_ssmpl = raw_depth(1:S:end,:);
    raw_depth_ssmpl(:,4) = int16(raw_depth_ssmpl(:,4)*w);
    raw_depth_ssmpl(:,5) = int16(h-raw_depth_ssmpl(:,5)*h);
    delta = max(raw_depth_ssmpl) - min(raw_depth_ssmpl);
    scale_z = mean([delta(4)/delta(1), delta(5)/delta(2)]);% compensates passing from coord to tex coord in z
    
    %% Rotate depth 
    depth = [raw_depth_ssmpl(:,4:5), scale_z*raw_depth_ssmpl(:,3)];
    
    landmarks = double([persondata.lm(:,1) persondata.lm(:,2) scale_z*persondata.lm(:,3)]);
        
    % Remove duplicates
    [~, U, ~] = unique(depth(:,1:2),'first','rows');
    U = sort(U);
    depth = depth(U,:);
    rot_depth = depth;

    %% Rotate landmarks 
    rot_landmarks = landmarks;   
    miu = mean(rot_landmarks);
    rot_landmarks = rot_landmarks - repmat(miu, length(rot_landmarks), 1);
    rot_landmarks = (R*rot_landmarks')';
    rot_landmarks = rot_landmarks + repmat(miu, length(rot_landmarks), 1);  
    
    %% Rotate mesh (substract same mean as for landmarks)
    rot_depth = rot_depth - repmat(miu, length(rot_depth), 1);
    rot_depth = (R*rot_depth')';
    rot_depth = rot_depth + repmat(miu, length(rot_depth), 1);
    
    %% Triangulate
    tri = delaunay(depth(:,1:2));
    
    if showimg
        figure(1);
        %trisurf(tri, finite_depth_ssmpl(:,1), finite_depth_ssmpl(:,2), finite_depth_ssmpl(:,3));
        trisurf(tri, depth(:,1), depth(:,2), depth(:,3)*2);
        shading interp;
        faceted;
        figure(2);
        %trisurf(tri, rot_fdepth(:,1), rot_fdepth(:,2), rot_fdepth(:,3));
        trisurf(tri, rot_depth(:,1), rot_depth(:,2), rot_depth(:,3));
    end
    
    warped = zeros(h, w, 3, 'uint8');
    tic
    
    % Sort idx_list triangle indices depending on rotated triangle z index
    tri_z = zeros(length(tri), 3);
    for k = 1:length(tri) % Get the z variable of each (rotated) triangle vertice
       tri_z(k,  :) =  rot_depth(tri(k,:), 3);
    end
    min_z = min(tri_z, [], 2); %% Get the idx of the triangles with minimm depth (aka closer to the camera)
    [~, sorted_idx] = sort(min_z, 'descend'); % Sort the triangle indexes from further to closer
    idx_list = sorted_idx;
    clear min_z tri_z sorted_idx

    % General target mask
    M = false(h,w);
    
    % General grid
    grid = zeros(w,h,4);
    
    %% Rendition
    for j = 1:length(idx_list)   % For each triangle...
 
        t = idx_list(j); % t will start with the triangle which are closer 
        
        % Compute afine transformation in pixel space btw source and target
        source = depth(tri(t,:),1:2);
        target = rot_depth(tri(t,:), 1:2);
               
        % If triangle changes
        %if (~isequal(source, target))
            % Include border
            source = includeBorders(source);
            target = includeBorders(target);
                
            % Check if colinear points in  in A or B
            if (~isColinear(source) && ~isColinear(target))
                                                          
                mask_source = poly2mask(source(:,1), source(:,2),h, w);               
                mask_target = poly2mask(target(:,1), target(:,2),h, w);

                % Compute target pixels
                [v,u] = find(mask_source);                                
                tform = maketform('affine',source,target); %% Matlab < 2013b
                [x, y] = tformfwd(tform, u, v); %% Matlab < 2013b
                
                y = round(y); x = round(x);  
                
                ind_target = y + (x-1)*h;
                ind_source = v + (u-1)*h;
                
                % If between borders and not occluded
                list = ind_target > 0 & ind_target <= w*h;
                ind_target = ind_target(list);
                ind_source = ind_source(list);
                list = M(ind_target) == 0;
                
                if (any(list))         
                    warped(ind_target(list)) = rgb(ind_source(list));
                    warped(ind_target(list) + h*w) = rgb(ind_source(list) + h*w);
                    warped(ind_target(list) + 2*h*w) = rgb(ind_source(list) + 2*h*w);
                end    
                
                % Mark rendered
                M = M | mask_target;
            end
        %end        
    end  
    
    %% Fill holes in target img
    % Get rotated face mask and set it to missing values
    BWw = warped > 0; % Binarize image to get a mask
    mask = imclose(BWw, strel('disk',10)); % Fill the holes in the mask to get the full face
    mask = (mask & ~BWw)*Inf; % Extract the holes and multiply it by Inf
    mask(isnan(mask)) = 0; % 0*Inf = NaN so put back the 0s
    filled = inpaintn(double(warped)+mask, 100); % Inpaint the inf
    
    if showimg
        toc
        figure(3);
        subplot(1,2,1);
        imshow(rgb);
        subplot(1,2,2);
        imshow(warped);
        figure;imshow(uint8(filled));
    end
    
    %% Add background to texture
    rotatedperson.rgb = uint8(filled);
    rotatedperson.landmarks = rot_landmarks;
    [filled, rot_landmarks] = createSample(rotatedperson);
    mask = find(filled==0);
    filled(mask) = background(mask);
    % Use a median filter for each channel
    filled(:,:,1) = medfilt2(filled(:,:,1),[2 2]);
    filled(:,:,2) = medfilt2(filled(:,:,2),[2 2]);
    filled(:,:,3) = medfilt2(filled(:,:,3),[2 2]);
    
    figure(); imshow(uint8(filled));
    rotatedperson.rgb = uint8(filled);
    rotatedperson.landmarks = rot_landmarks;
    rotatedperson.context = persondata.context;
    
    %% HELPER FUNCTIONS
    function [out] = isColinear(A)       
        out = A(1,1) * (A(2,2) - A(3,2)) + A(2,1) * (A(3,2) - A(1,2)) + A(3,1) * (A(1,2) - A(2,2)) == 0;
    end

    function [Tout] = includeBorders(T)
        
        % Order vertices on horizontal
        Tx = [T(1,1) T(2,1) T(3,1)]';
        Ty = [T(1,2) T(2,2) T(3,2)]';
        
        % Get extremes
        idxMinTx = find(Tx == min(Tx(:))); idxMaxTx = find(Tx == max(Tx(:)));
        idxMinTy = find(Ty == min(Ty(:))); idxMaxTy = find(Ty == max(Ty(:)));
        
        % Split extremes
        Tx(idxMinTx) = Tx(idxMinTx) - 1;
        Tx(idxMaxTx) = Tx(idxMaxTx) + 1;
        
        Ty(idxMinTy) = Ty(idxMinTy) - 1;
        Ty(idxMaxTy) = Ty(idxMaxTy) + 1;  
        
        % Out
        Tout = [Tx Ty];
    end
end

