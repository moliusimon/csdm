function [rgb, landmarks] = createSample(rotated)
    % Crop image
    mn = min(rotated.landmarks(:,1:2));
    mx = max(rotated.landmarks(:,1:2));
    margin = 0.2;
    [h, w] = size(rotated.rgb);
    bbox = [max(0,mn(1) - margin*(mx(1)-mn(1))), max(0,mn(2) - margin*(mx(1)-mn(1))), ...
            min(h,mx(1) + margin*(mx(1)-mn(1))), min(w,mx(2) + margin*(mx(1)-mn(1)))];        
    rotated.rgb = imcrop(rotated.rgb, bbox);
    rotated.landmarks(:,1:2) = bsxfun(@minus, rotated.landmarks(:,1:2), bbox(1:2));
    
    % Resize
    SIZE = [200,200];
    rsz_rotated = resize(rotated.rgb, rotated.landmarks, SIZE);
    
    % Landmarks
    miu_depth = mean(rsz_rotated.landmarks(:,3));
    
    rgb = rsz_rotated.rgb;
    landmarks = [rsz_rotated.landmarks(:,1:2) rsz_rotated.landmarks(:,3) - miu_depth];
end
