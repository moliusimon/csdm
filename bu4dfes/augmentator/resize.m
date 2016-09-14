function [out] = resize(rgb, rgb_lm, sz)

    % Scale to sz by keeping aspect ratio

    [val,idx] = max(size(rgb));
    scale = sz(idx)/val;

    if idx == 1               
        rgb = imresize(rgb, [sz(1) NaN]);
        offset_left = round((sz(1) - size(rgb,2))/2);
        offset_right = sz(1) - size(rgb,2) - offset_left;
        bckgl = zeros(sz(2), offset_left, size(rgb,3));
        bckgr = zeros(sz(2), offset_right, size(rgb,3));

        aratio = size(rgb,2)/sz(1);

        out.rgb = [bckgl rgb bckgr];
        out.landmarks = scale*rgb_lm + offset_left*[ones(1, length(rgb_lm)); zeros(2, length(rgb_lm))]';
    else
        rgb = imresize(rgb, [NaN sz(2)]);
        offset_up = round((sz(2) - size(rgb,1))/2);
        offset_down = sz(2) - size(rgb,1) - offset_up;
        bckgu = zeros(offset_up, sz(1), size(rgb,3));
        bckgd = zeros(offset_down, sz(1), size(rgb,3));
        
        aratio = size(rgb,1)/sz(2);

        out.rgb = [bckgu; rgb; bckgd];
        out.landmarks = scale*rgb_lm + offset_up*[zeros(1, length(rgb_lm)); ...
                    ones(1, length(rgb_lm)); zeros(1, length(rgb_lm))]';
    end           
end

