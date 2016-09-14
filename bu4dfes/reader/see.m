function [ ] = see(sample)
%SEE Visualize data sample
    figure()
    imshow(sample.rgb); hold on;
    
    [h,w] = size(sample.rgb);
    
    lm_x = int16(w*sample.depth(sample.lm(:,1)', 4));
    lm_y = h - int16(h*sample.depth(sample.lm(:,1)', 5));
    scatter(lm_x,lm_y);
end

