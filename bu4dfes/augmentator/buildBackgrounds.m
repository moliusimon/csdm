clear all;
addpath(genpath('../'));
load_path = '~/Research/data/bu4dfe/testSet_resize';
save_path = '~/Research/data/bu4dfe';

bReader = BU4DFEReader(load_path, save_path);
bckg_files = bReader.GetFiles(load_path, '*');
            
w = 200;
h = 200;

bckgs = uint8(zeros(750, w, h, 3));
idx = 0;

test_idx = [1,11,21,31,41,59,69,79,89,99];

for i = 1:length(bckg_files)
    idx = idx + 1;  
    i
    % Read image
    I = imread(strcat(load_path, filesep, bckg_files{i}));
    
    % If color
    if ndims(I)==2
        I = cat(3,I,I,I);  
    end
        
    % Resize to [w,h]
    sz_orig = size(I);
    I1 = imcrop(I, [0 0 200 200]);
    I2 = imcrop(I, [sz_orig(1)-w+1 sz_orig(2)-h+1 w h]);
     
    bckgs(2*idx-1,:,:,:) = I1;
    bckgs(2*idx,:,:,:) = fliplr(I2);
    
    % Save
    r = rem(i,375);  
    if r == 0
        q = fix(i/375);        
        fprintf('Processing image %d\n', i);
        idx = 0;
        
        if any(q==test_idx) 
            fprintf('Saving batch %d', q);
            save(strcat(save_path, filesep, 'bckgs', num2str(q),'.mat'), 'bckgs');
        end
    end
end

