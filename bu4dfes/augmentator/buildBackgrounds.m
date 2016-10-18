addpath(genpath('../'));

path2data = '~/Research/data/bu4dfe/places_dataset'; % add your path here
path2proc_data = '~/Research/data/bu4dfe/backgrounds'; % add your path here

% Get files from path
bReader = BU4DFEReader(path2data, path2proc_data);
bckg_files = bReader.GetFiles(load_path, '*');
         
% Init
w = 200; h = 200;
bckgs = uint8(zeros(750, w, h, 3));
idx = 0;
test_idx = [1,11,21,31,41,59,69,79,89,99];

%% 
for i = 1:length(bckg_files)
    idx = idx + 1;  

    % Read image
    I = imread(strcat(path2data, filesep, bckg_files{i}));
    
    % If color
    if ndims(I)==2
        I = cat(3,I,I,I);  
    end
        
    % Resize to [w,h]
    sz_orig = size(I);
    I1 = imcrop(I, [0 0 200 200]);
    I2 = imcrop(I, [sz_orig(1)-w+1 sz_orig(2)-h+1 w h]);
     
    % Save crop and specular crop
    bckgs(2*idx-1,:,:,:) = I1;
    bckgs(2*idx,:,:,:) = fliplr(I2);
    
    % Write to file
    r = rem(i,375);  
    if r == 0
        q = fix(i/375);        
        fprintf('Processing image %d\n', i);
        idx = 0;
        
        if any(q==test_idx) 
            fprintf('Saving batch %d', q);
            save(strcat(path2proc_data, filesep, 'bckgs', num2str(q),'.mat'), 'bckgs');
        end
    end
end

