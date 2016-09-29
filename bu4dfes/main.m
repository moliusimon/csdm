function [ ] = main( path2raw_data, path2proc_data, path2synt_data )
% Script for generatin synthesized data from BU4DFE
% Refer to "Continuous Supervised Descent Method forFacial Landmark
% Localisation, M. Oliu, C. Corneanu, L. Jeni, J. Cohn, T. Kanade,
% S. Escalera. ACCV 2016" for more details 

addpath(genpath('./'))

%% Read BU4DFE and save to file. 
% The contents of each persons' directory in the BU4DFE structure is
% loaded, processed and saved in path2proc_data/persx.mat
 read(path2raw_data, path2proc_data);

%% Generate synthesized data and save to file.
% From each existing preprocessed file in path2proc_data generate the
% synthesized images
n_pers = 99;
for pers = 1:n_pers
    fname = strcat(path2proc_data, filesep, 'person', int2str(pers), '.mat');
    if exist(fname, 'file')
        fprintf('-> Generating data for person %d, %d remaining\n', pers, n_pers-pers); 
        augment_pers(pers, path2proc_data, path2synt_data);
    end
end


end

