% Script for generatin synthesized data from BU4DFE
% Refer to "Continuous Supervised Descent Method forFacial Landmark
% Localisation, M. Oliu, C. Corneanu, L. Jeni, J. Cohn, T. Kanade,
% S. Escalera. ACCV 2016" for more details 

addpath(genpath('./'))

path2raw_data = '/Users/cipriancorneanu/Research/data/bu4dfe/data'; % add your path here
path2proc_data = '/Users/cipriancorneanu/Research/data/bu4dfe/pers'; % add your path here
path2synt_data = '/Users/cipriancorneanu/Research/data/bu4dfe/pers'; % add your path here


%% Read BU4DFE and save to file. 
% The contents of each persons' directory in the BU4DFE structure is
% loaded, processed and saved in path2proc_data/persx.mat
% TODO: If file already existing don't read. 
read(path2raw_data, path2proc_data);

%% Generate synthesized data and save to file.
% From each existing preprocessed file in path2proc_data generate the
% synthesized images
% TODO: Look for existing preprocessed files 
n_pers = 99;
for pers = 1:n_pers
    fprintf('-> Generating data for person %d, %d remaining\n', pers, n_pers-pers); 
    augment_pers(pers, path2proc_data, path2synt_data);
end

