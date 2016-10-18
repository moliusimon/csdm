% Script for generatin synthesized data from BU4DFE
% Refer to "Continuous Supervised Descent Method forFacial Landmark
% Localisation, M. Oliu, C. Corneanu, L. Jeni, J. Cohn, T. Kanade,
% S. Escalera. ACCV 2016" for more details 

addpath(genpath('./'))

path2raw_data = '/Users/cipriancorneanu/Research/data/bu4dfe/data'; 
path2proc_data = '/Users/cipriancorneanu/Research/data/bu4dfe/pers'; % add your path here
path2synt_data = '/Users/cipriancorneanu/Research/data/bu4dfe/pers'; % add your path here

main(path2raw_data, path2proc_data, path2synt_data);

