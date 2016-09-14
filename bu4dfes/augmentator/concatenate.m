%% Mean mesh sampling method. 
%addpath(genpath('../'));

%clear all;

% Data path
local_input_path = '/input';
local_output_path = '/output';
cluster_input_path = '../../../data/bu4dfe/output';
cluster_output_path = '../../../data/bu4dfe/output/conc';

input_path = cluster_input_path;
output_path = cluster_output_path;

start_pers = 23;
stop_pers = 51;

for p = start_pers:stop_pers
    pers = [];
    fprintf('Reading person %d\n', p);
    for emo = 1:36
        fprintf('\tReading emo % d\n', emo);
        file_name = strcat(input_path, filesep, 'augperson', num2str(p), '_', num2str(emo), '.mat');
        if exist(file_name, 'file') == 2
            sample = importdata(file_name);
            pers = [pers; sample(emo,:)];
        end
    end
    
    save(strcat(output_path, filesep, 'augperson', num2str(p), '.mat'), 'augperson');
end


