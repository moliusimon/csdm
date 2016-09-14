function [] = augment(idx,emo)

    %% Mean mesh sampling method. 
    %addpath(genpath('../'));

    %clear all;

    % Data path
    local_input_path = '/Users/cipriancorneanu/Research/DATASETS/BU4DFEproc';
    local_output_path = '/Users/cipriancorneanu/Research/DATASETS/BU4DFEproc';
    cluster_input_path = '../../../data/bu4dfe/input';
    cluster_output_path = '../../../data/bu4dfe/output';

    input_path = cluster_input_path;
    output_path = cluster_output_path;

    start_batch = idx;
    stop_batch = idx;
    
    pers_per_batch = 1;
    
    N = 25;
    pitch = (pi*rand(1,N) - pi/2)';
    yaw = (pi*rand(1,N) - pi/2)';
    rots = [pitch yaw zeros(N,1)];
    bckgs = importdata(strcat(input_path, filesep, 'bckgs.mat'));

    for j = start_batch:stop_batch    
        batch_name = strcat(num2str((j-1)*pers_per_batch + 1));
        procperson_filename = strcat('person', batch_name, '.mat');

        % Load frontal
        procperson = importdata(strcat(input_path,filesep,procperson_filename));

        for i = emo 
            % For each rotation
            for r = 1:size(rots,1)                
                % Load background
                bckgs_idx = mod(r+(i-1)*size(rots,1), length(bckgs));
                if bckgs_idx == 0
                    bckgs_idx = bckgs_idx + 1;
                end
                
                fprintf('Building rotated batch for Sample %d, %d Yaw and %d Pitch\n', i, rots(r,1), rots(r,2));

                source_rots = repmat([procperson{i}.context.pose.yaw ...
                              procperson{i}.context.pose.pitch...
                              procperson{i}.context.pose.roll], size(rots,1), 1);
                d_rots = rots - source_rots;
                
                rotated = createRotatedPerson(procperson{i}, d_rots(r,1), d_rots(r,2), d_rots(r,3), bckgs{bckgs_idx}, 0);

                % Update rotation 
                rotated.context.pose.roll = rots(r,3)*180/pi;        
                rotated.context.pose.pitch = rots(r,1)*180/pi;
                rotated.context.pose.yaw = rots(r,2)*180/pi;
    
                % Visualize                  
                %{
		figure();
                imshow(rotated.rgb);                
                hold on;
                scatter3(rotated.landmarks(:,1),...
                        rotated.landmarks(:,2), 500+rotated.landmarks(:,3), 'filled');
                %}
                    
                augperson(i,r) = rotated;                               
            end
        end
        % Save data
        batch_fn = strcat(output_path,filesep,'augperson', batch_name,'_', num2str(emo),'.mat');
        save(batch_fn, 'augperson');
    end
end
