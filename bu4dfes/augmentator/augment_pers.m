function [] = augment_pers(idx)

    %% Mean mesh sampling method. 
    warning off;

    % Data path
    local_input_path = '/Users/cipriancorneanu/Research/data/bu4dfe/pers/color';
    local_output_path = '/Users/cipriancorneanu/Research/data/bu4dfe/pers/color';
    cluster_input_path = '/home/cvc/corneanu/data/bu4dfe/pers/color';
    cluster_output_path = '/home/cvc/corneanu/bmvc/bu4dfe/test/color';

    input_path = local_input_path;
    output_path = local_output_path;

    start_batch = idx;
    stop_batch = idx;
    
    pers_per_batch = 1;
    
    N_ROT = 25;
    
    pdistro = importdata(strcat(input_path, filesep, 'pitch_distro.mat'));
    ydistro = importdata(strcat(input_path, filesep, 'yaw_distro.mat'));
       
    for j = start_batch:stop_batch    
        bckgs = importdata(strcat(input_path, filesep, 'bckgs', num2str(j),'.mat'));
        batch_name = strcat(num2str((j-1)*pers_per_batch + 1));
        procperson_filename = strcat('person', batch_name, '.mat');

        % Load frontal
        procperson = importdata(strcat(input_path,filesep,procperson_filename));

        for i = 1:1%numel(procperson) 
            % Define rotations
            start_slice = (j-1)*N_ROT*numel(procperson) + (i-1)*N_ROT+1;
            stop_slice = (j-1)*N_ROT*numel(procperson) + i*N_ROT;
            
            yaw = (pi/2)*ydistro(start_slice:stop_slice); 
            pitch = (pi/4)*pdistro(start_slice:stop_slice); 
            rots = [pitch yaw zeros(N_ROT,1)];
 
            % For each rotation
            for r = 1:size(rots,1)
                tic;
                
                bckg = squeeze(bckgs((i-1)*N_ROT+(r-1)+1,:,:,:));
                
                fprintf('Building rotated batch for Sample %d: %.2f Pitch and  %.2f Yaw', i, rots(r,1)*180/pi, rots(r,2)*180/pi);

                source_rots = repmat([procperson{i}.context.pose.yaw ...
                              procperson{i}.context.pose.pitch...
                              procperson{i}.context.pose.roll], size(rots,1), 1);
                d_rots = rots - source_rots;
                
                rotated = createRotatedPerson(procperson{i}, d_rots(r,1), d_rots(r,2), d_rots(r,3), bckg, 1);

                % Update rotation 
                rotated.context.pose.roll = rots(r,3)*180/pi;        
                rotated.context.pose.pitch = rots(r,1)*180/pi;
                rotated.context.pose.yaw = rots(r,2)*180/pi;
    
                % Visualize                   
                figure();
                imshow(rotated.rgb);                
                hold on;
                scatter3(rotated.landmarks(:,1),...
                        rotated.landmarks(:,2), 500+rotated.landmarks(:,3), 'filled');
                                    
                augperson(i,r) = rotated;
                fprintf('->\t %.1f secs\n', toc);               
            end
        end
        % Save data
        batch_fn = strcat(output_path,filesep,'augperson', batch_name,'.mat');
        save(batch_fn, 'augperson');
    end
end
