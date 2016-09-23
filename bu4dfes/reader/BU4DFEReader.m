classdef BU4DFEReader < DataReader
    %BOSPHORUSREADER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods        
        %% Constructor
        function obj = BU4DFEReader(data_path, save_path)
            obj.DataPath = data_path;
            obj.SavePath = save_path;
        end
        
        %% Dataset iteration
        function Iterate(obj, Action)
            
            ParentSubdirs = GetSubdirs(obj, obj.DataPath);
            
            for subject = 1:size(ParentSubdirs,2)
                Path = strcat(obj.DataPath, filesep,...
                                 ParentSubdirs{subject});
                             
                rgb_files = GetFiles(obj, Path, '*.png');  
                depth_files = GetFiles(obj, Path, '*.bnt');
                rgb_lm_files = GetFiles(obj, Path, '*.lm2');
                depth_lm_files = GetFiles(obj, Path, '*.lm3');               
                     
                % For all samples
                for file = 1:numel(rgb_files)         
                    % Read data
                    rgb{file} = rgb2gray(imread(strcat(Path, filesep, rgb_files{file})));
                    rgb_lm{file} = read_lm2file(strcat(Path, filesep, rgb_lm_files{file}));
                    depth{file} = read_bntfile(strcat(Path, filesep, depth_files{file}));
                    depth_lm{file} = read_lm3file(strcat(Path, filesep, depth_lm_files{file}));
                end

                % Create context
                %context = struct('Subject', subject, 'Sequence', seq);

                % Apply action on data in context                          
                display_rgb(rgb, rgb_lm);
            end
        end  
        
        function [] = Read(obj)         
            % Get all subdirectories in path; each one corresponds to a
            % different person
            PersonSubdirs = GetSubdirs(obj, obj.DataPath);
                       
            for person = 1:length(PersonSubdirs) 
                samples = {};
                fprintf(strcat('Reading person:\t', PersonSubdirs{person}, '\n'));                              
                Path = strcat(obj.DataPath, filesep,PersonSubdirs{person});
                             
                % Get all subdirectories in path; each one corresponds to a
                % different emotion
                EmoSubdirs = GetSubdirs(obj, Path);
                
                if isempty(EmoSubdirs)
                    fprintf('\tEmpty folder\n');
                end
                
                index = 1;
                for emo = 1:length(EmoSubdirs)                   
                    fprintf(strcat('\tReading emo:\t ', EmoSubdirs{emo}, '\n')); 

                    Path = strcat(obj.DataPath, filesep,PersonSubdirs{person}, filesep, EmoSubdirs{emo});
             
                    % Get all files by type from each subdir
                    rgb_files = GetFiles(obj, Path, '*.jpg');  
                    depth_files = GetFiles(obj, Path, '*.wrl');
                    lm_files = GetFiles(obj, Path, '*.bnd');             

                    M = numel(rgb_files);
                    if M>1
                        N = 5; % number of sampled frames                      
                        % Save each file in corresponding sample
                        step = floor(M/(N+1));
                        file_idxs = step:step:M-step;
                        for i = 1:length(file_idxs)
                            file_idx = file_idxs(i);

                            % Read data
                            sample.rgb = rgb2gray(imread(strcat(Path, filesep, rgb_files{file_idx})));                                    
                            bnd = read_bnd(strcat(Path, filesep, lm_files{file_idx}));
                            depth = read_vrml(strcat(Path, filesep, depth_files{file_idx}));
                           
                            % Remove out of border and outliers  
                            lim_depth = remove_outliers(depth, depth(bnd(:,1)', 1:3));

                            % Prepare landmarks                           
                            [h,w,~] = size(sample.rgb);
                            lm_x = int16(w*depth(bnd(:,1)', 4));
                            lm_y = h - int16(h*depth(bnd(:,1)', 5));
                            lm_z = depth(bnd(:,1)', 3);                            
 
                            sample.lm = [lm_x lm_y lm_z];
                            sample.bnd = bnd;
                            sample.depth = lim_depth;
                            sample.context.person = PersonSubdirs{person};
                            sample.context.emo = EmoSubdirs{person};
                            sample.context.pose.yaw = 0;
                            sample.context.pose.pitch = 0;
                            sample.context.pose.roll = 0;
                            
                            % Resize and center
                            %[sample] = obj.Resize(rgb, rgb_lm, depth, depth_lm, [200, 200]);   
                            samples{index} = sample;
                            index = index + 1;
                        end                   
                    else
                        fprintf('\t\t No files found\n');
                    end
                end
                save(strcat(obj.SavePath, filesep, 'person', num2str(person), '.mat'), 'samples');
            end
        end
        
        function [out] = Resize(obj, rgb, rgb_lm, depth, depth_lm, sz)
             
            % Scale to sz by keeping aspect ratio
            [val,idx] = max(size(rgb));
            scale = sz(idx)/val;
            depth = extract_bckg(depth);

            if idx == 1               
                rgb = imresize(rgb, [sz(1) NaN]);
                offset_left = round((sz(1) - size(rgb,2))/2);
                offset_right = sz(1) - size(rgb,2) - offset_left;
                bckgl = repmat(zeros(1, offset_left), sz(2), 1);
                bckgr = repmat(zeros(1, offset_right), sz(2), 1);
                
                aratio = size(rgb,2)/sz(1);
                
                out.rgb = [bckgl rgb bckgr];
                out.rgb_lm = scale*rgb_lm + offset_left*[ones(1, length(rgb_lm)); zeros(1, length(rgb_lm))];
                
                %{
                out.depth(:,1:3) = depth(:,1:3);
                out.depth(:,4) = aratio*depth(:,4)+(1-aratio)/2;
                out.depth(:,5) = depth(:,5);
                out.depth_lm = depth_lm;
                %}
            else
                rgb = imresize(rgb, [NaN sz(2)]);
                offset_up = round((sz(2) - size(rgb,1))/2);
                offset_down = sz(2) - size(rgb,1) - offset_up;
                bckgu = repmat(zeros(1,sz(1)), offset_up, 1);
                bckgd = repmat(zeros(1,sz(1)), offset_down, 1);
                
                aratio = size(rgb,1)/sz(2);
                
                out.rgb = [bckgu; rgb; bckgd];
                out.rgb_lm = scale*rgb_lm + offset_up*[zeros(1, length(rgb_lm)); ones(1, length(rgb_lm))];
                
                %{
                out.depth(:,1:3) = depth(:,1:3);
                out.depth(:,4) = depth(:,4);
                out.depth(:,5) = aratio*depth(:,5)+(1-aratio)/2;
                out.depth_lm = depth_lm;
                %}
            end           
        end
    end  
end



