classdef DataReader
    % DATASETREADER Read dataset.
    % Refer to TOADD    
    %
    % DatasetReader Properties:
    %    DataPath - 
    %    
    % CKReader Methods:
    %    GetFiles - 
    %    getSubdirs  - 
 
    properties
        DataPath;
        SavePath;
    end
    
    methods     
        %{
        %% Default constructor
        function obj = DataReader(data_path)
            obj.DataPath = data_path;
        end
        %}
        
        %% Get files of specified path
        function [ files ] = GetFiles( obj, path, varargin )
            
            nVarargs = length(varargin);
 
            % Get the data for the subject directory      
            if nVarargs>0
                data = dir(fullfile(path,varargin{1}));
            else
                data = dir(path);
            end    
            
            % Find the index for directories      
            index = ~[data.isdir]; 

            % Get a list of the subject subdirectories
            files = {data(index).name};  

            % Remove current and parent directory from the list
            invalid_index = find(ismember({data(index).name},{'.','..', '.DS_Store'}));

            % Remove current and parent dir from subdirs
            files(invalid_index) = [];
        end
         
        %% Get subdirectories of specified path
        function [ subdirs ] = GetSubdirs( obj, path )
            % Get the data for the subject directory                
            data = dir(path); 

            % Find the index for directories      
            index = [data.isdir]; 

            % Get a list of the subject subdirectories
            subdirs = {data(index).name};  

            % Remove current and parent directory from the list
            invalid_index = find(ismember({data(index).name},{'.','..'}));

            % Remove current and parent dir from subdirs
            subdirs(invalid_index) = [];
        end
        
        %% Formatted read line by line of text file
        function [ datas ] = ReadLines( obj, file, FormattedRead )
        %READLINES
        %   TO ADD ARGS DESCR
            datas = [];

            % Open file
            fid = fopen(file);

            % Read file line by line
            tline = fgetl(fid);
            while ischar(tline)
                data = strsplit(tline, ' ');
                
                % Get formatted data
                data = FormattedRead(obj, data);

                % Save in array
                datas = [datas, data];

                % Get next line
                tline = fgetl(fid);
            end
            fclose(fid);
        end               
    end    
end

