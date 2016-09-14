function [w3d] = read_vrml(filename)

    keynames=char(' Coordinate { point', ' TextureCoordinate { point', 'texCoordIndex', 'coordIndex');

    fp = fopen(filename,'r');
    if fp == -1
        fclose all;
        str = sprintf('Cannot open file %s \n',filename);
        errordlg(str);
        error(str);
    end

    %* initialise arrays & counters */
    N = size(keynames,1);
    fv = zeros(1,N);
    foundkey=zeros(1,N); %* flags to determine if keywords found */
    tempstr = ' ';

    function [out ,f] = read_coordinates(f)
        out = [];
        tempstr = ' ';
        npt=0;
        coord_end = [];
        
        while tempstr~=-1
            tempstr = fgets(f);
            
            % Break when reaching end of element
            coord_end = strfind(tempstr, ']');            
            if ~isempty(coord_end)
                break;
            end
            
            sp = strfind(tempstr,'[');            
            if isempty(sp)
               %/* points data in x y z columns */
               [fv,nv]=sscanf(tempstr,'%f %f %f,');
            else
               %/* if block start marker [ in line - need to skip over it to data 
               %   hence pointer to marker incremented */
               sp = sp +1;
               [fv,nv]=sscanf(tempstr(sp:length(tempstr)),'%f %f %f,');
            end

            if(nv>0)
               if mod(nv,3) ~= 0
                  fclose(f);
                  error('Error reading 3d wire co-ordinates: should be x y z, on each line');
               end 
               nov = fix(nv/3);
               for p = 1:nov
                  npt = npt+1;
                  out(npt,1:3)=fv(3*p-2:3*p); 
               end
            end                  
        end
    end
 
    function [out, f] = read_texture_coordinates(f)
        out = [];
        tempstr = ' ';
        coord_end = [];
        npt=0;
        
        while tempstr~=-1
            tempstr = fgets(f);
            
            % Break when reaching end of element
            coord_end = strfind(tempstr, ']');            
            if ~isempty(coord_end)
                break;
            end
            
            sp = strfind(tempstr,'[');            
            if isempty(sp)
               %/* points data in x y z columns */
               [fv,nv]=sscanf(tempstr,'%f %f,');
            else
               %/* if block start marker [ in line - need to skip over it to data 
               %   hence pointer to marker incremented */
               sp = sp +1;
               [fv,nv]=sscanf(tempstr(sp:length(tempstr)),'%f %f,');
            end

            if(nv>0)
                if mod(nv,2) ~= 0
                  fclose(f);
                  error('Error reading 3d wire co-ordinates: should be x y z, on each line');
                end 
                npt = npt+1;
                out(npt,1:2)=fv'; 
            end                  
        end
    end

    function [out ,f] = read_mesh(f)
        out = [];
        tempstr = ' ';
        npt=0;
        coord_end = [];
        
        while tempstr~=-1
            tempstr = fgets(f);
            
            % Break when reaching end of element
            coord_end = strfind(tempstr, ']');            
            if ~isempty(coord_end)
                break;
            end
            
            sp = strfind(tempstr,'[');            
            if isempty(sp)
               %/* points data in x y z columns */
               [fv,nv]=sscanf(tempstr,'%f, %f, %f, %f,');
            else
               %/* if block start marker [ in line - need to skip over it to data 
               %   hence pointer to marker incremented */
               sp = sp +1;
               [fv,nv]=sscanf(tempstr(sp:length(tempstr)),'%f, %f, %f, %f,');
            end

            if(nv>0)
               if mod(nv,4) ~= 0
                  fclose(f);
                  error('Error reading 3d wire co-ordinates: should be x y z, on each line');
               end 
               nov = fix(nv/4);
               for p = 1:nov
                    npt = npt+1;
                    out(npt,1:3)=fv(1:3,:)'; 
               end
            end                  
        end
    end

    %/* start of main loop for reading file line by line */
    while ( tempstr ~= -1)
        tempstr = fgets(fp); % -1 if eof 

        for i=1:N  %/* check for each keyword in line */
            key = deblank(keynames(i,:));
            if ~isempty(findstr(tempstr,key)) 
               if ~foundkey(i), foundkey(i)=1;else foundkey(i)=0; end
            end
        end

        if(foundkey(1)) %/* start of if A  first 2 keys found */
            [coord, fp] = read_coordinates(fp); 
            foundkey(1) = 0;
        end

        if(foundkey(2))
            [text_coord, fp] = read_texture_coordinates(fp);
            foundkey(2) = 0;
        end
        
        %if(foundkey(3))
        %    [mesh, fp] = read_mesh(fp);
        %    foundkey(3) = 0;            
        %end
    end %/* end of main loop */
  
    w3d = [coord text_coord];
    
    fclose(fp);
end  
%  END OF FUNCTION read_vrml

%=====================================================================================
