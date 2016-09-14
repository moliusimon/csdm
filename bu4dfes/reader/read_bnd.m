function [out] = read_bnd(filename)

    fp = fopen(filename,'r');
    if fp == -1
        fclose all;
        str = sprintf('Cannot open file %s \n',filename);
        errordlg(str);
        error(str);
    end
    
    tempstr = ' ';
    
    ln = 1;
    while true
        tempstr = fgets(fp); % -1 if eof 
        
        if tempstr== -1
            break;
        end
        
        [fv,~]=sscanf(tempstr,'%f %f %f %f,');
        
        out(ln,:) = fv';
        ln = ln+1;
    end
end

