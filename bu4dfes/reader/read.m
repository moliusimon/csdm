function [] = read(path2data, path2save)
    % Define reader class
    bu4dfeReader = BU4DFEReader(path2data, path2save);

    % Read and save
    bu4dfeReader.Read();
end


