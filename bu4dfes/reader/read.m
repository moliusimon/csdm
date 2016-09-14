% Main reading script for BU4DFE

path2data = '/Users/cipriancorneanu/Research/BU4DFE';
path2save = '/Users/cipriancorneanu/Research/BU4DFE';

% Define reader class
bu4dfeReader = BU4DFEReader(path2data, path2save);

% Read and save
bu4dfeReader.Read();


