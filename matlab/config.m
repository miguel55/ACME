% Configuration
addpath('tracking');
%% 1. Data configuration
% We need at least 2 channels: venule and cell
venule_channel=2;
cell_channels=1;
DEPTH=16;               % Depth of 3D segmentation network volumes
v_size=[170.75/256, 170.75/256, 2]; % Voxel size
n_classes=3;            % Background, cell and venule
% Group description
nG=4;                           % Number of groups
groups=1:nG;                    % Group numerical identifiers
groups_beh_discovery=[1,2];     % Groups for behavior discovery and number
nG_BD=length(unique(groups_beh_discovery));
group_def={'WT','antiPlt','FGR KO', 'FGR INH'};     % Group definitions: wild type, anti-Platelet, FGR knockout and FGR inhibitor
%% 2. Database
% Get capture sequences
sequences=dir('../data/annotation/group*/*/*.mat');
valids=true(length(sequences),1);
for j=1:length(sequences)
    aux=strsplit(sequences(j).folder,filesep);
    if (endsWith(aux{end},'labels'))
        valids(j)=0;
    end
end
sequences=sequences(valids);
sequences={sequences(:).folder};
sequences=natsortfiles(unique(sequences));

%% 3. 3D three-pass tracking
JPG=true;

%% 4. Feature extraction module
id_features='features';     % Identifier for feature files
INTERP=true;                % Interpolate volumes for unify voxel size (voxel size in 3rd dim = voxel size in 1st/2nd dim)
N_feat_inst=28;             % Number of instantaneous features            
N_feat_dyn=53;              % Number of dynamic features
range_orientation=5;        % Range of venule sections to estimate bloodstream direction
n_div_interp=100;           % Number of divisions to calculate cell height
n_pixels_interp=100;        % Number of pixels to calculate cell height
seed=0;                     % Seed to fix results
TRAJECTORY=21;              % Minimum length of trajectory
DIST=21*4;                  % Hanning window length

%% 5. Cell selection module
M=4;                                    % Number of folds
TH_score=0.5; % Threshold to consider a neutrophil as a well segmented one
TH_score_sel=0.6815;                    % 95 per cent of precision
id_label='labels';                      % Identifier for feature files
useful_features_dyn=1:52;               % Useful dynamic features 
useful_features_inst=[1:21 28 26];      % Useful instantaneous features 


%% 6. Hierarchical explainability
nK=10;                          % Maximum number of behaviors
% L1 classifier
FAITHFULNESS_LEVEL=0.985;       % Relative faithfulness level to 
EXPLICABILITY_LEVEL=1;          % How much importance we set (fix to 1 with FFFS)
FFFS=true;                      % Apply Feed Forward Feature Selection or not
feature_names=cell(74,1);       % Feature names
feature_names{1}='Number of temporal instants';
feature_names{2}='Traveled distance';
feature_names{3}='Maximum velocity';
feature_names{4}='Mean velocity';
feature_names{5}='Meander ratio';
feature_names{6}='Tortuosity';
feature_names{7}='Angle between trajectory and blood vessel';
feature_names{8}='Volume (mean)';
feature_names{9}='Superficial area (mean)';
feature_names{10}='Height with respect to blood vessel surface (mean)';
feature_names{11}='Maximum width (mean)';
feature_names{12}='Height/width ratio (mean)';
feature_names{13}='Sphericity (mean)';
feature_names{14}='Prolate ellipticity (mean)';
feature_names{15}='Oblate ellipticity (mean)';
feature_names{16}='Principal axis length (mean)';
feature_names{17}='Second principal axis length (mean)';
feature_names{18}='Third principal axis length (mean)';
feature_names{19}='Extent (mean)';
feature_names{20}='Solidity (mean)';
feature_names{21}='Equivalent diameter (mean)';
feature_names{22}='Volume (standard deviation)';
feature_names{23}='Superficial area (standard deviation)';
feature_names{24}='Height wrt blood vessel surface (standard deviation)';
feature_names{25}='Maximum width (standard deviation)';
feature_names{26}='Height/width ratio (standard deviation)';
feature_names{27}='Sphericity (standard deviation)';
feature_names{28}='Prolate ellipticity (standard deviation)';
feature_names{29}='Oblate ellipticity(standard deviation)';
feature_names{30}='Principal axis length (standard deviation)';
feature_names{31}='Second principal axis length (standard deviation)';
feature_names{32}='Third principal axis length (standard deviation)';
feature_names{33}='Extent (standard deviation)';
feature_names{34}='Solidity (standard deviation)';
feature_names{35}='Equivalent diameter (standard deviation)';
feature_names{36}='Polar radius (mean)';
feature_names{37}='Polar angle (mean)';
feature_names{38}='Polar radius (standard deviation)';
feature_names{39}='Polar angle (standard deviation)';
feature_names{40}='Orientation wrt the blood vessel in X (mean)';
feature_names{41}='Orientation wrt the blood vessel in Y (mean)';
feature_names{42}='Orientation wrt the blood vessel in Z (mean)';
feature_names{43}='Orientation wrt the blood vessel in X (standard deviation)';
feature_names{44}='Orientation wrt the blood vessel in Y (standard deviation)';
feature_names{45}='Orientation wrt the blood vessel in Z (standard deviation)';
feature_names{46}='Distance from cell center to blood vessel surface (mean)';
feature_names{47}='Minimum distance from cell to blood vessel surface (mean)';
feature_names{48}='Minimum distance from trajectory to blood vessel surface';
feature_names{49}='Well_segmented cell probability (mean)';
feature_names{50}='CNN maximum score';
feature_names{51}='CNN minimum score';
feature_names{52}='CNN score range';
feature_names{53}='Volume';
feature_names{54}='Superficial area';
feature_names{55}='Height wrt blood vessel surface';
feature_names{56}='Maximum width';
feature_names{57}='Height/width ratio';
feature_names{58}='Sphericity';
feature_names{59}='Prolate ellipticity';
feature_names{60}='Oblate ellipticity';
feature_names{61}='Principal axis length';
feature_names{62}='Second principal axis length';
feature_names{63}='Third principal axis length';
feature_names{64}='Extent';
feature_names{65}='Solidity';
feature_names{66}='Equivalent diameter';
feature_names{67}='Orientation wrt the blood vessel in X';
feature_names{68}='Orientation wrt the blood vessel in Y';
feature_names{69}='Orientation wrt the blood vessel in Z';
feature_names{70}='Polar radius';
feature_names{71}='Polar angle';
feature_names{72}='Cell center distance to the blood vessel surface';
feature_names{73}='Minimum distance to the blood vessel surface';
feature_names{74}='Well_segmented cell probability';
