clear, close all; 
config;
load('models/cell_selection_module.mat');
% Classify the rest of the samples
d=dir(['results/*/*/',id_features,'.mat']);
for i=1:length(d)
    % Load instantaneous and dynamic features
    features_simple=load([d(i).folder '/' d(i).name]);
    features_simple=features_simple.features_simple;
    features_track=load([d(i).folder '/' strrep(d(i).name,id_features,[id_features,'_track'])]);
    features_track=features_track.features_track;
    % Get scores
    scores=features_simple(:,useful_features_inst(1:length(useful_features_inst)-1));
    % Generate the feature vector
    features_aux=[features_track(:,useful_features_dyn(1:length(useful_features_dyn))) scores];
    % Trajectory-length filtering
    ids_init=find(features_aux(:,1)>TRAJECTORY);
    features_aux=features_aux(ids_init,:);
    % Low confidence position filtering and volume filtering
    ids=sum(isnan(features_aux) | isinf(features_aux),2)==0 & features_aux(:,53)<3060 & features_aux(:,53)>110; %& isout==0;
    features_aux=features_aux(ids,:);
    
    % Ensemble of four classifiers: cell selection
    scores=zeros(size(features_aux,1),1);
    for j=1:M
        features_fold=features_aux;
        features_fold=(features_fold-repmat(mus{j},[size(features_fold,1) 1]))./repmat(stds{j},[size(features_fold,1) 1]);
        [y_test,score_test] = predict(classifiers{j},features_fold);
        scores=scores+score_test(:,2);
    end
    scores=scores/M;
    
    % Get and save the labels for each cell instance
    labels2=scores>TH_score_sel;
    labels_aux=zeros(size(ids,1),1);
    labels_aux(ids)=labels2;
    labels=false(size(features_simple,1),1);
    labels(ids_init)=labels_aux;
    save([d(i).folder filesep id_label '.mat'],'labels');
    disp(i);
end
